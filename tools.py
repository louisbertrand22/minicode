"""Les outils de l'agent.

Un outil, c'est deux choses :
  1. un SCHEMA (nom + description + JSON Schema des paramètres) qu'on envoie au
     modèle pour qu'il sache que l'outil existe et comment l'appeler ;
  2. une FONCTION Python que *notre* code exécute quand le modèle le demande.

Le modèle n'exécute jamais rien lui-même : il renvoie un bloc `tool_use`
("je voudrais appeler read_file avec path=..."), et c'est le harness qui
décide de l'exécuter et de renvoyer le résultat.
"""

import re
import subprocess
from pathlib import Path

# Racine du projet sur lequel l'agent travaille. Les outils de fichiers refusent
# d'en sortir. `bash`, lui, peut tout faire : d'où la permission demandée avant.
WORKSPACE = Path.cwd().resolve()

MAX_OUTPUT_CHARS = 20_000  # on ne met pas un fichier de 5 Mo dans le contexte
MAX_GREP_MATCHES = 100
BASH_TIMEOUT = 60  # secondes ; une commande bloquée ne doit pas figer l'agent
IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "target", ".claude"}

# Outils qui modifient le disque ou exécutent du code : le harness demande la
# permission à l'utilisateur avant de les lancer (voir minicode.py).
DANGEROUS_TOOLS = {"edit_file", "bash"}


class ToolError(Exception):
    """Erreur renvoyée au modèle (is_error=True) au lieu de faire planter le harness."""


def _resolve(path: str) -> Path:
    p = (WORKSPACE / path).resolve()
    if not p.is_relative_to(WORKSPACE):
        raise ToolError(f"Accès refusé : {path} est en dehors du workspace {WORKSPACE}")
    return p


def read_file(path: str) -> str:
    p = _resolve(path)
    if not p.is_file():
        raise ToolError(f"Fichier introuvable : {path}")
    text = p.read_text(errors="replace")
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + f"\n... [tronqué, {len(text)} caractères au total]"
    # Numéroter les lignes aide le modèle à citer / éditer précisément.
    return "\n".join(f"{i:>5}\t{line}" for i, line in enumerate(text.splitlines(), 1))


def list_dir(path: str = ".") -> str:
    p = _resolve(path)
    if not p.is_dir():
        raise ToolError(f"Dossier introuvable : {path}")
    entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
    lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries if e.name != ".git"]
    return "\n".join(lines) or "(dossier vide)"


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n... [tronqué, {len(text)} caractères au total]"
    return text


def grep(pattern: str, path: str) -> str:
    """Cherche une regex dans les fichiers texte. Évite de lire tout le projet."""
    root = _resolve(path)
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"Regex invalide : {e}")
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    matches = []
    for f in files:
        if not f.is_file() or IGNORED_DIRS.intersection(f.relative_to(WORKSPACE).parts):
            continue
        try:
            lines = f.read_text().splitlines()
        except (UnicodeDecodeError, OSError):
            continue  # fichier binaire ou illisible
        for i, line in enumerate(lines, 1):
            if regex.search(line):
                matches.append(f"{f.relative_to(WORKSPACE)}:{i}: {line.strip()}")
                if len(matches) >= MAX_GREP_MATCHES:
                    return "\n".join(matches) + f"\n... [arrêté à {MAX_GREP_MATCHES} résultats, affine ta recherche]"
    return "\n".join(matches) or "Aucun résultat."


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Remplace un texte EXACT par un autre. old_string vide = créer un nouveau fichier.

    Pourquoi pas "réécrire tout le fichier" ? Parce que le modèle devrait alors
    recopier des centaines de lignes sans erreur (lent, cher, risqué). Avec un
    remplacement exact, il n'écrit que ce qui change, et si old_string ne
    correspond pas, on le sait tout de suite au lieu d'abîmer le fichier.
    """
    p = _resolve(path)
    if old_string == "":
        if p.exists():
            raise ToolError(f"{path} existe déjà : donne un old_string pour le modifier.")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(new_string)
        return f"Fichier créé : {path} ({len(new_string.splitlines())} lignes)"
    if not p.is_file():
        raise ToolError(f"Fichier introuvable : {path}")
    text = p.read_text()
    count = text.count(old_string)
    if count == 0:
        raise ToolError(
            "old_string introuvable dans le fichier. Relis le fichier avec read_file et "
            "recopie le texte exactement (espaces et indentation compris, sans les numéros de ligne)."
        )
    if count > 1:
        raise ToolError(f"old_string apparaît {count} fois : ajoute des lignes autour pour qu'il soit unique.")
    p.write_text(text.replace(old_string, new_string, 1))
    return f"Modifié : {path}"


def bash(command: str) -> str:
    """Lance une commande shell dans le dossier du projet."""
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKSPACE, capture_output=True, text=True,
            timeout=BASH_TIMEOUT,
            stdin=subprocess.DEVNULL,  # une commande qui attend une saisie ne bloque pas l'agent
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"Commande arrêtée après {BASH_TIMEOUT} s (trop longue ou bloquée).")
    output = (r.stdout + r.stderr).strip() or "(aucune sortie)"
    # Le code de sortie est crucial : c'est grâce à lui que le modèle sait si
    # ses tests passent ou non.
    return _truncate(f"{output}\n[code de sortie : {r.returncode}]")


# --- Ce que le modèle voit --------------------------------------------------
# La qualité des descriptions change énormément le comportement du modèle :
# c'est du "prompt engineering" appliqué aux outils.
TOOL_SCHEMAS = [
    {
        "name": "read_file",
        "description": (
            "Lit un fichier texte du projet et renvoie son contenu avec les numéros de ligne. "
            "Le chemin est relatif à la racine du projet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Chemin relatif, ex: src/main.py"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "list_dir",
        "description": "Liste les fichiers et sous-dossiers d'un dossier du projet (les dossiers finissent par /).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Chemin relatif, '.' pour la racine"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "grep",
        "description": (
            "Cherche une expression régulière (syntaxe Python) dans les fichiers du projet et renvoie "
            "les lignes trouvées sous la forme fichier:ligne: texte. Utilise-le pour trouver où une "
            "fonction ou un mot est défini/utilisé, au lieu de lire tous les fichiers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": r"Regex, ex: def run_tool ou TODO|FIXME"},
                "path": {"type": "string", "description": "Dossier ou fichier où chercher, '.' pour tout le projet"},
            },
            "required": ["pattern", "path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "edit_file",
        "description": (
            "Modifie un fichier en remplaçant old_string par new_string. old_string doit être recopié "
            "EXACTEMENT depuis le fichier (indentation comprise, sans les numéros de ligne de read_file) "
            "et n'apparaître qu'une seule fois. Lis toujours le fichier avant de le modifier. "
            "Pour créer un nouveau fichier : old_string vide et tout le contenu dans new_string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Chemin relatif du fichier"},
                "old_string": {"type": "string", "description": "Texte exact à remplacer ('' pour créer un fichier)"},
                "new_string": {"type": "string", "description": "Nouveau texte"},
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "bash",
        "description": (
            "Exécute une commande shell dans le dossier du projet et renvoie sa sortie et son code de "
            f"sortie (0 = succès). Limite : {BASH_TIMEOUT} s, pas de saisie clavier possible. "
            "Utile pour lancer les tests (ex: uv run pytest) et vérifier ton travail après une modification."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "La commande, ex: uv run pytest -q"}},
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

# --- Ce que le harness exécute ----------------------------------------------
TOOL_FUNCTIONS = {
    "read_file": read_file,
    "list_dir": list_dir,
    "grep": grep,
    "edit_file": edit_file,
    "bash": bash,
}


def run_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Exécute un outil. Renvoie (résultat, is_error)."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Outil inconnu : {name}", True
    try:
        return fn(**tool_input), False
    except ToolError as e:
        return str(e), True
    except Exception as e:  # un bug dans un outil ne doit pas tuer la session
        return f"{type(e).__name__}: {e}", True

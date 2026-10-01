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
IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", "target", ".claude", ".minicode"}
# Dossier des réglages de minicode (règles de permission). L'agent ne doit pas
# pouvoir y écrire, sinon il pourrait s'autoriser lui-même n'importe quoi.
PROTECTED_DIR = ".minicode"

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
    if not text.strip():
        # Un résultat vide est ambigu pour le modèle ; on le dit explicitement.
        return "(fichier vide : pour l'écrire, utilise edit_file avec old_string='')"
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


def plan_edit(path: str, old_string: str, new_string: str) -> tuple[Path, str, str]:
    """Prépare un edit_file SANS rien écrire : renvoie (fichier, nouveau contenu, message).

    Lève ToolError si la modification ne peut pas marcher. Le harness l'appelle
    AVANT de demander la permission : inutile de déranger l'utilisateur pour une
    modification qui va échouer de toute façon.
    """
    p = _resolve(path)
    if p.is_relative_to(WORKSPACE / PROTECTED_DIR):
        raise ToolError(f"{PROTECTED_DIR}/ est protégé : seul l'utilisateur peut modifier les réglages de minicode.")
    if old_string == "":
        # Un fichier existant mais VIDE compte comme "à créer" : sinon aucun
        # old_string ne peut jamais y être trouvé et le modèle reste bloqué.
        if p.exists() and p.read_text().strip():
            raise ToolError(f"{path} existe déjà et n'est pas vide : lis-le, puis donne un old_string pour le modifier "
                            "(pour ajouter au début, old_string = la 1re ligne actuelle, new_string = ajout + cette ligne).")
        return p, new_string, f"Fichier créé : {path} ({len(new_string.splitlines())} lignes)"
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
    return p, text.replace(old_string, new_string, 1), f"Modifié : {path}"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Remplace un texte EXACT par un autre. old_string vide = créer un nouveau fichier.

    Pourquoi pas "réécrire tout le fichier" ? Parce que le modèle devrait alors
    recopier des centaines de lignes sans erreur (lent, cher, risqué). Avec un
    remplacement exact, il n'écrit que ce qui change, et si old_string ne
    correspond pas, on le sait tout de suite au lieu d'abîmer le fichier.
    """
    p, content, message = plan_edit(path, old_string, new_string)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return message


def bash(command: str, stdin: str | None = None) -> str:
    """Lance une commande shell dans le dossier du projet.

    Pas de clavier : un agent ne peut pas taper au milieu d'une commande. Si
    `stdin` est donné, ce texte est envoyé au programme comme si on le tapait
    (une ligne par saisie) ; sinon le programme reçoit "fin de saisie" tout de
    suite, au lieu de bloquer l'agent indéfiniment.
    """
    feed = {"input": stdin} if stdin is not None else {"stdin": subprocess.DEVNULL}
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKSPACE, capture_output=True, text=True,
            timeout=BASH_TIMEOUT, **feed,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"Commande arrêtée après {BASH_TIMEOUT} s (trop longue ou bloquée).")
    output = (r.stdout + r.stderr).strip() or "(aucune sortie)"
    if "EOFError" in output:
        # On aide le modèle à comprendre l'erreur, au lieu de le laisser deviner
        # (sans indice, il concluait que le programme était buggé).
        if stdin is None:
            output += ("\n[indice minicode : ce programme attend des saisies au clavier. Relance la "
                       "commande avec le paramètre stdin (une ligne par saisie).]")
        else:
            output += (f"\n[indice minicode : le programme a demandé plus de saisies que les "
                       f"{len(stdin.splitlines())} lignes fournies dans stdin. Ce n'est pas un bug du "
                       "programme. Relance TOI-MÊME bash avec assez de lignes pour aller jusqu'au bout "
                       "(par exemple, pour un nombre à deviner entre 1 et 100 : les 100 valeurs, une par ligne).]")
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
            f"sortie (0 = succès). Limite : {BASH_TIMEOUT} s. Utile pour lancer les tests (ex: uv run pytest) "
            "et vérifier ton travail après une modification. Il n'y a pas de clavier : pour tester un "
            "programme qui pose des questions (input()), donne les réponses dans stdin."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "La commande, ex: uv run pytest -q"},
                # Pas d'exemple concret ici : les petits modèles RECOPIENT les exemples
                # (ils envoyaient "50\n75\n62\n" à n'importe quel programme).
                "stdin": {
                    "type": "string",
                    "description": "Optionnel : texte envoyé au programme comme s'il était tapé au clavier, "
                                   "une ligne par saisie. Prévois une ligne pour CHAQUE question que posera "
                                   "le programme, sinon il s'arrête avec EOFError.",
                },
            },
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


SCHEMAS_BY_NAME = {s["name"]: s["input_schema"] for s in TOOL_SCHEMAS}


def validate_input(name: str, tool_input) -> None:
    """Vérifie les arguments AVANT d'exécuter : ne jamais faire confiance à ce que
    renvoie le modèle (un petit modèle se trompe, et en streaming un JSON peut être tronqué)."""
    schema = SCHEMAS_BY_NAME[name]
    if not isinstance(tool_input, dict):
        raise ToolError(f"Arguments invalides pour {name} : un objet JSON est attendu.")
    unknown = set(tool_input) - set(schema["properties"])
    if unknown:
        raise ToolError(f"Arguments inconnus pour {name} : {', '.join(sorted(unknown))}")
    for key in schema["required"]:
        if not isinstance(tool_input.get(key), str):
            raise ToolError(f"Argument manquant ou invalide pour {name} : {key} (texte attendu)")
    for key, value in tool_input.items():  # les arguments optionnels aussi doivent être du texte
        if not isinstance(value, str):
            raise ToolError(f"Argument invalide pour {name} : {key} (texte attendu)")


def precheck(name: str, tool_input) -> str | None:
    """Vérifie un appel AVANT de demander la permission. Renvoie l'erreur, ou None si l'appel peut marcher."""
    if name not in TOOL_FUNCTIONS:
        return f"Outil inconnu : {name}"
    try:
        validate_input(name, tool_input)
        if name == "edit_file":
            plan_edit(**tool_input)
    except ToolError as e:
        return str(e)
    return None


def run_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Exécute un outil. Renvoie (résultat, is_error)."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Outil inconnu : {name}", True
    try:
        validate_input(name, tool_input)
        return fn(**tool_input), False
    except ToolError as e:
        return str(e), True
    except Exception as e:  # un bug dans un outil ne doit pas tuer la session
        return f"{type(e).__name__}: {e}", True

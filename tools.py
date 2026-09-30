"""Les outils de l'agent.

Un outil, c'est deux choses :
  1. un SCHEMA (nom + description + JSON Schema des paramètres) qu'on envoie au
     modèle pour qu'il sache que l'outil existe et comment l'appeler ;
  2. une FONCTION Python que *notre* code exécute quand le modèle le demande.

Le modèle n'exécute jamais rien lui-même : il renvoie un bloc `tool_use`
("je voudrais appeler read_file avec path=..."), et c'est le harness qui
décide de l'exécuter et de renvoyer le résultat.
"""

from pathlib import Path

# Racine du projet sur lequel l'agent travaille. Les outils refusent d'en
# sortir : c'est un premier garde-fou (l'étape 5 ajoutera les permissions).
WORKSPACE = Path.cwd().resolve()

MAX_OUTPUT_CHARS = 20_000  # on ne met pas un fichier de 5 Mo dans le contexte


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
]

# --- Ce que le harness exécute ----------------------------------------------
TOOL_FUNCTIONS = {
    "read_file": read_file,
    "list_dir": list_dir,
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

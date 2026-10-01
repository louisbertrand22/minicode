"""Étape 5 : les règles de permission (allow / deny), façon Claude Code.

Fichier `.minicode/permissions.json` à la racine du projet :

    {
      "allow": ["bash(uv run pytest*)", "edit_file(src/*)"],
      "deny":  ["bash(rm -rf*)", "bash(git push*)"]
    }

Une règle s'écrit `outil(motif)`. Le motif (syntaxe glob : * = n'importe quoi)
est comparé à la commande pour `bash`, au chemin pour `edit_file`.

Ordre de décision :  deny  >  commande composée (toujours demander)  >  allow  >  demander.
"""

import fnmatch
import glob
import json
import re

import tools

# `uv run pytest*` autoriserait aussi `uv run pytest; rm -rf ~` ! Une commande qui
# enchaîne, redirige ou imbrique d'autres commandes n'est donc JAMAIS acceptée
# automatiquement par une règle allow : on demande toujours.
SHELL_META = re.compile(r"[;&|`<>\n]|\$\(")

RULE = re.compile(r"^(\w+)\((.*)\)$")


def rules_file():
    return tools.WORKSPACE / tools.PROTECTED_DIR / "permissions.json"


def load_rules():
    try:
        data = json.loads(rules_file().read_text())
    except FileNotFoundError:
        return {"allow": [], "deny": []}
    return {"allow": data.get("allow", []), "deny": data.get("deny", [])}


def add_allow_rule(rule):
    rules = load_rules()
    if rule not in rules["allow"]:
        rules["allow"].append(rule)
    rules_file().parent.mkdir(exist_ok=True)
    rules_file().write_text(json.dumps(rules, indent=2, ensure_ascii=False) + "\n")


def _target(name, tool_input):
    """Ce à quoi on compare le motif d'une règle."""
    return tool_input["command"] if name == "bash" else tool_input.get("path", "")


def matches(rule, name, tool_input):
    m = RULE.match(rule)
    return bool(m) and m.group(1) == name and fnmatch.fnmatchcase(_target(name, tool_input), m.group(2))


def decide(name, tool_input, rules=None):
    """Renvoie (décision, raison) avec décision ∈ {"allow", "deny", "ask"}."""
    rules = load_rules() if rules is None else rules
    for rule in rules["deny"]:
        if matches(rule, name, tool_input):
            return "deny", rule
    if name == "bash" and SHELL_META.search(tool_input["command"]):
        return "ask", "commande composée : toujours demander"
    for rule in rules["allow"]:
        if matches(rule, name, tool_input):
            return "allow", rule
    return "ask", None


def suggest_rule(name, tool_input):
    """La règle proposée quand l'utilisateur répond « toujours » : la plus étroite possible.

    glob.escape : si la commande contient elle-même * ou [, ils doivent compter comme
    du texte, pas comme des jokers (sinon « toujours » autoriserait bien plus que prévu).
    """
    return f"{name}({glob.escape(_target(name, tool_input))})"


def can_remember(name, tool_input):
    """On ne propose pas « toujours » pour une commande composée (elle redemanderait quand même)."""
    return not (name == "bash" and SHELL_META.search(tool_input["command"]))

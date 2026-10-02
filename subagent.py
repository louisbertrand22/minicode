"""ÉTAPE 9 : les sous-agents (l'outil `task`).

Problème : pour répondre à « où est géré X dans ce projet ? », l'agent lit 5 fichiers.
Ces 5 fichiers restent ensuite dans SON historique, et la fenêtre (8k tokens) se remplit
de texte dont il n'a plus besoin (l'étape 8 doit alors effacer ou résumer).

Idée : déléguer. L'outil `task` lance une DEUXIÈME boucle d'agent (la même que celle
de l'étape 3), avec :
  - un historique VIDE : juste la consigne écrite par l'agent principal ;
  - son propre prompt système, et seulement des outils de LECTURE ;
  - un nombre d'étapes limité.
Le sous-agent lit autant qu'il veut ; à la fin, seul son RAPPORT (quelques lignes)
revient à l'agent principal, comme résultat de l'outil `task`. Tout le reste est jeté.

Ce n'est pas un autre modèle ni un autre programme : même modèle, même client, même
code de boucle. Un « sous-agent », c'est juste une autre liste `messages`.

Pourquoi seulement la lecture ? Le sous-agent travaille sans que l'utilisateur voie
ses demandes de permission dans leur contexte, et l'agent principal ne voit pas ce
qu'il fait : on limite les dégâts possibles. Et pas de `task` pour lui : pas de
sous-sous-agents à l'infini.
"""

READ_ONLY_TOOLS = ("read_file", "list_dir", "grep")
MAX_STEPS = 12  # appels au modèle pour UN sous-agent

TASK_SCHEMA = {
    "name": "task",
    "description": (
        "Délègue une RECHERCHE à un sous-agent : il part d'une conversation vide, lit les fichiers "
        "qu'il veut (read_file, list_dir, grep ; il ne peut rien modifier) et te renvoie seulement "
        "un rapport court. Utilise-le quand il faut parcourir plusieurs fichiers pour trouver une "
        "information : ton propre contexte reste petit. Le sous-agent ne voit PAS ta conversation : "
        "écris dans prompt tout ce qu'il doit savoir et ce que tu attends dans son rapport."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Titre court (3-6 mots), ex: Trouver la gestion des permissions"},
            "prompt": {"type": "string", "description": "La consigne complète pour le sous-agent"},
        },
        "required": ["description", "prompt"],
        "additionalProperties": False,
    },
    "strict": True,
}

SYSTEM_PROMPT = """Tu es un sous-agent de minicode : un autre agent t'a confié une recherche dans le projet situé à : {workspace}
Tu ne peux que LIRE : read_file, list_dir, grep. Commence par grep ou list_dir pour savoir où chercher,
puis lis les passages utiles. Ne devine jamais le contenu d'un fichier.

Quand tu as trouvé, réponds par un RAPPORT court (moins de 200 mots), en français :
- la réponse à la question ;
- les fichiers et numéros de ligne concernés ;
- les extraits de code vraiment utiles (quelques lignes, pas des fichiers entiers).
L'agent qui t'a confié la tâche ne voit que ce rapport."""

FINISH_NOW = "Limite d'étapes atteinte : arrête de chercher et écris ton rapport final MAINTENANT, sans outil."


def validate(tool_input) -> str | None:
    """Vérifie les arguments de `task` (même principe que tools.validate_input)."""
    if not isinstance(tool_input, dict) or set(tool_input) - {"description", "prompt"}:
        return "Arguments invalides pour task : description et prompt attendus."
    if not isinstance(tool_input.get("prompt"), str) or not tool_input["prompt"].strip():
        return "Argument manquant pour task : prompt (la consigne du sous-agent)."
    return None

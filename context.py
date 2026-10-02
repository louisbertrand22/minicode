"""ÉTAPE 8 : la gestion du contexte.

Le modèle n'a pas de mémoire : à chaque appel on lui renvoie le prompt système, les
outils et TOUT l'historique (`messages`). Ce paquet doit tenir dans sa FENÊTRE DE
CONTEXTE (8 192 tokens avec notre réglage Ollama), réponse comprise. Quand ça
déborde, Ollama coupe le début en silence : le modèle « oublie » sans prévenir.

Le harness surveille donc la taille, et dès qu'elle dépasse un seuil il fait de la
place, du moins cher au plus cher :
  1. plafonner chaque résultat d'outil (voir output_limit et read_file/start_line) ;
  2. EFFACER LES VIEUX RÉSULTATS D'OUTILS : le modèle en a déjà tiré ce qu'il
     fallait. On garde le tool_use (ce qu'il a fait) et on remplace seulement le
     contenu du tool_result. Gratuit : aucun appel au modèle ;
  3. RÉSUMER LES VIEUX TOURS : un appel au modèle écrit un résumé, qui remplace
     le début de la conversation (c'est le /compact de Claude Code).

Ce fichier ne contient que des fonctions sur la liste `messages` ; l'appel au
modèle pour le résumé est fait par minicode.py.
"""

import json
import os

from tracelog import to_jsonable

# Un token ≈ 4 caractères en anglais, plutôt 3 en français ou en code. On prend 3 :
# mieux vaut SURESTIMER (compacter un peu tôt) que déborder sans le savoir.
CHARS_PER_TOKEN = 3.0
# On fait de la place au-delà de 70 % de la fenêtre : les 30 % restants sont pour la
# réponse du modèle (et sa réflexion), qui doit tenir dans la même fenêtre.
COMPACT_AT = 0.70
KEEP_TOOL_RESULTS = 2    # les N derniers messages de résultats d'outils restent intacts
CLEARED = "[résultat effacé par minicode pour libérer du contexte ; relance l'outil si tu en as encore besoin]"
SUMMARY_PREFIX = "[Résumé de la conversation précédente, écrit pour libérer du contexte]\n"
SUMMARY_ACK = "Compris. Je continue à partir de ce résumé."

SUMMARY_INSTRUCTIONS = """Voici le début d'une conversation entre un utilisateur et minicode (un agent de code).
Elle est devenue trop longue : écris un résumé qui la REMPLACERA, pour continuer le travail sans elle.

Garde, en puces courtes :
- ce que l'utilisateur a demandé, et ses préférences ou consignes ;
- les fichiers lus ou modifiés (chemins exacts) et ce qu'on y a appris d'important ;
- les modifications faites et leur résultat (tests, erreurs) ;
- ce qui reste à faire, et les erreurs à ne pas refaire.
N'invente rien. Moins de 250 mots. Réponds seulement avec le résumé, en français.

Conversation :
"""


def window_for(provider: str) -> int:
    """Taille de la fenêtre de contexte, en tokens."""
    if os.environ.get("MINICODE_CONTEXT_WINDOW"):
        return int(os.environ["MINICODE_CONTEXT_WINDOW"])
    if provider == "ollama":
        # Doit correspondre au OLLAMA_CONTEXT_LENGTH donné à `ollama serve` (8192 dans le README).
        return int(os.environ.get("OLLAMA_CONTEXT_LENGTH", 8192))
    return 200_000


def output_limit(window: int) -> int:
    """Plafond (en caractères) d'UN résultat d'outil : au plus ~20 % de la fenêtre.

    Avec 8k tokens, un fichier de 20 000 caractères (≈ 6 000 tokens) remplirait tout.
    """
    return max(2_000, min(20_000, int(window * 0.20 * CHARS_PER_TOKEN)))


def estimate_tokens(obj) -> int:
    """Estimation grossière : le nombre de caractères du JSON envoyé, divisé par 3."""
    return int(len(json.dumps(to_jsonable(obj), ensure_ascii=False, default=str)) / CHARS_PER_TOKEN)


class Budget:
    """Combien de tokens on enverrait au prochain appel, et le seuil à ne pas dépasser.

    L'estimation par caractères est imprécise. Mais après chaque appel le serveur
    nous dit le VRAI nombre de tokens (usage) : on en déduit un facteur de correction
    (`ratio`), utilisé pour les estimations suivantes.
    """

    def __init__(self, window: int, system: str, tools: list):
        self.window = window
        self.limit = int(window * COMPACT_AT)
        self.set_fixed(system, tools)
        self.ratio = 1.0

    def set_fixed(self, system: str, tools: list):
        """Ce qui est envoyé à chaque appel (change si AGENTS.md change, étape 6)."""
        self.fixed = estimate_tokens(system) + estimate_tokens(tools)

    def raw(self, messages) -> int:
        return self.fixed + estimate_tokens(messages)

    def estimate(self, messages) -> int:
        return int(self.raw(messages) * self.ratio)

    def calibrate(self, messages, real_tokens: int):
        """Appelé après un appel au modèle, avec le nombre de tokens réellement lus."""
        if real_tokens > 0:
            # Bornes : une mesure bizarre (serveur qui ne compte pas tout) ne doit pas tout fausser.
            self.ratio = min(1.5, max(0.6, real_tokens / max(1, self.raw(messages))))

    def over(self, messages) -> bool:
        return self.estimate(messages) > self.limit


def _blocks(message):
    content = message["content"]
    return content if isinstance(content, list) else []


def _field(block, name, default=None):
    """Les blocs sont des dict (ceux qu'on écrit) ou des objets du SDK (réponses du modèle)."""
    return block.get(name, default) if isinstance(block, dict) else getattr(block, name, default)


def _has_tool_results(message) -> bool:
    return message["role"] == "user" and any(_field(b, "type") == "tool_result" for b in _blocks(message))


def clear_old_tool_results(messages, keep=KEEP_TOOL_RESULTS) -> int:
    """Remplace le contenu des vieux tool_result par un court texte. Renvoie le nombre effacé.

    Les `keep` derniers messages de résultats restent intacts (le modèle travaille
    probablement dessus). On crée de NOUVEAUX dict au lieu de modifier les anciens :
    une copie de la liste faite avant (pour annuler une demande) reste intacte.
    """
    indexes = [i for i, m in enumerate(messages) if _has_tool_results(m)]
    cleared = 0
    for i in indexes[:len(indexes) - keep] if keep else indexes:
        new_blocks = []
        for block in messages[i]["content"]:
            if block.get("type") == "tool_result" and block.get("content") != CLEARED \
                    and len(str(block.get("content", ""))) > len(CLEARED):
                block = {**block, "content": CLEARED}
                cleared += 1
            new_blocks.append(block)
        messages[i] = {**messages[i], "content": new_blocks}
    return cleared


def _short(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def transcript(messages, max_chars: int) -> str:
    """La conversation en texte simple, pour le modèle qui va la résumer.

    La réflexion (thinking) est ignorée, les résultats d'outils raccourcis. Si c'est
    encore trop long, on garde la FIN (le plus récent) — et un ancien résumé éventuel.
    """
    lines = []
    for message in messages:
        if isinstance(message["content"], str):
            who = "Utilisateur" if message["role"] == "user" else "minicode"
            lines.append(f"{who} : {_short(message['content'], 3000)}")
            continue
        for block in message["content"]:
            kind = _field(block, "type")
            if kind == "text" and _field(block, "text", "").strip():
                who = "Utilisateur" if message["role"] == "user" else "minicode"
                lines.append(f"{who} : {_short(_field(block, 'text'), 3000)}")
            elif kind == "tool_use":
                args = json.dumps(_field(block, "input", {}), ensure_ascii=False)
                lines.append(f"minicode appelle {_field(block, 'name')} {_short(args, 300)}")
            elif kind == "tool_result":
                error = " (erreur)" if _field(block, "is_error") else ""
                lines.append(f"  résultat{error} : {_short(_field(block, 'content', ''), 400)}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    first = messages[0]["content"]
    old_summary = isinstance(first, str) and first.startswith(SUMMARY_PREFIX)
    head = lines[0] + "\n[…]\n" if old_summary else "[…]\n"
    return head + text[-(max_chars - len(head)):]


def summary_messages(summary: str):
    """Les 2 messages qui remplacent la partie résumée (user puis assistant : les rôles alternent)."""
    return [
        {"role": "user", "content": SUMMARY_PREFIX + summary.strip()},
        {"role": "assistant", "content": SUMMARY_ACK},
    ]


def is_summary(messages) -> bool:
    """Ces messages sont-ils déjà juste un résumé ? (le résumer encore ne libérerait rien)"""
    first = messages[0]["content"] if messages else None
    return len(messages) == 2 and isinstance(first, str) and first.startswith(SUMMARY_PREFIX)


def fallback_summary(messages) -> str:
    """Résumé SANS modèle, si l'appel de résumé échoue : les demandes et les fichiers touchés."""
    requests, files = [], []
    for message in messages:
        if message["role"] == "user" and isinstance(message["content"], str):
            requests.append(_short(message["content"].removeprefix(SUMMARY_PREFIX), 300))
        for block in _blocks(message):
            if _field(block, "type") == "tool_use":
                path = (_field(block, "input") or {}).get("path")
                if path and path not in files:
                    files.append(path)
    text = "Demandes de l'utilisateur :\n" + "\n".join(f"- {r}" for r in requests[-10:])
    if files:
        text += "\nFichiers utilisés : " + ", ".join(files[-20:])
    return text


def breakdown(messages):
    """Où partent les tokens (estimation), pour la commande /context."""
    parts = {"demandes et réponses": 0, "appels d'outils": 0, "résultats d'outils": 0, "réflexion": 0}
    for message in messages:
        if isinstance(message["content"], str):
            parts["demandes et réponses"] += estimate_tokens(message["content"])
            continue
        for block in message["content"]:
            kind = _field(block, "type")
            key = {"tool_use": "appels d'outils", "tool_result": "résultats d'outils",
                   "thinking": "réflexion", "redacted_thinking": "réflexion"}.get(kind, "demandes et réponses")
            parts[key] += estimate_tokens(block)
    return parts

"""minicode : un mini agent de code, écrit à la main pour comprendre un harness.

Étapes couvertes ici :
  1. REPL de chat  -> le modèle n'a PAS de mémoire : on renvoie tout l'historique
                      (`messages`) à chaque appel.
  2. Un outil      -> le modèle *demande* un appel (bloc tool_use), on l'exécute,
                      on renvoie un bloc tool_result.
  3. Boucle agent  -> on rappelle le modèle tant qu'il demande des outils.
  4. Outils qui AGISSENT : grep, edit_file, bash (l'agent peut modifier et vérifier).
  5. Permissions (début) -> le harness demande "o/N" avant edit_file et bash.

Lancer :  uv run minicode.py      (dans le dossier du projet à explorer)

Deux "fournisseurs" de modèle, même boucle :
  - ollama    (défaut, gratuit) : un modèle qui tourne sur ta machine. Ollama
                parle le même format que l'API Anthropic, donc on garde le même
                SDK en changeant juste l'adresse du serveur.
  - anthropic (payant) : les modèles Claude, via ANTHROPIC_API_KEY.
"""

import os
import sys

import anthropic

from tools import DANGEROUS_TOOLS, TOOL_SCHEMAS, WORKSPACE, run_tool

PROVIDER = os.environ.get("MINICODE_PROVIDER", "ollama")  # ollama | anthropic
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODELS = {"ollama": "qwen3:8b", "anthropic": "claude-opus-5-5"}
MODEL = os.environ.get("MINICODE_MODEL", DEFAULT_MODELS[PROVIDER])
EFFORT = os.environ.get("MINICODE_EFFORT", "medium")  # anthropic seulement : low | medium | high | xhigh | max
MAX_STEPS = 30  # garde-fou : nombre max d'appels au modèle pour UNE demande
# MINICODE_YOLO=1 : accepte tout sans demander (comme le mode sans permissions de
# Claude Code). Pratique pour les tests automatiques, dangereux sur un vrai projet.
YOLO = os.environ.get("MINICODE_YOLO") == "1"

SYSTEM_PROMPT = f"""Tu es minicode, un assistant de programmation qui tourne dans le terminal.
Tu travailles dans le projet situé à : {WORKSPACE}

Méthode :
- Explore avant de répondre : grep pour trouver où est quelque chose, read_file pour le lire. Ne devine jamais le contenu d'un fichier.
- Avant de modifier un fichier avec edit_file, lis-le. Fais des modifications petites et ciblées.
- Après une modification, vérifie ton travail avec bash (par exemple en lançant les tests).
- L'utilisateur peut refuser une action : dans ce cas, ne la retente pas, demande-lui comment procéder.

Réponds de façon concise, en français."""

# Couleurs ANSI, pour distinguer ce que fait le harness de ce que dit le modèle.
DIM, CYAN, RED, GREEN, YELLOW, RESET = "\033[2m", "\033[36m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"

REFUSED = "L'utilisateur a refusé cette action. Ne la retente pas ; demande-lui comment il veut procéder."
# Les petits modèles finissent parfois leur tour avec seulement de la réflexion
# (bloc thinking) : ni outil, ni texte. Le harness les relance une fois.
NUDGE = "Tu n'as rien répondu. Continue : utilise un outil si tu dois agir, sinon donne ta réponse."
MAX_NUDGES = 1


def make_client():
    if PROVIDER == "ollama":
        # Le serveur local ne vérifie pas la clé, mais le SDK en exige une.
        return anthropic.Anthropic(base_url=OLLAMA_URL, api_key="ollama")
    return anthropic.Anthropic()  # lit ANTHROPIC_API_KEY dans l'environnement


def call_model(client, messages):
    """UN appel au modèle. Tout le "cerveau" est là ; tout le reste est du harness."""
    params = dict(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        tools=TOOL_SCHEMAS,
        messages=messages,
    )
    if PROVIDER == "ollama":
        # Ollama ne connaît que l'API de base (pas les options bêta ci-dessous).
        return client.messages.create(**params)
    return client.beta.messages.create(
        **params,
        output_config={"effort": EFFORT},
        # Si un filtre de sécurité refuse la requête, l'API la rejoue sur un
        # autre modèle au lieu d'échouer (paramètre côté serveur, en bêta).
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )


def _short(tool_input, limit=60):
    """Version courte des arguments, pour la ligne grise `→ outil(...)`."""
    parts = []
    for key, value in tool_input.items():
        value = repr(value)
        parts.append(f"{key}={value[:limit] + '…' if len(value) > limit else value}")
    return ", ".join(parts)


def _preview(name, tool_input, max_lines=15):
    """Montre à l'utilisateur ce que l'outil VA faire, avant qu'il le fasse."""
    if name == "bash":
        return f"{YELLOW}  $ {tool_input['command']}{RESET}"
    if name == "edit_file":
        lines = [f"{YELLOW}  fichier : {tool_input['path']}{RESET}"]
        for prefix, color, key in (("-", RED, "old_string"), ("+", GREEN, "new_string")):
            text_lines = tool_input[key].splitlines()
            lines += [f"{color}  {prefix} {line}{RESET}" for line in text_lines[:max_lines]]
            if len(text_lines) > max_lines:
                lines.append(f"{DIM}  ... ({len(text_lines) - max_lines} lignes de plus){RESET}")
        return "\n".join(lines)
    return f"  {tool_input}"


def ask_permission(name, tool_input):
    """ÉTAPE 5 (début) : c'est le harness, pas le modèle, qui décide si une action a lieu."""
    print(_preview(name, tool_input))
    if YOLO:
        print(f"{DIM}  (MINICODE_YOLO=1 : accepté automatiquement){RESET}")
        return True
    try:
        answer = input(f"{YELLOW}  Autoriser {name} ? [o/N] {RESET}")
    except EOFError:
        return False
    return answer.strip().lower() in ("o", "oui", "y", "yes")


def run_turn(client, messages, user_input, confirm=ask_permission):
    """Traite une demande utilisateur : la BOUCLE D'AGENT (étape 3).

    Modifie `messages` sur place. Renvoie le texte final du modèle.
    `confirm(name, input) -> bool` est appelé avant chaque outil dangereux.
    """
    turn_start = len(messages)
    messages.append({"role": "user", "content": user_input})
    nudges = 0

    for step in range(MAX_STEPS):
        response = call_model(client, messages)

        # Important : on ajoute `response.content` TEL QUEL (pas seulement le texte).
        # Il contient les blocs tool_use (dont l'API a besoin pour relier les
        # tool_result) et les blocs de réflexion, qu'on doit renvoyer sans les modifier.
        if response.stop_reason in ("refusal", "max_tokens"):
            # Réponse inutilisable (et peut-être un tool_use tronqué) : on annule
            # toute la demande pour garder un historique valide.
            del messages[turn_start:]
            return f"{RED}[arrêt : {response.stop_reason}] Demande annulée, reformule-la.{RESET}"

        messages.append({"role": "assistant", "content": response.content})

        text_parts = []
        tool_results = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                # Le modèle DEMANDE un outil ; c'est nous qui l'exécutons.
                print(f"{DIM}  → {block.name}({_short(block.input)}){RESET}")
                if block.name in DANGEROUS_TOOLS and not confirm(block.name, block.input):
                    # Refus : on ne l'exécute pas, mais on DOIT quand même renvoyer
                    # un tool_result, sinon l'API rejette l'historique.
                    result, is_error = REFUSED, True
                else:
                    result, is_error = run_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # relie le résultat à la demande
                    "content": result,
                    "is_error": is_error,
                })

        if response.stop_reason != "tool_use":
            if not text_parts and nudges < MAX_NUDGES:
                # Réponse vide : on relance au lieu de laisser l'utilisateur sans rien.
                nudges += 1
                print(f"{DIM}  (réponse vide, minicode relance le modèle){RESET}")
                messages.append({"role": "user", "content": NUDGE})
                continue
            # Le modèle n'a plus besoin d'outils : la demande est terminée.
            return "\n".join(text_parts) or f"{DIM}(le modèle n'a rien répondu){RESET}"

        # Texte intermédiaire éventuel ("je vais regarder le fichier X...").
        if text_parts:
            print(f"{DIM}{' '.join(text_parts)}{RESET}")

        # TOUS les résultats partent dans UN SEUL message "user".
        messages.append({"role": "user", "content": tool_results})

    return f"{RED}[arrêt : {MAX_STEPS} étapes atteintes sans réponse finale]{RESET}"


def main():
    client = make_client()
    messages = []  # TOUT l'état de la conversation tient dans cette liste

    print(f"minicode — {PROVIDER} / {MODEL}, projet {WORKSPACE}")
    print("Tape ta demande (Ctrl-D ou 'exit' pour quitter).\n")
    while True:
        try:
            user_input = input(f"{CYAN}> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input in ("exit", "quit"):
            break
        if not user_input:
            continue
        start = len(messages)
        try:
            print(run_turn(client, messages, user_input), "\n")
        except anthropic.AuthenticationError:
            sys.exit(f"{RED}Clé API invalide ou absente : exporte ANTHROPIC_API_KEY.{RESET}")
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            # Une erreur en plein tour laisse l'historique à moitié écrit (ex : un
            # tool_use sans son tool_result), que l'API refuserait ensuite. On annule.
            del messages[start:]
            print(f"{RED}Erreur API : {e} Demande annulée.{RESET}")
            if PROVIDER == "ollama" and isinstance(e, anthropic.APIConnectionError):
                print(f"{RED}Ollama ne répond pas sur {OLLAMA_URL} : lance `ollama serve`.{RESET}")
            elif PROVIDER == "ollama" and isinstance(e, anthropic.NotFoundError):
                print(f"{RED}Modèle absent : lance `ollama pull {MODEL}`.{RESET}")
            print()


if __name__ == "__main__":
    main()

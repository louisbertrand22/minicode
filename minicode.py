"""minicode : un mini agent de code, écrit à la main pour comprendre un harness.

Étapes couvertes ici :
  1. REPL de chat  -> le modèle n'a PAS de mémoire : on renvoie tout l'historique
                      (`messages`) à chaque appel.
  2. Un outil      -> le modèle *demande* un appel (bloc tool_use), on l'exécute,
                      on renvoie un bloc tool_result.
  3. Boucle agent  -> on rappelle le modèle tant qu'il demande des outils.

Lancer :  uv run minicode.py      (dans le dossier du projet à explorer)
"""

import os
import sys

import anthropic

from tools import TOOL_SCHEMAS, WORKSPACE, run_tool

MODEL = os.environ.get("MINICODE_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("MINICODE_EFFORT", "medium")  # low | medium | high | xhigh | max
MAX_STEPS = 30  # garde-fou : nombre max d'appels au modèle pour UNE demande

SYSTEM_PROMPT = f"""Tu es minicode, un assistant de programmation qui tourne dans le terminal.
Tu travailles dans le projet situé à : {WORKSPACE}
Utilise les outils pour explorer le code avant de répondre ; ne devine pas le contenu d'un fichier.
Réponds de façon concise, en français."""

# Couleurs ANSI, pour distinguer ce que fait le harness de ce que dit le modèle.
DIM, CYAN, RED, RESET = "\033[2m", "\033[36m", "\033[31m", "\033[0m"


def call_model(client, messages):
    """UN appel au modèle. Tout le "cerveau" est là ; tout le reste est du harness."""
    return client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        tools=TOOL_SCHEMAS,
        messages=messages,
        output_config={"effort": EFFORT},
        # Si un filtre de sécurité refuse la requête, l'API la rejoue sur un
        # autre modèle au lieu d'échouer (paramètre côté serveur, en bêta).
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )


def run_turn(client, messages, user_input):
    """Traite une demande utilisateur : la BOUCLE D'AGENT (étape 3).

    Modifie `messages` sur place. Renvoie le texte final du modèle.
    """
    turn_start = len(messages)
    messages.append({"role": "user", "content": user_input})

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
                print(f"{DIM}  → {block.name}({block.input}){RESET}")
                result, is_error = run_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # relie le résultat à la demande
                    "content": result,
                    "is_error": is_error,
                })

        if response.stop_reason != "tool_use":
            # Le modèle n'a plus besoin d'outils : la demande est terminée.
            return "\n".join(text_parts)

        # Texte intermédiaire éventuel ("je vais regarder le fichier X...").
        if text_parts:
            print(f"{DIM}{' '.join(text_parts)}{RESET}")

        # TOUS les résultats partent dans UN SEUL message "user".
        messages.append({"role": "user", "content": tool_results})

    return f"{RED}[arrêt : {MAX_STEPS} étapes atteintes sans réponse finale]{RESET}"


def main():
    client = anthropic.Anthropic()  # lit ANTHROPIC_API_KEY dans l'environnement
    messages = []  # TOUT l'état de la conversation tient dans cette liste

    print(f"minicode — modèle {MODEL}, projet {WORKSPACE}")
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
            print(f"{RED}Erreur API : {e}. Demande annulée.{RESET}\n")


if __name__ == "__main__":
    main()

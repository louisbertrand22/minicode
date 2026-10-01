"""minicode : un mini agent de code, écrit à la main pour comprendre un harness.

Étapes couvertes ici :
  1. REPL de chat  -> le modèle n'a PAS de mémoire : on renvoie tout l'historique
                      (`messages`) à chaque appel.
  2. Un outil      -> le modèle *demande* un appel (bloc tool_use), on l'exécute,
                      on renvoie un bloc tool_result.
  3. Boucle agent  -> on rappelle le modèle tant qu'il demande des outils.
  4. Outils qui AGISSENT : grep, edit_file, bash (l'agent peut modifier et vérifier).
  5. Permissions   -> le harness demande avant edit_file et bash, sauf si une
                      règle de .minicode/permissions.json décide (voir permissions.py).
  7. Streaming + journal -> la réponse s'affiche au fil de l'eau, et chaque échange est
                      enregistré dans .minicode/traces/ (voir tracelog.py, show_trace.py).

Tout l'affichage (façon Claude Code) est dans ui.py : ce fichier-ci ne contient
que le harness, et appelle `ui.xxx()` pour montrer ce qui se passe.

Lancer :  uv run minicode.py      (dans le dossier du projet à explorer)

Deux "fournisseurs" de modèle, même boucle :
  - ollama    (défaut, gratuit) : un modèle qui tourne sur ta machine. Ollama
                parle le même format que l'API Anthropic, donc on garde le même
                SDK en changeant juste l'adresse du serveur.
  - anthropic (payant) : les modèles Claude, via ANTHROPIC_API_KEY.
"""

import json
import os
import sys
import time

import anthropic

import permissions
from tools import DANGEROUS_TOOLS, PROTECTED_DIR, TOOL_SCHEMAS, WORKSPACE, precheck, run_tool, stop_all_sessions
from tracelog import Trace
from ui import TerminalUI, edit_diff

PROVIDER = os.environ.get("MINICODE_PROVIDER", "ollama")  # ollama | anthropic
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODELS = {"ollama": "qwen3:8b", "anthropic": "claude-opus-5-5"}
MODEL = os.environ.get("MINICODE_MODEL", DEFAULT_MODELS[PROVIDER])
EFFORT = os.environ.get("MINICODE_EFFORT", "medium")  # anthropic seulement : low | medium | high | xhigh | max
MAX_STEPS = 30  # garde-fou : nombre max d'appels au modèle pour UNE demande
# MINICODE_YOLO=1 : accepte tout sans demander (comme le mode sans permissions de
# Claude Code). Pratique pour les tests automatiques, dangereux sur un vrai projet.
YOLO = os.environ.get("MINICODE_YOLO") == "1"
SHOW_THINKING = os.environ.get("MINICODE_THINKING", "1") != "0"  # 0 = cacher l'aperçu de la réflexion
TRACE = os.environ.get("MINICODE_TRACE", "1") != "0"             # 0 = pas de journal

SYSTEM_PROMPT = f"""Tu es minicode, un assistant de programmation qui tourne dans le terminal.
Tu travailles dans le projet situé à : {WORKSPACE}

Méthode :
- Explore avant de répondre : grep pour trouver où est quelque chose, read_file pour le lire. Ne devine jamais le contenu d'un fichier.
- Avant de modifier un fichier avec edit_file, lis-le. Fais des modifications petites et ciblées.
- Après une modification, vérifie ton travail avec bash (par exemple en lançant les tests).
- Pour tester un programme interactif (input()) : lis-le d'abord avec read_file pour savoir quelles
  questions il pose. Si tes réponses ne dépendent pas de ce qu'il affiche, passe-les toutes dans le
  paramètre stdin de bash. Si elles en dépendent (jeu avec indices, nombre d'essais limité...),
  utilise interactive_start puis interactive_send, une réponse à la fois, en lisant chaque réponse.
  Lance les programmes Python avec python3, et utilise des chemins relatifs au projet.
- Quand l'utilisateur demande de corriger, modifier ou créer quelque chose, FAIS-LE avec les outils
  au lieu de demander « voulez-vous que je le fasse ? » : il valide chaque action dangereuse.
- Fais les vérifications toi-même au lieu de demander à l'utilisateur de les faire.
- Ne modifie JAMAIS un programme juste pour qu'un test passe (par exemple en remplaçant une
  valeur aléatoire par une valeur fixe) : adapte le test, pas le programme.
- L'utilisateur peut refuser une action : dans ce cas, ne la retente pas, demande-lui comment procéder.

Réponds de façon concise, en français. Tu peux utiliser du Markdown."""

REFUSED = "L'utilisateur a refusé cette action. Ne la retente pas ; demande-lui comment il veut procéder."
# Les petits modèles finissent parfois leur tour avec seulement de la réflexion
# (bloc thinking) : ni outil, ni texte. Le harness les relance une fois.
NUDGE = "Tu n'as rien répondu. Continue : utilise un outil si tu dois agir, sinon donne ta réponse."
MAX_NUDGES = 1

_ui = None


def get_ui():
    """L'interface par défaut (créée à la première utilisation)."""
    global _ui
    if _ui is None:
        _ui = TerminalUI(PROVIDER, MODEL, SHOW_THINKING, history_file=_history_file())
    return _ui


def _history_file():
    path = WORKSPACE / PROTECTED_DIR / "history"
    path.parent.mkdir(exist_ok=True)
    return path


def make_client():
    if PROVIDER == "ollama":
        # Le serveur local ne vérifie pas la clé, mais le SDK en exige une.
        return anthropic.Anthropic(base_url=OLLAMA_URL, api_key="ollama")
    return anthropic.Anthropic()  # lit ANTHROPIC_API_KEY dans l'environnement


def request_params(messages):
    """Exactement ce qu'on envoie au modèle (hors en-têtes HTTP)."""
    params = dict(model=MODEL, system=SYSTEM_PROMPT, messages=messages)
    if PROVIDER == "ollama":
        # Ollama ne connaît que l'API de base (pas les options bêta ci-dessous).
        return {**params, "max_tokens": 16000, "tools": TOOL_SCHEMAS}
    return {
        **params,
        "max_tokens": 64000,  # en streaming, pas de risque de timeout : on laisse de la marge
        # eager_input_streaming : les arguments d'un outil (ex : tout un fichier pour
        # edit_file) arrivent au fil de l'eau. Contrepartie : l'API ne les valide plus,
        # c'est run_tool() qui vérifie qu'ils sont complets.
        "tools": [{**t, "eager_input_streaming": True} for t in TOOL_SCHEMAS],
        "thinking": {"type": "adaptive", "display": "summarized"},  # sinon la réflexion arrive vide
        "output_config": {"effort": EFFORT},
        # Si un filtre de sécurité refuse la requête, l'API la rejoue sur un
        # autre modèle au lieu d'échouer (paramètre côté serveur, en bêta).
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
    }


def call_model(client, messages, trace=None, ui=None):
    """UN appel au modèle, en STREAMING (étape 7).

    Au lieu d'attendre la réponse complète, on reçoit des petits événements
    ("thinking", "text"...) qu'on passe à l'interface dès qu'ils arrivent. À la
    fin, le SDK reconstitue le message complet (get_final_message), identique à
    ce que renverrait un appel normal : le reste de la boucle ne change pas.
    """
    ui = ui or get_ui()
    api = client.messages if PROVIDER == "ollama" else client.beta.messages
    start = time.time()
    with ui.model_call() as view, api.stream(**request_params(messages)) as stream:
        for event in stream:
            if event.type == "thinking":
                view.on_thinking(event.thinking)
            elif event.type == "text":
                view.on_text(event.text)
        response = stream.get_final_message()
    seconds = time.time() - start
    # Le total grossit à chaque appel (tout l'historique), mais le serveur garde en
    # CACHE le début déjà vu : `input_tokens` ne compte que la partie nouvelle.
    ui.record_usage(getattr(response, "usage", None))
    if trace:
        trace.log_call(messages, response, seconds)
    return response


def ask_permission(name, tool_input, ui=None):
    """ÉTAPE 5 : c'est le harness, pas le modèle, qui décide si une action a lieu.

    Renvoie (autorisé, consigne de l'utilisateur ou None).
    """
    ui = ui or get_ui()
    decision, reason = permissions.decide(name, tool_input)
    if decision == "deny":  # une interdiction l'emporte sur tout, même sur YOLO
        ui.error(f"✗ interdit par la règle {reason}")
        return False, None
    if decision == "allow":
        ui.info(f"✓ autorisé par la règle {reason}")
        return True, None
    if YOLO:
        ui.info("(MINICODE_YOLO=1 : accepté automatiquement)")
        return True, None
    if reason:  # ex : commande composée
        ui.info(f"({reason})")

    rule = permissions.suggest_rule(name, tool_input) if permissions.can_remember(name, tool_input) else None
    answer, feedback = ui.permission(name, tool_input, rule)
    if answer == "always":
        permissions.add_allow_rule(rule)
        ui.info(f"règle ajoutée dans {permissions.rules_file()} : {rule}")
    return answer in ("yes", "always"), feedback


def run_turn(client, messages, user_input, confirm=None, trace=None, ui=None):
    """Traite une demande utilisateur : la BOUCLE D'AGENT (étape 3).

    Modifie `messages` sur place. Renvoie le texte final du modèle (déjà affiché).
    `confirm(name, input)` est appelé avant chaque outil dangereux ; il renvoie
    un booléen, ou (booléen, consigne de l'utilisateur).
    """
    try:
        return _agent_loop(client, messages, user_input, confirm, trace, ui)
    finally:
        # Quoi qu'il arrive (fin normale, erreur, Ctrl-C), on n'abandonne pas de
        # programmes interactifs qui tourneraient encore en arrière-plan.
        stop_all_sessions()


def _agent_loop(client, messages, user_input, confirm, trace, ui):
    ui = ui or get_ui()
    confirm = confirm or (lambda name, tool_input: ask_permission(name, tool_input, ui))
    turn_start = len(messages)
    messages.append({"role": "user", "content": user_input})
    nudges = 0
    json_retries = 0
    failed_calls = set()  # appels qui ont échoué pendant cette demande

    for step in range(MAX_STEPS):
        try:
            response = call_model(client, messages, trace, ui)
        except ValueError:
            # En streaming, les arguments d'un outil arrivent par morceaux de JSON.
            # S'ils sont illisibles, il n'y a pas de tool_use complet à qui répondre :
            # on refait simplement l'appel (au plus 2 fois de suite).
            json_retries += 1
            if json_retries > 2:
                raise
            ui.info("(arguments d'outil illisibles, minicode relance l'appel)")
            continue
        json_retries = 0

        # Important : on ajoute `response.content` TEL QUEL (pas seulement le texte).
        # Il contient les blocs tool_use (dont l'API a besoin pour relier les
        # tool_result) et les blocs de réflexion, qu'on doit renvoyer sans les modifier.
        if response.stop_reason in ("refusal", "max_tokens"):
            # Réponse inutilisable (et peut-être un tool_use tronqué) : on annule
            # toute la demande pour garder un historique valide.
            del messages[turn_start:]
            message = f"[arrêt : {response.stop_reason}] Demande annulée, reformule-la."
            ui.error(message)
            return message

        messages.append({"role": "assistant", "content": response.content})

        text_parts = []
        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text.strip():  # un texte d'espaces = pas de réponse
                text_parts.append(block.text)
            elif block.type == "tool_use":
                # Le modèle DEMANDE un outil ; c'est nous qui l'exécutons.
                ui.tool_call(block.name, block.input)
                diff = _diff_before_edit(block.name, block.input)
                allowed, feedback = True, None
                problem = precheck(block.name, block.input)
                if problem:
                    # L'appel va échouer : inutile de demander la permission.
                    result, is_error = problem, True
                else:
                    if block.name in DANGEROUS_TOOLS:
                        answer = confirm(block.name, block.input)
                        allowed, feedback = answer if isinstance(answer, tuple) else (answer, None)
                    if allowed:
                        result, is_error = run_tool(block.name, block.input)
                    else:
                        # Refus : on ne l'exécute pas, mais on DOIT quand même renvoyer
                        # un tool_result, sinon l'API rejette l'historique.
                        result = REFUSED + (f" Consigne de l'utilisateur : {feedback}" if feedback else "")
                        is_error = True
                if is_error and allowed:
                    # Les petits modèles refont parfois EXACTEMENT le même appel raté, en boucle.
                    key = json.dumps([block.name, block.input], sort_keys=True)
                    if key in failed_calls:
                        result += ("\n[indice minicode : tu as déjà fait exactement cet appel et il a échoué "
                                   "de la même façon. Change d'approche.]")
                    failed_calls.add(key)
                ui.tool_result(block.name, block.input, result if allowed else "refusé par l'utilisateur",
                               is_error, diff)
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
                ui.info("(réponse vide, minicode relance le modèle)")
                messages.append({"role": "user", "content": NUDGE})
                continue
            # Le modèle n'a plus besoin d'outils : la demande est terminée.
            if not text_parts:
                ui.info("(le modèle n'a rien répondu)")
            return "\n".join(text_parts) or "(le modèle n'a rien répondu)"

        # TOUS les résultats partent dans UN SEUL message "user".
        messages.append({"role": "user", "content": tool_results})

    message = f"[arrêt : {MAX_STEPS} étapes atteintes sans réponse finale]"
    ui.error(message)
    return message


def _diff_before_edit(name, tool_input):
    """Pour afficher le diff APRÈS l'édition, il faut le calculer AVANT (le fichier va changer)."""
    if name != "edit_file":
        return None
    try:
        return edit_diff(tool_input)
    except (KeyError, OSError):
        return None


def main():
    client = make_client()
    messages = []  # TOUT l'état de la conversation tient dans cette liste
    trace = None
    if TRACE:
        trace = Trace(WORKSPACE / PROTECTED_DIR / "traces",
                      provider=PROVIDER, model=MODEL, system=SYSTEM_PROMPT, tools=TOOL_SCHEMAS)
    ui = get_ui()
    ui.welcome(WORKSPACE, trace.path if trace else None)

    while True:
        user_input = ui.read_input()
        if user_input is None or user_input.strip() in ("/exit", "exit", "quit"):
            break
        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input == "/help":
            ui.help()
            continue
        if user_input == "/clear":
            # Le modèle n'a pas de mémoire : vider la liste = nouvelle conversation.
            messages.clear()
            ui.reset()
            ui.info("Nouvelle conversation : l'historique envoyé au modèle est vide.")
            continue
        if user_input == "/trace":
            ui.info(f"journal : {trace.path}" if trace else "journal désactivé (MINICODE_TRACE=0)")
            continue

        start, started_at = len(messages), time.time()
        calls_before = ui.calls
        try:
            run_turn(client, messages, user_input, trace=trace, ui=ui)
            ui.turn_done(ui.calls - calls_before, time.time() - started_at)
        except KeyboardInterrupt:
            # Ctrl-C : on abandonne la demande en cours. L'historique peut être à
            # moitié écrit (un tool_use sans son tool_result) : on l'annule.
            del messages[start:]
            ui.warn("Interrompu. La demande a été annulée.")
        except anthropic.AuthenticationError:
            sys.exit("Clé API invalide ou absente : exporte ANTHROPIC_API_KEY.")
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            # Une erreur en plein tour laisse l'historique à moitié écrit : on annule.
            del messages[start:]
            ui.error(f"Erreur API : {e} Demande annulée.")
            if PROVIDER == "ollama" and isinstance(e, anthropic.APIConnectionError):
                ui.error(f"Ollama ne répond pas sur {OLLAMA_URL} : lance `ollama serve`.")
            elif PROVIDER == "ollama" and isinstance(e, anthropic.NotFoundError):
                ui.error(f"Modèle absent : lance `ollama pull {MODEL}`.")


if __name__ == "__main__":
    main()

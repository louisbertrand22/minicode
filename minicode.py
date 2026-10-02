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
  6. Contexte projet -> les AGENTS.md du projet sont ajoutés au prompt système
                      (voir agents_md.py ; /init en fait écrire un par l'agent).
  7. Streaming + journal -> la réponse s'affiche au fil de l'eau, et chaque échange est
                      enregistré dans .minicode/traces/ (voir tracelog.py, show_trace.py).
  8. Contexte      -> avant chaque appel, on estime la taille de ce qu'on envoie ; si
                      la fenêtre du modèle va déborder, on efface les vieux résultats
                      d'outils, puis on résume les anciens tours (voir context.py).
  9. Sous-agents   -> l'outil `task` lance une 2e boucle d'agent, avec un historique
                      vide et des outils de lecture ; seul son rapport revient (subagent.py).

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

import agents_md
import context
import permissions
import subagent
import tools
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
CONTEXT_WINDOW = context.window_for(PROVIDER)  # MINICODE_CONTEXT_WINDOW pour la changer
# Étape 8 : un seul résultat d'outil ne doit pas remplir la fenêtre à lui seul.
tools.MAX_OUTPUT_CHARS = context.output_limit(CONTEXT_WINDOW)

BASE_SYSTEM_PROMPT = f"""Tu es minicode, un assistant de programmation qui tourne dans le terminal.
Tu travailles dans le projet situé à : {WORKSPACE}

Méthode :
- Explore avant de répondre : grep pour trouver où est quelque chose, read_file pour le lire. Ne devine jamais le contenu d'un fichier.
- Pour une recherche qui demande de parcourir PLUSIEURS fichiers (« où est géré X ? », « comment marche Y ? »),
  délègue-la à task : un sous-agent cherche et te renvoie un court rapport, ton contexte reste petit.
  Pour lire UN fichier dont tu connais le nom, utilise directement read_file.
- Avant de modifier un fichier avec edit_file, lis-le. Fais des modifications petites et ciblées.
- Après une modification, vérifie ton travail avec bash (par exemple en lançant les tests).
- Pour tester un programme interactif (input()) : lis-le d'abord avec read_file pour savoir quelles
  questions il pose. Si tes réponses ne dépendent pas de ce qu'il affiche, passe-les toutes dans le
  paramètre stdin de bash. Si elles en dépendent (jeu avec indices, nombre d'essais limité...),
  utilise interactive_start puis interactive_send, une réponse à la fois, en lisant chaque réponse.
  Lance les programmes Python avec python3, et utilise des chemins relatifs au projet.
- Quand l'utilisateur demande de corriger, modifier ou créer quelque chose, FAIS-LE avec les outils
  au lieu de demander « voulez-vous que je le fasse ? » : il valide chaque action dangereuse.
- Fais les vérifications toi-même au lieu de demander à l'utilisateur de les faire. Vérifie le
  COMPORTEMENT attendu (ex : le compteur d'essais diminue bien), pas seulement l'absence d'erreur.
- Une variable qui doit garder sa valeur d'un tour de boucle à l'autre s'initialise AVANT la boucle.
- Ne modifie JAMAIS un programme juste pour qu'un test passe (par exemple en remplaçant une
  valeur aléatoire par une valeur fixe) : adapte le test, pas le programme.
- L'utilisateur peut refuser une action : dans ce cas, ne la retente pas, demande-lui comment procéder.

Réponds de façon concise, en français. Tu peux utiliser du Markdown."""
# Étape 6 : le prompt réellement envoyé = BASE_SYSTEM_PROMPT + les AGENTS.md du projet.
# load_project_context() le (re)calcule ; avant ça, c'est juste la base.
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT
AGENTS_FILES = []
_agents_fingerprint = None
# AGENTS.md part à CHAQUE appel : au plus ~10 % de la fenêtre (≈ 2 400 caractères avec 8k).
AGENTS_MAX_CHARS = max(2_000, min(20_000, int(CONTEXT_WINDOW * 0.10 * context.CHARS_PER_TOKEN)))

REFUSED = "L'utilisateur a refusé cette action. Ne la retente pas ; demande-lui comment il veut procéder."
# Les petits modèles finissent parfois leur tour avec seulement de la réflexion
# (bloc thinking) : ni outil, ni texte. Le harness les relance une fois.
NUDGE = "Tu n'as rien répondu. Continue : utilise un outil si tu dois agir, sinon donne ta réponse."
MAX_NUDGES = 1
MAX_SAME_FAILURES = 3  # au 3e appel identique raté, on arrête la demande
# Étape 9 : l'agent principal a en plus l'outil `task` ; un sous-agent n'a que la lecture.
MAIN_TOOLS = TOOL_SCHEMAS + [subagent.TASK_SCHEMA]
SUB_TOOLS = [t for t in TOOL_SCHEMAS if t["name"] in subagent.READ_ONLY_TOOLS]
PROJECT_CONTEXT = ""  # le texte des AGENTS.md (étape 6), donné aussi aux sous-agents
BUDGET = context.Budget(CONTEXT_WINDOW, SYSTEM_PROMPT, MAIN_TOOLS)
SUMMARY_SYSTEM = "Tu résumes des conversations de travail, fidèlement et brièvement."

_ui = None


def load_project_context(trace=None, ui=None) -> bool:
    """ÉTAPE 6 : (re)lit les AGENTS.md et reconstruit le prompt système.

    Appelé au démarrage puis avant chaque demande : si l'agent (ou toi) modifie
    AGENTS.md pendant la session, la demande suivante en tient compte. Renvoie True
    si le prompt a changé.
    """
    global SYSTEM_PROMPT, AGENTS_FILES, PROJECT_CONTEXT, _agents_fingerprint
    fingerprint = agents_md.fingerprint(WORKSPACE)
    if fingerprint == _agents_fingerprint:
        return False  # rien n'a changé : on garde le même prompt (et le cache du serveur)
    first_time = _agents_fingerprint is None
    _agents_fingerprint = fingerprint
    PROJECT_CONTEXT, AGENTS_FILES = agents_md.load(WORKSPACE, AGENTS_MAX_CHARS)
    SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + PROJECT_CONTEXT
    BUDGET.set_fixed(SYSTEM_PROMPT, MAIN_TOOLS)  # étape 8 : la partie fixe a changé de taille
    if trace and not first_time:
        trace.log_system(SYSTEM_PROMPT)
    if ui and not first_time:
        names = ", ".join(str(p) for p in AGENTS_FILES) or "aucun"
        ui.info(f"AGENTS.md rechargé ({names}) : le prompt système a changé.")
    return True


def get_ui():
    """L'interface par défaut (créée à la première utilisation)."""
    global _ui
    if _ui is None:
        _ui = TerminalUI(PROVIDER, MODEL, SHOW_THINKING, history_file=_history_file(),
                         context_window=CONTEXT_WINDOW)
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


def request_params(messages, system=None, tool_schemas=None):
    """Exactement ce qu'on envoie au modèle (hors en-têtes HTTP).

    Par défaut : l'agent principal. Un sous-agent (étape 9) passe son prompt et ses outils.
    """
    system = SYSTEM_PROMPT if system is None else system
    tool_schemas = MAIN_TOOLS if tool_schemas is None else tool_schemas
    params = dict(model=MODEL, system=system, messages=messages)
    if PROVIDER == "ollama":
        # Ollama ne connaît que l'API de base (pas les options bêta ci-dessous).
        return {**params, "max_tokens": 16000, "tools": tool_schemas}
    return {
        **params,
        "max_tokens": 64000,  # en streaming, pas de risque de timeout : on laisse de la marge
        # eager_input_streaming : les arguments d'un outil (ex : tout un fichier pour
        # edit_file) arrivent au fil de l'eau. Contrepartie : l'API ne les valide plus,
        # c'est run_tool() qui vérifie qu'ils sont complets.
        "tools": [{**t, "eager_input_streaming": True} for t in tool_schemas],
        "thinking": {"type": "adaptive", "display": "summarized"},  # sinon la réflexion arrive vide
        "output_config": {"effort": EFFORT},
        # Si un filtre de sécurité refuse la requête, l'API la rejoue sur un
        # autre modèle au lieu d'échouer (paramètre côté serveur, en bêta).
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
    }


def call_model(client, messages, trace=None, ui=None, system=None, tool_schemas=None, budget=None, agent=None):
    """UN appel au modèle, en STREAMING (étape 7).

    Au lieu d'attendre la réponse complète, on reçoit des petits événements
    ("thinking", "text"...) qu'on passe à l'interface dès qu'ils arrivent. À la
    fin, le SDK reconstitue le message complet (get_final_message), identique à
    ce que renverrait un appel normal : le reste de la boucle ne change pas.

    `agent` : None pour l'agent principal, "sous-agent" pour l'étape 9. Le texte d'un
    sous-agent n'est pas affiché : c'est un rapport pour l'agent principal, pas pour toi.
    """
    ui = ui or get_ui()
    budget = budget or BUDGET
    api = client.messages if PROVIDER == "ollama" else client.beta.messages
    start = time.time()
    with ui.model_call() as view, api.stream(**request_params(messages, system, tool_schemas)) as stream:
        for event in stream:
            if event.type == "thinking":
                view.on_thinking(event.thinking)
            elif event.type == "text" and agent is None:
                view.on_text(event.text)
        response = stream.get_final_message()
    seconds = time.time() - start
    # Le total grossit à chaque appel (tout l'historique), mais le serveur garde en
    # CACHE le début déjà vu : `input_tokens` ne compte que la partie nouvelle.
    usage = getattr(response, "usage", None)
    # La barre du bas montre le contexte de l'agent PRINCIPAL ; un sous-agent compte
    # seulement comme un appel de plus.
    ui.record_usage(usage, update_context=agent is None)
    if usage is not None:
        # Étape 8 : le VRAI nombre de tokens corrige nos estimations suivantes.
        budget.calibrate(messages, context.real_tokens(usage))
    if trace:
        trace.log_call(messages, response, seconds, agent=agent, system=system)
    return response


def summarize(client, old_messages, trace=None, ui=None):
    """ÉTAPE 8 : demande au modèle un résumé de `old_messages` (sans outils, sans historique).

    Le texte de la conversation doit lui-même tenir dans la fenêtre : context.transcript
    le raccourcit. Si l'appel échoue, on se rabat sur un résumé fabriqué sans modèle.
    """
    ui = ui or get_ui()
    max_chars = int(BUDGET.limit * 0.6 * context.CHARS_PER_TOKEN)
    request = [{"role": "user", "content": context.SUMMARY_INSTRUCTIONS + context.transcript(old_messages, max_chars)}]
    start = time.time()
    try:
        # Le texte du résumé n'est PAS envoyé à l'affichage : ce n'est pas une réponse pour
        # l'utilisateur. On montre seulement l'animation (et la réflexion).
        with ui.model_call() as view, client.messages.stream(
                model=MODEL, system=SUMMARY_SYSTEM, messages=request, max_tokens=4000) as stream:
            for event in stream:
                if event.type == "thinking":
                    view.on_thinking(event.thinking)
            response = stream.get_final_message()
    except (anthropic.APIStatusError, anthropic.APIConnectionError, ValueError) as e:
        ui.warn(f"(résumé impossible : {e} ; minicode garde juste la liste des demandes)")
        return context.fallback_summary(old_messages)
    if trace:
        trace.log_call(request, response, time.time() - start, agent="résumé", system=SUMMARY_SYSTEM)
    summary = "".join(b.text for b in response.content if b.type == "text").strip()
    return summary or context.fallback_summary(old_messages)


def compact(client, messages, end, trace=None, ui=None):
    """Remplace messages[:end] par un résumé (2 messages). Renvoie le nouvel indice de fin (2)."""
    ui = ui or get_ui()
    before = BUDGET.estimate(messages)
    summary = summarize(client, messages[:end], trace, ui)
    messages[:end] = context.summary_messages(summary)
    after = BUDGET.estimate(messages)
    ui.context_freed(f"{end} anciens messages remplacés par un résumé", before, after)
    if trace:
        trace.log_compact("summary", before, after, summary)
    return 2


def fit_context(client, messages, turn_start, trace=None, ui=None, budget=None):
    """ÉTAPE 8 : appelé AVANT chaque appel au modèle. Fait de la place si besoin.

    Du moins cher au plus cher. Renvoie le nouvel indice de début de la demande en
    cours (il change si les tours précédents sont résumés).
    """
    ui = ui or get_ui()
    budget = budget or BUDGET  # un sous-agent a son propre budget (étape 9)
    if not budget.over(messages):
        return turn_start
    # 1. Gratuit : les vieux résultats d'outils ont déjà servi.
    before = budget.estimate(messages)
    cleared = context.clear_old_tool_results(messages)
    if cleared:
        after = budget.estimate(messages)
        ui.context_freed(f"{cleared} ancien(s) résultat(s) d'outil effacé(s)", before, after)
        if trace:
            trace.log_compact("clear_tool_results", before, after)
    # 2. Un appel au modèle : résumer les tours PRÉCÉDENTS. La demande en cours reste
    #    intacte : on ne coupe jamais entre un tool_use et son tool_result.
    if budget.over(messages) and turn_start > 0 and not context.is_summary(messages[:turn_start]):
        turn_start = compact(client, messages, turn_start, trace, ui)
    # 3. Dernier recours : la demande en cours est énorme à elle seule.
    before = budget.estimate(messages)
    if budget.over(messages) and context.clear_old_tool_results(messages, keep=1):
        ui.context_freed("seul le dernier résultat d'outil est gardé", before, budget.estimate(messages))
    if budget.over(messages):
        ui.warn(f"(contexte toujours trop grand : ~{budget.estimate(messages)}/{CONTEXT_WINDOW} tokens. "
                "Le modèle risque d'oublier le début ; fais /clear si ses réponses se dégradent.)")
    return turn_start


def run_subagent(client, tool_input, trace=None, ui=None):
    """ÉTAPE 9 : l'outil `task`. Une 2e boucle d'agent, avec son PROPRE historique.

    C'est la même boucle que _agent_loop, en plus simple : un historique qui commence
    vide (juste la consigne), des outils de lecture (pas de permission à demander), un
    nombre d'étapes limité. Renvoie (rapport, is_error) : le rapport devient le
    tool_result de `task` chez l'agent principal ; tout le reste est oublié.
    """
    ui = ui or get_ui()
    system = subagent.SYSTEM_PROMPT.format(workspace=WORKSPACE) + PROJECT_CONTEXT
    budget = context.Budget(CONTEXT_WINDOW, system, SUB_TOOLS)
    messages = [{"role": "user", "content": tool_input["prompt"]}]  # PAS l'historique principal
    ui.subagent_start(tool_input.get("description", ""))
    for step in range(subagent.MAX_STEPS + 1):
        fit_context(client, messages, 0, trace, ui, budget)  # étape 8, aussi pour lui
        try:
            response = call_model(client, messages, trace, ui, system, SUB_TOOLS, budget, agent="sous-agent")
        except ValueError:
            return "Le sous-agent a produit un appel d'outil illisible. Relance task ou cherche toi-même.", True
        text = "\n".join(b.text for b in response.content if b.type == "text" and b.text.strip())
        if response.stop_reason != "tool_use":
            ui.subagent_done(step + 1)
            return text or "(le sous-agent n'a rien répondu)", not text
        if step == subagent.MAX_STEPS:
            ui.subagent_done(step + 1)
            note = f"[sous-agent arrêté : {subagent.MAX_STEPS} étapes sans rapport final]"
            return (text + "\n" + note) if text else note, not text
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name in subagent.READ_ONLY_TOOLS:
                result, is_error = run_tool(block.name, block.input)
            else:  # le modèle peut demander un outil qu'il n'a pas : on refuse proprement
                result, is_error = f"{block.name} n'est pas disponible pour un sous-agent (lecture seule).", True
            ui.subagent_step(block.name, block.input, is_error)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": result, "is_error": is_error})
        if step == subagent.MAX_STEPS - 1:
            results.append({"type": "text", "text": subagent.FINISH_NOW})  # dernier appel : le rapport
        messages.append({"role": "user", "content": results})


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
    failed_calls = {}  # appel raté -> nombre de fois, pendant cette demande

    for step in range(MAX_STEPS):
        turn_start = fit_context(client, messages, turn_start, trace, ui)
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
        stuck = False
        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text.strip():  # un texte d'espaces = pas de réponse
                text_parts.append(block.text)
            elif block.type == "tool_use":
                # Le modèle DEMANDE un outil ; c'est nous qui l'exécutons.
                ui.tool_call(block.name, block.input)
                diff = _diff_before_edit(block.name, block.input)
                allowed, feedback = True, None
                if block.name == "task":
                    problem = subagent.validate(block.input)
                else:
                    problem = precheck(block.name, block.input)
                if problem:
                    # L'appel va échouer : inutile de demander la permission.
                    result, is_error = problem, True
                else:
                    if block.name in DANGEROUS_TOOLS:
                        answer = confirm(block.name, block.input)
                        allowed, feedback = answer if isinstance(answer, tuple) else (answer, None)
                    if allowed and block.name == "task":
                        # Étape 9 : pas d'outil Python ici, mais une boucle d'agent entière.
                        result, is_error = run_subagent(client, block.input, trace, ui)
                        result = tools._truncate(result)
                    elif allowed:
                        result, is_error = run_tool(block.name, block.input)
                    else:
                        # Refus : on ne l'exécute pas, mais on DOIT quand même renvoyer
                        # un tool_result, sinon l'API rejette l'historique.
                        result = REFUSED + (f" Consigne de l'utilisateur : {feedback}" if feedback else "")
                        is_error = True
                if is_error and allowed:
                    # Les petits modèles refont parfois EXACTEMENT le même appel raté, en boucle.
                    key = json.dumps([block.name, block.input], sort_keys=True)
                    failed_calls[key] = failed_calls.get(key, 0) + 1
                    if failed_calls[key] >= 2:
                        result += ("\n[indice minicode : tu as déjà fait exactement cet appel et il a échoué "
                                   "de la même façon. Change d'approche.]")
                    if failed_calls[key] >= MAX_SAME_FAILURES:
                        stuck = True
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

        if stuck:
            # Le modèle tourne en boucle malgré les indices : inutile de brûler du temps.
            # L'historique reste valide (chaque tool_use a son tool_result) : l'utilisateur
            # peut reformuler, ou faire /clear.
            message = (f"[arrêt : le modèle a refait {MAX_SAME_FAILURES} fois le même appel raté. "
                       "Reformule ta demande plus précisément, ou fais la modification toi-même.]")
            ui.error(message)
            return message

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
    load_project_context()  # étape 6 : AVANT le journal, qui enregistre le prompt système
    if TRACE:
        trace = Trace(WORKSPACE / PROTECTED_DIR / "traces",
                      provider=PROVIDER, model=MODEL, system=SYSTEM_PROMPT, tools=MAIN_TOOLS)
    ui = get_ui()
    ui.welcome(WORKSPACE, trace.path if trace else None, AGENTS_FILES)

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
        if user_input == "/context":
            ui.context_report(CONTEXT_WINDOW, BUDGET.limit, int(BUDGET.fixed * BUDGET.ratio),
                              {k: int(v * BUDGET.ratio) for k, v in context.breakdown(messages).items()},
                              BUDGET.estimate(messages))
            continue
        if user_input == "/compact":
            if len(messages) <= 2:
                ui.info("Rien à résumer.")
            else:
                try:
                    compact(client, messages, len(messages), trace, ui)
                except KeyboardInterrupt:
                    ui.warn("Interrompu : la conversation n'a pas été résumée.")
            continue
        if user_input == "/trace":
            ui.info(f"journal : {trace.path}" if trace else "journal désactivé (MINICODE_TRACE=0)")
            continue
        load_project_context(trace, ui)  # AGENTS.md a pu changer depuis la dernière demande
        if user_input == "/agents":
            ui.agents_report(AGENTS_FILES, len(SYSTEM_PROMPT) - len(BASE_SYSTEM_PROMPT), AGENTS_MAX_CHARS)
            continue
        if user_input == "/init":
            # Pas de magie : /init est une demande toute prête, traitée comme les autres.
            user_input = agents_md.INIT_REQUEST

        # Copie de la liste pour pouvoir annuler la demande : un simple indice ne suffit
        # plus, car l'étape 8 peut résumer le DÉBUT de l'historique pendant la demande.
        snapshot, started_at = list(messages), time.time()
        calls_before = ui.calls
        try:
            run_turn(client, messages, user_input, trace=trace, ui=ui)
            ui.turn_done(ui.calls - calls_before, time.time() - started_at)
        except KeyboardInterrupt:
            # Ctrl-C : on abandonne la demande en cours. L'historique peut être à
            # moitié écrit (un tool_use sans son tool_result) : on l'annule.
            messages[:] = snapshot
            ui.warn("Interrompu. La demande a été annulée.")
        except anthropic.AuthenticationError:
            sys.exit("Clé API invalide ou absente : exporte ANTHROPIC_API_KEY.")
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            # Une erreur en plein tour laisse l'historique à moitié écrit : on annule.
            messages[:] = snapshot
            ui.error(f"Erreur API : {e} Demande annulée.")
            if PROVIDER == "ollama" and isinstance(e, anthropic.APIConnectionError):
                ui.error(f"Ollama ne répond pas sur {OLLAMA_URL} : lance `ollama serve`.")
            elif PROVIDER == "ollama" and isinstance(e, anthropic.NotFoundError):
                ui.error(f"Modèle absent : lance `ollama pull {MODEL}`.")


if __name__ == "__main__":
    main()

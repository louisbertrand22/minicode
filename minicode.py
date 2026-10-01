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
  7. Streaming + journal -> la réponse s'affiche mot à mot, et chaque échange est
                      enregistré dans .minicode/traces/ (voir tracelog.py, show_trace.py).

Lancer :  uv run minicode.py      (dans le dossier du projet à explorer)

Deux "fournisseurs" de modèle, même boucle :
  - ollama    (défaut, gratuit) : un modèle qui tourne sur ta machine. Ollama
                parle le même format que l'API Anthropic, donc on garde le même
                SDK en changeant juste l'adresse du serveur.
  - anthropic (payant) : les modèles Claude, via ANTHROPIC_API_KEY.
"""

import os
import sys
import time

import anthropic

import permissions
from tools import DANGEROUS_TOOLS, PROTECTED_DIR, TOOL_SCHEMAS, WORKSPACE, run_tool
from tracelog import Trace

PROVIDER = os.environ.get("MINICODE_PROVIDER", "ollama")  # ollama | anthropic
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODELS = {"ollama": "qwen3:8b", "anthropic": "claude-opus-5-5"}
MODEL = os.environ.get("MINICODE_MODEL", DEFAULT_MODELS[PROVIDER])
EFFORT = os.environ.get("MINICODE_EFFORT", "medium")  # anthropic seulement : low | medium | high | xhigh | max
MAX_STEPS = 30  # garde-fou : nombre max d'appels au modèle pour UNE demande
# MINICODE_YOLO=1 : accepte tout sans demander (comme le mode sans permissions de
# Claude Code). Pratique pour les tests automatiques, dangereux sur un vrai projet.
YOLO = os.environ.get("MINICODE_YOLO") == "1"
SHOW_THINKING = os.environ.get("MINICODE_THINKING", "1") != "0"  # 0 = cacher la réflexion
TRACE = os.environ.get("MINICODE_TRACE", "1") != "0"             # 0 = pas de journal

SYSTEM_PROMPT = f"""Tu es minicode, un assistant de programmation qui tourne dans le terminal.
Tu travailles dans le projet situé à : {WORKSPACE}

Méthode :
- Explore avant de répondre : grep pour trouver où est quelque chose, read_file pour le lire. Ne devine jamais le contenu d'un fichier.
- Avant de modifier un fichier avec edit_file, lis-le. Fais des modifications petites et ciblées.
- Après une modification, vérifie ton travail avec bash (par exemple en lançant les tests).
- Pour tester un programme interactif (input()) : lis-le d'abord avec read_file pour savoir quelles
  questions il pose, puis passe les réponses dans le paramètre stdin de bash. Si le résultat est
  aléatoire, choisis des saisies qui marchent quand même (ex : 1 à 100 pour un nombre à deviner).
- Fais les vérifications toi-même au lieu de demander à l'utilisateur de les faire.
- Ne modifie JAMAIS un programme juste pour qu'un test passe (par exemple en remplaçant une
  valeur aléatoire par une valeur fixe) : adapte le test, pas le programme.
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


class LivePrinter:
    """Affiche le flux au fil de l'eau : la réflexion en gris (💭), la réponse en normal."""

    def __init__(self):
        self.kind = None  # ce qu'on est en train d'afficher : None, "thinking" ou "text"

    def show(self, kind, chunk):
        if kind == "thinking" and not SHOW_THINKING:
            return
        if kind != self.kind:
            self.end()
            if kind == "thinking":
                print(f"{DIM}  💭 ", end="")  # gris jusqu'au RESET de end()
            self.kind = kind
        print(chunk, end="", flush=True)

    def end(self):
        if self.kind:
            print(RESET)  # fin de la couleur + retour à la ligne
            self.kind = None


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


def call_model(client, messages, trace=None):
    """UN appel au modèle, en STREAMING (étape 7).

    Au lieu d'attendre la réponse complète, on reçoit des petits événements
    ("thinking", "text"...) qu'on affiche dès qu'ils arrivent. À la fin, le SDK
    reconstitue le message complet (get_final_message), identique à ce que
    renverrait un appel normal : le reste de la boucle ne change pas.
    """
    api = client.messages if PROVIDER == "ollama" else client.beta.messages
    printer = LivePrinter()
    start = time.time()
    try:
        with api.stream(**request_params(messages)) as stream:
            for event in stream:
                if event.type == "thinking":
                    printer.show("thinking", event.thinking)
                elif event.type == "text":
                    printer.show("text", event.text)
            response = stream.get_final_message()
    finally:
        printer.end()
    seconds = time.time() - start

    usage = getattr(response, "usage", None)
    if usage:
        # Le total grossit à chaque appel : c'est tout l'historique qu'on renvoie.
        # Mais le serveur garde en CACHE le début déjà vu (prompt caching) : seule la
        # partie nouvelle est vraiment recalculée, d'où input_tokens qui reste petit.
        cached = (getattr(usage, "cache_read_input_tokens", 0) or 0)
        total = usage.input_tokens + cached + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        print(f"{DIM}  · {total} tokens envoyés (dont {cached} déjà en cache), "
              f"{usage.output_tokens} reçus, {seconds:.1f} s{RESET}")
    if trace:
        trace.log_call(messages, response, seconds)
    return response


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
        preview = f"{YELLOW}  $ {tool_input['command']}{RESET}"
        if tool_input.get("stdin") is not None:
            typed = tool_input["stdin"].splitlines()
            shown = ", ".join(typed[:10]) + (f", … ({len(typed)} saisies)" if len(typed) > 10 else "")
            preview += f"\n{YELLOW}  ⌨ saisies envoyées : {shown}{RESET}"
        return preview
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
    """ÉTAPE 5 : c'est le harness, pas le modèle, qui décide si une action a lieu."""
    print(_preview(name, tool_input))
    decision, reason = permissions.decide(name, tool_input)
    if decision == "deny":  # une interdiction l'emporte sur tout, même sur YOLO
        print(f"{RED}  ✗ interdit par la règle {reason}{RESET}")
        return False
    if decision == "allow":
        print(f"{DIM}  ✓ autorisé par la règle {reason}{RESET}")
        return True
    if YOLO:
        print(f"{DIM}  (MINICODE_YOLO=1 : accepté automatiquement){RESET}")
        return True
    if reason:  # ex : commande composée
        print(f"{DIM}  ({reason}){RESET}")

    remember = permissions.can_remember(name, tool_input)
    choices = "[o]ui / [t]oujours / [N]on" if remember else "[o]ui / [N]on"
    try:
        answer = input(f"{YELLOW}  Autoriser {name} ? {choices} {RESET}").strip().lower()
    except EOFError:
        return False
    if remember and answer in ("t", "toujours", "a", "always"):
        rule = permissions.suggest_rule(name, tool_input)
        permissions.add_allow_rule(rule)
        print(f"{DIM}  règle ajoutée dans {permissions.rules_file()} : {rule}{RESET}")
        return True
    return answer in ("o", "oui", "y", "yes")


def _harness_message(text):
    """Un message du harness (pas du modèle) : affiché en couleur ET renvoyé."""
    print(text)
    return text


def run_turn(client, messages, user_input, confirm=ask_permission, trace=None):
    """Traite une demande utilisateur : la BOUCLE D'AGENT (étape 3).

    Modifie `messages` sur place. Renvoie le texte final du modèle (déjà affiché
    pendant le streaming). `confirm(name, input) -> bool` est appelé avant chaque
    outil dangereux.
    """
    turn_start = len(messages)
    messages.append({"role": "user", "content": user_input})
    nudges = 0
    json_retries = 0

    for step in range(MAX_STEPS):
        try:
            response = call_model(client, messages, trace)
        except ValueError:
            # En streaming, les arguments d'un outil arrivent par morceaux de JSON.
            # S'ils sont illisibles, il n'y a pas de tool_use complet à qui répondre :
            # on refait simplement l'appel (au plus 2 fois de suite).
            json_retries += 1
            if json_retries > 2:
                raise
            print(f"{DIM}  (arguments d'outil illisibles, minicode relance l'appel){RESET}")
            continue
        json_retries = 0

        # Important : on ajoute `response.content` TEL QUEL (pas seulement le texte).
        # Il contient les blocs tool_use (dont l'API a besoin pour relier les
        # tool_result) et les blocs de réflexion, qu'on doit renvoyer sans les modifier.
        if response.stop_reason in ("refusal", "max_tokens"):
            # Réponse inutilisable (et peut-être un tool_use tronqué) : on annule
            # toute la demande pour garder un historique valide.
            del messages[turn_start:]
            return _harness_message(f"{RED}[arrêt : {response.stop_reason}] Demande annulée, reformule-la.{RESET}")

        messages.append({"role": "assistant", "content": response.content})

        text_parts = []
        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text.strip():  # un texte d'espaces = pas de réponse
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
            return "\n".join(text_parts) or _harness_message(f"{DIM}(le modèle n'a rien répondu){RESET}")

        # TOUS les résultats partent dans UN SEUL message "user".
        messages.append({"role": "user", "content": tool_results})

    return _harness_message(f"{RED}[arrêt : {MAX_STEPS} étapes atteintes sans réponse finale]{RESET}")


def main():
    client = make_client()
    messages = []  # TOUT l'état de la conversation tient dans cette liste
    trace = None
    if TRACE:
        trace = Trace(WORKSPACE / PROTECTED_DIR / "traces",
                      provider=PROVIDER, model=MODEL, system=SYSTEM_PROMPT, tools=TOOL_SCHEMAS)

    print(f"minicode — {PROVIDER} / {MODEL}, projet {WORKSPACE}")
    if trace:
        print(f"{DIM}journal : {trace.path}{RESET}")
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
            run_turn(client, messages, user_input, trace=trace)  # la réponse s'affiche en direct
            print()
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

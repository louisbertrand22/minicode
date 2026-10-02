"""Relit un journal de minicode (.minicode/traces/*.jsonl) de façon lisible.

    uv run show_trace.py                  # le journal le plus récent du dossier courant
    uv run show_trace.py FICHIER.jsonl    # un journal précis
    uv run show_trace.py --call 3         # la requête COMPLÈTE de l'appel n°3, en JSON brut

Pour chaque appel au modèle, on affiche ce qui a été AJOUTÉ à l'historique depuis
l'appel précédent : c'est la meilleure façon de voir la conversation grossir.
"""

import argparse
import json
from pathlib import Path

DIM, BOLD, CYAN, YELLOW, GREEN, RED, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[32m", "\033[31m", "\033[0m"


def short(text, limit=100):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def describe_block(block):
    """Une ligne lisible pour un bloc de contenu (texte, réflexion, appel d'outil...)."""
    kind = block.get("type")
    if kind == "text":
        return f"{GREEN}texte{RESET}       {short(block['text'])}"
    if kind == "thinking":
        return f"{DIM}réflexion   {short(block.get('thinking', ''))}{RESET}"
    if kind == "tool_use":
        return f"{YELLOW}tool_use{RESET}    {block['name']}({short(json.dumps(block['input'], ensure_ascii=False), 80)})"
    if kind == "tool_result":
        color = RED if block.get("is_error") else CYAN
        return f"{color}tool_result{RESET} {short(block.get('content', ''))}"
    return f"{kind}"


def describe_message(message):
    content = message["content"]
    if isinstance(content, str):
        return [f"{GREEN}texte{RESET}       {short(content)}"]
    return [describe_block(b) for b in content]


def latest_trace():
    files = sorted(Path(".minicode/traces").glob("*.jsonl"))
    if not files:
        raise SystemExit("Aucun journal dans .minicode/traces/ (lance minicode depuis ce dossier).")
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", type=Path)
    parser.add_argument("--call", type=int, help="affiche la requête complète de cet appel (JSON brut)")
    args = parser.parse_args()

    path = args.file or latest_trace()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    session = next(r for r in records if r["type"] == "session")
    calls = [r for r in records if r["type"] == "call"]
    # Étape 6 : le prompt système change si AGENTS.md est modifié pendant la session.
    system_at, system = [], session["system"]
    for record in records:
        if record["type"] == "system":
            system = record["system"]
        elif record["type"] == "call":
            system_at.append(system)

    if args.call:
        call = calls[args.call - 1]
        # Un sous-agent (étape 9) a son propre prompt, enregistré avec l'appel.
        print(json.dumps({"system": call.get("system") or system_at[args.call - 1], "tools": session["tools"],
                          "messages": call["request_messages"]}, ensure_ascii=False, indent=2))
        return

    print(f"{BOLD}{path}{RESET}")
    print(f"{session['provider']} / {session['model']} — {len(calls)} appels au modèle")
    print(f"{DIM}prompt système : {len(session['system'])} caractères, {len(session['tools'])} outils "
          f"(envoyés à CHAQUE appel ; voir --call N){RESET}\n")

    # Chaque agent a SA conversation (étape 9) : on suit ce qui est nouveau pour chacun.
    seen_by_agent = {}
    for i, call in enumerate(calls, 1):
        sent = call["request_messages"]
        agent = call.get("agent")
        seen = seen_by_agent.get(agent, 0)
        usage = call.get("usage") or {}
        cached = usage.get("cache_read_input_tokens") or 0
        total = (usage.get("input_tokens") or 0) + cached + (usage.get("cache_creation_input_tokens") or 0)
        who = f" ({agent})" if agent else ""
        print(f"{BOLD}── appel {i}{who}{RESET}  {len(sent)} messages envoyés · "
              f"{total} tokens (dont {cached} en cache) → {usage.get('output_tokens', '?')} tokens · "
              f"{call['seconds']} s · stop={call['stop_reason']}")
        if len(sent) < seen:
            print(f"{DIM}   (historique plus court qu'avant : nouvelle conversation, demande annulée ou contexte résumé){RESET}")
            seen = 0
        for message in sent[seen:]:  # seulement ce qui est NOUVEAU depuis l'appel précédent
            for line in describe_message(message):
                print(f"   + {message['role']:<9} {line}")
        for block in call["response_content"]:
            print(f"   ← modèle    {describe_block(block)}")
        # au prochain appel, la réponse du modèle fera partie de l'historique envoyé
        seen_by_agent[agent] = len(sent) + 1
        print()


if __name__ == "__main__":
    main()

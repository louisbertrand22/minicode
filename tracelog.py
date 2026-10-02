"""Étape 7 : le journal (trace) de chaque échange avec le modèle.

Un fichier JSONL = une ligne JSON par événement :
  - 1re ligne  : {"type": "session", ...} le prompt système et les outils envoyés ;
  - ensuite    : {"type": "call", ...} pour CHAQUE appel au modèle, avec la liste
                 `messages` complète telle qu'envoyée et la réponse reçue.

On enregistre l'historique complet à chaque appel, exprès : c'est exactement ce
que le modèle reçoit, et tu verras le fichier grossir à chaque tour. C'est la
meilleure façon de comprendre "le modèle n'a pas de mémoire".

Relire une trace :  uv run show_trace.py      (ou avec jq, voir le README)
"""

import json
import time
from datetime import datetime
from pathlib import Path


def to_jsonable(obj):
    """Convertit les objets du SDK (blocs de réponse...) en JSON simple."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):  # autre objet simple : on prend ses attributs
        return to_jsonable(vars(obj))
    return obj


class Trace:
    def __init__(self, directory: Path, **session_info):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
        self._write({"type": "session", "ts": time.time(), **session_info})

    def _write(self, record):
        with self.path.open("a") as f:
            # default=str : un journal ne doit jamais faire planter l'agent
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False, default=str) + "\n")

    def log_call(self, messages, response, seconds, agent=None, system=None):
        """`agent` : None pour l'agent principal, "sous-agent" pour l'étape 9 (avec son prompt)."""
        usage = getattr(response, "usage", None)
        extra = {"agent": agent, "system": system} if agent else {}
        self._write({
            **extra,
            "type": "call",
            "ts": time.time(),
            "seconds": round(seconds, 2),
            "request_messages": messages,           # TOUT ce qui a été envoyé
            "response_content": response.content,   # TOUT ce qui a été reçu
            "stop_reason": response.stop_reason,
            "usage": usage,
        })

    def log_system(self, system):
        """Étape 6 : AGENTS.md a changé, le prompt système envoyé aussi."""
        self._write({"type": "system", "ts": time.time(), "system": system})

    def log_compact(self, how, tokens_before, tokens_after, summary=None):
        """Étape 8 : le harness a fait de la place dans l'historique (voir context.py)."""
        self._write({"type": "compact", "ts": time.time(), "how": how,
                     "tokens_before": tokens_before, "tokens_after": tokens_after, "summary": summary})

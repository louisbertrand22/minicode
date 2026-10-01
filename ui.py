"""L'interface terminal de minicode, inspirée de Claude Code.

Tout l'AFFICHAGE est ici ; la boucle d'agent (minicode.py) ne fait qu'appeler
ces méthodes. C'est une séparation importante : le harness ne dépend pas de
l'interface. On pourrait remplacer ce fichier par une page web sans toucher à
la boucle.

Deux bibliothèques :
  - rich           : couleurs, cadres, Markdown, animation pendant que le modèle travaille ;
  - prompt_toolkit : zone de saisie (historique ↑/↓, complétion des /commandes, menus).

Quand minicode ne tourne pas dans un vrai terminal (tests, `echo ... | minicode`),
on retombe sur input()/print() simples.
"""

import sys
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import choice
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

import tools

ACCENT = "#D97757"  # l'orange de Claude
TOOL_LABELS = {"read_file": "Read", "list_dir": "List", "grep": "Search", "edit_file": "Update", "bash": "Bash",
               "interactive_start": "Run", "interactive_send": "Type"}
SLASH_COMMANDS = {
    "/help": "afficher cette aide",
    "/clear": "nouvelle conversation (vide l'historique envoyé au modèle)",
    "/trace": "chemin du journal de la session",
    "/exit": "quitter (ou Ctrl-D)",
}


def _bullet(renderable, style="white"):
    """Une ligne « ⏺ contenu », comme les messages de Claude Code."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=1)
    grid.add_column()
    grid.add_row(Text("⏺", style=style), renderable)
    return grid


def _result_line(renderable):
    """La ligne « ⎿ résultat » sous un appel d'outil."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=4)
    grid.add_column()
    grid.add_row(Text("  ⎿", style="dim"), renderable)
    return grid


def edit_diff(tool_input, context=2):
    """Le diff d'un edit_file AVANT qu'il soit appliqué : liste de (n° de ligne, signe, texte)."""
    path = tools.WORKSPACE / tool_input["path"]
    old, new = tool_input["old_string"], tool_input["new_string"]
    content = path.read_text(errors="replace") if path.is_file() else ""
    if old == "":
        return [(i, "+", line) for i, line in enumerate(new.splitlines(), 1)]
    index = content.find(old)
    if index < 0:
        return [(None, "-", line) for line in old.splitlines()] + [(None, "+", line) for line in new.splitlines()]
    lines = content.splitlines()
    start = content[:index].count("\n") + 1  # 1re ligne modifiée (numérotation à partir de 1)
    # Le modèle donne souvent un BOUT de ligne (sans l'indentation) : on affiche les
    # lignes entières, avant et après, pour que le diff corresponde au vrai fichier.
    line_start = content.rfind("\n", 0, index) + 1
    line_end = content.find("\n", index + len(old))
    line_end = len(content) if line_end < 0 else line_end
    old_lines = content[line_start:line_end].splitlines()
    new_lines = (content[line_start:index] + new + content[index + len(old):line_end]).splitlines()
    rows = [(n, " ", lines[n - 1]) for n in range(max(1, start - context), start)]
    rows += [(start + i, "-", line) for i, line in enumerate(old_lines)]
    rows += [(start + i, "+", line) for i, line in enumerate(new_lines)]
    after = start + len(old_lines)  # 1re ligne inchangée après le bloc (ancienne numérotation)
    shift = len(new_lines) - len(old_lines)
    rows += [(n + shift, " ", lines[n - 1]) for n in range(after, min(len(lines), after + context - 1) + 1)]
    return rows


def render_diff(rows, max_rows=30):
    text = Text()
    styles = {"-": "red on #3b1d1d", "+": "green on #1d3b24", " ": "dim"}
    for lineno, sign, line in rows[:max_rows]:
        number = f"{lineno:>4}" if lineno else "    "
        text.append(f"{number} {sign} {line}\n", style=styles[sign])
    if len(rows) > max_rows:
        text.append(f"     … {len(rows) - max_rows} lignes de plus\n", style="dim")
    text.rstrip()  # pas de ligne vide en trop à la fin
    return text


class StreamView:
    """L'affichage d'UN appel au modèle pendant le streaming.

    Pendant la génération : une animation « ✻ Réflexion… 12 s » (avec les dernières
    lignes de la réflexion du modèle), puis la réponse qui s'écrit. Tout ça est
    temporaire (transient) : à la fin, on efface et on imprime la réponse finale
    proprement, en Markdown.
    """

    def __init__(self, ui):
        self.ui = ui
        self.thinking = ""
        self.text = ""
        self.start = time.time()
        self.spinner = Spinner("dots", style=ACCENT)
        self.live = Live(console=ui.console, get_renderable=self._render, refresh_per_second=8,
                         transient=True, vertical_overflow="crop")

    def __enter__(self):
        if self.ui.interactive:  # pas d'animation dans un fichier ou un tube (|) : juste le résultat
            self.live.start()
        return self

    def __exit__(self, *exc):
        if self.ui.interactive:
            self.live.stop()
        if self.text.strip():
            self.ui.console.print(_bullet(Markdown(self.text.strip())))
        return False

    def on_thinking(self, chunk):
        self.thinking += chunk

    def on_text(self, chunk):
        self.text += chunk

    def _render(self):
        elapsed = time.time() - self.start
        produced = (len(self.thinking) + len(self.text)) // 4  # ~4 caractères par token
        status = "Rédaction…" if self.text else "Réflexion…"
        self.spinner.update(text=Text(f"{status} ({elapsed:.0f} s · ↓ ~{produced} tokens · ctrl+c pour interrompre)",
                                      style=ACCENT))
        parts = []
        if self.text:
            # Seulement la fin de la réponse : ce qui dépasse l'écran ne peut pas être effacé ensuite.
            height = max(5, self.ui.console.height - 4)
            tail = self.text.splitlines()[-height:]
            hidden = "\n".join(self.text.splitlines()[:-height])
            prefix = "```\n" if hidden.count("```") % 2 else ""  # on est au milieu d'un bloc de code
            parts.append(_bullet(Markdown(prefix + "\n".join(tail))))
        elif self.ui.show_thinking and self.thinking.strip():
            last = [line for line in self.thinking.splitlines() if line.strip()][-3:]
            width = max(20, self.ui.console.width - 6)
            parts.append(Text("\n".join("  " + line[:width] for line in last), style="dim italic"))
        parts.append(self.spinner)
        return Group(*parts)


class TerminalUI:
    def __init__(self, provider="", model="", show_thinking=True, history_file=None):
        self.console = Console(highlight=False)
        self.provider, self.model = provider, model
        self.show_thinking = show_thinking
        self.interactive = sys.stdin.isatty() and sys.stdout.isatty()
        self.context_tokens = self.cached_tokens = 0
        self.calls = 0
        self.session = None
        if self.interactive:
            self.session = PromptSession(history=FileHistory(str(history_file)) if history_file else None)

    # --- accueil et saisie -------------------------------------------------------

    def welcome(self, workspace, trace_path=None):
        body = Text.assemble(("✻ ", ACCENT), ("Bienvenue dans ", ""), ("minicode", "bold"), (" !\n\n", ""),
                             ("  /help pour l'aide, /clear pour une nouvelle conversation\n\n", "dim"),
                             (f"  dossier : {workspace}\n", "dim"),
                             (f"  modèle  : {self.provider} / {self.model}", "dim"))
        if trace_path:
            body.append(f"\n  journal : {trace_path}", style="dim")
        self.console.print(Panel(body, border_style=ACCENT, expand=False))
        self.console.print()

    def _toolbar(self):
        context = f"contexte {self.context_tokens} tokens ({self.cached_tokens} en cache)" if self.calls else "nouvelle conversation"
        return f" {self.provider} / {self.model} · {context} · /help"

    def read_input(self):
        """Renvoie la demande de l'utilisateur, ou None pour quitter."""
        if not self.interactive:
            try:
                return input("> ")
            except EOFError:
                return None
        while True:
            try:
                return self.session.prompt(
                    "> ", show_frame=True, bottom_toolbar=self._toolbar, placeholder="Tape ta demande, ou /help",
                    completer=WordCompleter(list(SLASH_COMMANDS), sentence=True), complete_while_typing=True,
                )
            except KeyboardInterrupt:
                continue  # Ctrl-C efface la ligne ; Ctrl-D quitte
            except EOFError:
                return None

    def help(self):
        table = Table.grid(padding=(0, 2))
        for command, description in SLASH_COMMANDS.items():
            table.add_row(Text(command, style=ACCENT), description)
        self.console.print(Panel(Group(
            table, Text(),
            Text("Ctrl-C pendant une réponse : interrompre le modèle\n"
                 "Variables : MINICODE_MODEL, MINICODE_PROVIDER, MINICODE_THINKING=0, MINICODE_YOLO=1", style="dim"),
        ), title="aide", border_style=ACCENT, expand=False))

    # --- un appel au modèle ---------------------------------------------------------

    def model_call(self):
        return StreamView(self)

    def record_usage(self, usage):
        if usage is None:
            return
        self.calls += 1
        self.cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        self.context_tokens = (usage.input_tokens + self.cached_tokens
                               + (getattr(usage, "cache_creation_input_tokens", 0) or 0))

    def turn_done(self, calls, seconds):
        self.console.print(Text(f"✻ Terminé en {seconds:.0f} s · {calls} appel(s) au modèle · "
                                f"contexte {self.context_tokens} tokens ({self.cached_tokens} en cache)",
                                style="dim"))
        self.console.print()

    def reset(self):
        self.context_tokens = self.cached_tokens = self.calls = 0

    # --- outils -----------------------------------------------------------------

    @staticmethod
    def _label(name, tool_input):
        if name == "edit_file" and tool_input.get("old_string") == "":
            return "Create"
        return TOOL_LABELS.get(name, name)

    @staticmethod
    def _argument(name, tool_input):
        if name in ("bash", "interactive_start"):
            arg = tool_input.get("command", "").splitlines()[0] if tool_input.get("command") else ""
        elif name == "interactive_send":
            arg = f"{tool_input.get('text', '')} → session {tool_input.get('session_id', '?')}"
        elif name == "grep":
            arg = f"\"{tool_input.get('pattern', '')}\" dans {tool_input.get('path', '.')}"
        else:
            arg = tool_input.get("path", "")
        return arg if len(arg) <= 80 else arg[:79] + "…"

    def tool_call(self, name, tool_input):
        self.console.print(_bullet(Text.assemble((self._label(name, tool_input), "bold"),
                                                 f"({self._argument(name, tool_input)})"), style="green"))

    @staticmethod
    def _output_summary(lines, max_lines=5):
        """Les premières lignes d'une sortie, PLUS toutes les notes du harness.

        Les notes [indice minicode …], [code de sortie …], [session …] sont ce que le
        harness dit au modèle : on les montre toujours, même quand la sortie est coupée.
        """
        notes = [line for line in lines if line.startswith("[")]
        output = [line for line in lines if not line.startswith("[") and line.strip()]
        summary = Text("\n".join(output[:max_lines]))
        if len(output) > max_lines:
            summary.append(f"\n… +{len(output) - max_lines} lignes", style="dim")
        for note in notes:
            if note.startswith("[indice minicode"):
                style = "yellow"
            elif "code de sortie" in note and not note.endswith(": 0]"):
                style = "red"
            else:
                style = "dim"
            summary.append(("\n" if summary.plain else "") + note, style=style)
        return summary

    def tool_result(self, name, tool_input, result, is_error, diff_rows=None):
        lines = result.splitlines()
        if is_error:
            # Le message d'erreur en entier (les détails utiles sont souvent après la 1re ligne :
            # « dans le fichier : … / dans ton texte : … »), les indices en jaune.
            summary = Text()
            for line in lines[:8] or ["erreur"]:
                style = "yellow" if line.startswith("[indice minicode") else "red"
                summary.append(("\n" if summary.plain else "") + line, style=style)
            self.console.print(_result_line(summary))
            return
        if name == "read_file":
            summary = Text(f"{len(lines)} lignes lues")
        elif name == "list_dir":
            summary = Text(f"{len(lines)} éléments")
        elif name == "grep":
            summary = Text("aucun résultat" if result == "Aucun résultat." else f"{len(lines)} résultats")
        elif name == "edit_file":
            added = sum(1 for _, s, _ in diff_rows or [] if s == "+")
            removed = sum(1 for _, s, _ in diff_rows or [] if s == "-")
            summary = Group(Text(f"{tool_input['path']} : +{added} −{removed} lignes"), render_diff(diff_rows or []))
        elif name in ("bash", "interactive_start", "interactive_send"):
            summary = self._output_summary(lines)
        else:
            summary = Text(lines[0] if lines else "")
        self.console.print(_result_line(summary))

    # --- permissions ---------------------------------------------------------------

    def permission(self, name, tool_input, rule=None):
        """La boîte « Voulez-vous continuer ? ». Renvoie ("yes" | "always" | "no", consigne)."""
        if name in ("bash", "interactive_start"):
            title = "Commande bash" if name == "bash" else "Programme interactif"
            body = Text(f"  {tool_input['command']}", style="bold")
            if tool_input.get("stdin") is not None:
                typed = tool_input["stdin"].splitlines()
                shown = ", ".join(typed[:10]) + (f", … ({len(typed)} saisies)" if len(typed) > 10 else "")
                body.append(f"\n  ⌨ saisies envoyées : {shown}", style="dim")
        else:
            title = ("Créer " if tool_input.get("old_string") == "" else "Modifier ") + tool_input["path"]
            body = render_diff(edit_diff(tool_input))
        self.console.print(Panel(body, title=title, title_align="left", border_style=ACCENT))

        options = [("yes", "Oui")]
        if rule:
            options.append(("always", f"Oui, et ne plus demander pour {rule}"))
        options.append(("no", "Non, et dire à minicode quoi faire autrement"))

        if self.interactive:
            try:
                answer = choice("Voulez-vous continuer ?", options=options, default="yes")
            except KeyboardInterrupt:
                answer = "no"
        else:
            keys = {"1": "yes", "o": "yes", "oui": "yes", "y": "yes",
                    "2": "always" if rule else "no", "t": "always" if rule else "no",
                    "3": "no", "n": "no", "non": "no"}
            menu = " / ".join(f"[{i}] {label}" for i, (_, label) in enumerate(options, 1))
            try:
                answer = keys.get(input(f"Voulez-vous continuer ? {menu} ").strip().lower(), "no")
            except EOFError:
                answer = "no"
        if answer == "always" and len(options) == 2:
            answer = "no"  # « 2 » sans option « toujours » = Non

        feedback = None
        if answer == "no":
            try:
                if self.interactive:
                    feedback = self.session.prompt("Que doit faire minicode à la place ? (Entrée pour rien) ").strip()
                else:
                    feedback = input("Que doit faire minicode à la place ? ").strip()
            except (EOFError, KeyboardInterrupt):
                feedback = ""
        return answer, feedback or None

    # --- messages du harness ---------------------------------------------------------

    def info(self, message):
        self.console.print(Text(f"  {message}", style="dim"))

    def warn(self, message):
        self.console.print(Text(f"  {message}", style="yellow"))

    def error(self, message):
        self.console.print(Text(f"  {message}", style="red"))

"""Les outils de l'agent.

Un outil, c'est deux choses :
  1. un SCHEMA (nom + description + JSON Schema des paramètres) qu'on envoie au
     modèle pour qu'il sache que l'outil existe et comment l'appeler ;
  2. une FONCTION Python que *notre* code exécute quand le modèle le demande.

Le modèle n'exécute jamais rien lui-même : il renvoie un bloc `tool_use`
("je voudrais appeler read_file avec path=..."), et c'est le harness qui
décide de l'exécuter et de renvoyer le résultat.
"""

import itertools
import json
import os
import pty
import re
import select
import signal
import subprocess
import termios
import time
from pathlib import Path

# Racine du projet sur lequel l'agent travaille. Les outils de fichiers refusent
# d'en sortir. `bash`, lui, peut tout faire : d'où la permission demandée avant.
WORKSPACE = Path.cwd().resolve()

# On ne met pas un fichier de 5 Mo dans le contexte. minicode.py abaisse ce plafond selon
# la taille de la fenêtre de contexte (étape 8, voir context.output_limit).
MAX_OUTPUT_CHARS = 20_000
MAX_GREP_MATCHES = 100
MAX_EDIT_LINES = 15  # un old_string plus long = le modèle essaie de recopier tout le fichier
BASH_TIMEOUT = 60  # secondes ; une commande bloquée ne doit pas figer l'agent
IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".tox",
               "target", "dist", "build", ".next", ".gradle", ".claude", ".minicode"}
# Dossier des réglages de minicode (règles de permission). L'agent ne doit pas
# pouvoir y écrire, sinon il pourrait s'autoriser lui-même n'importe quoi.
PROTECTED_DIR = ".minicode"

# Outils qui modifient le disque ou exécutent du code : le harness demande la
# permission à l'utilisateur avant de les lancer (voir minicode.py).
DANGEROUS_TOOLS = {"edit_file", "bash", "interactive_start"}
# interactive_send n'y est pas : il ne fait que taper dans un programme déjà autorisé.


class ToolError(Exception):
    """Erreur renvoyée au modèle (is_error=True) au lieu de faire planter le harness."""


def _resolve(path: str) -> Path:
    p = (WORKSPACE / path).resolve()
    if not p.is_relative_to(WORKSPACE):
        raise ToolError(f"Accès refusé : {path} est en dehors du workspace {WORKSPACE}")
    return p


def read_file(path: str, start_line: int = 1) -> str:
    p = _resolve(path)
    if not p.is_file():
        raise ToolError(f"Fichier introuvable : {path}")
    text = p.read_text(errors="replace")
    if not text.strip():
        # Un résultat vide est ambigu pour le modèle ; on le dit explicitement.
        return "(fichier vide : pour l'écrire, utilise edit_file avec old_string='')"
    lines = text.splitlines()
    start = max(1, int(start_line))
    if start > len(lines):
        raise ToolError(f"{path} n'a que {len(lines)} lignes.")
    # Étape 8 : un gros fichier se lit par MORCEAUX de lignes entières, au lieu d'être
    # coupé au milieu ; le modèle sait où reprendre. Numéroter les lignes l'aide aussi
    # à citer / éditer précisément.
    out, size = [], 0
    for n in range(start, len(lines) + 1):
        line = f"{n:>5}\t{lines[n - 1]}"[:MAX_OUTPUT_CHARS]
        if out and size + len(line) > MAX_OUTPUT_CHARS:
            break
        out.append(line)
        size += len(line) + 1
    last = start + len(out) - 1
    if last < len(lines):
        out.append(f"... [lignes {start}-{last} sur {len(lines)}. Suite : read_file avec start_line={last + 1}]")
    elif start > 1:
        out.append(f"... [lignes {start}-{last} sur {len(lines)} : fin du fichier]")
    return "\n".join(out)


def list_dir(path: str = ".") -> str:
    p = _resolve(path)
    if not p.is_dir():
        raise ToolError(f"Dossier introuvable : {path}")
    entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
    lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries if e.name != ".git"]
    return "\n".join(lines) or "(dossier vide)"


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n... [tronqué, {len(text)} caractères au total]"
    return text


def grep(pattern: str, path: str) -> str:
    """Cherche une regex dans les fichiers texte. Évite de lire tout le projet."""
    root = _resolve(path)
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"Regex invalide : {e}")
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    matches = []
    for f in files:
        if not f.is_file() or IGNORED_DIRS.intersection(f.relative_to(WORKSPACE).parts):
            continue
        try:
            lines = f.read_text().splitlines()
        except (UnicodeDecodeError, OSError):
            continue  # fichier binaire ou illisible
        for i, line in enumerate(lines, 1):
            if regex.search(line):
                matches.append(f"{f.relative_to(WORKSPACE)}:{i}: {line.strip()}")
                if len(matches) >= MAX_GREP_MATCHES:
                    return "\n".join(matches) + f"\n... [arrêté à {MAX_GREP_MATCHES} résultats, affine ta recherche]"
    return "\n".join(matches) or "Aucun résultat."


def _mismatch_hint(text: str, old: str) -> str:
    """Dit au modèle OÙ son old_string cesse de correspondre au fichier.

    « Introuvable » tout court ne l'aide pas à se corriger ; « ligne 7 : le fichier
    dit X, toi tu as écrit Y » oui. On cherche le plus long début de old_string
    présent dans le fichier (recherche dichotomique sur la longueur).
    """
    low, high = 0, len(old)
    while low < high:
        mid = (low + high + 1) // 2
        if old[:mid] in text:
            low = mid
        else:
            high = mid - 1
    if low == 0:
        return "Même son début n'existe pas dans le fichier : relis-le avec read_file."
    end = text.find(old[:low]) + low  # où le fichier et old_string se séparent
    line_no = text[:end].count("\n") + 1
    file_line = text.splitlines()[line_no - 1] if line_no <= len(text.splitlines()) else ""
    your_line = (old[:low].rsplit("\n", 1)[-1] + old[low:]).split("\n", 1)[0]
    return (f"Ton old_string correspond au fichier jusqu'à la ligne {line_no}, puis diffère :\n"
            f"  dans le fichier : {file_line!r}\n  dans ton texte  : {your_line!r}")


def _minimal_hunk(old: str, new: str):
    """Réduit une modification aux seules lignes qui changent (+ 1 ligne de contexte).

    Ex : 18 lignes recopiées pour ajouter `current_attempts = 0` avant `while True:`
    -> old = ["while True:"], new = ["current_attempts = 0", "while True:"].
    """
    a, b = old.split("\n"), new.split("\n")
    start = 0
    while start < min(len(a), len(b)) and a[start] == b[start]:
        start += 1
    end = 0
    while end < min(len(a), len(b)) - start and a[-1 - end] == b[-1 - end]:
        end += 1
    before = 1 if start > 0 else 0              # une ligne de contexte avant…
    after = 1 if not before and end > 0 else 0  # …ou après, s'il n'y a rien avant
    a_hunk = a[start - before:len(a) - end + after]
    b_hunk = b[start - before:len(b) - end + after]
    if not any(line.strip() for line in a_hunk):
        return None
    return a_hunk, b_hunk


def _find_in_file(text: str, old_lines, new_lines):
    """Retrouve old_lines dans le fichier, même avec une autre indentation, et
    ré-indente new_lines pareil. Renvoie (old_string, new_string) exacts, ou None."""
    exact = "\n".join(old_lines)
    if text.count(exact) == 1:
        return exact, "\n".join(new_lines)
    file_lines, n = text.split("\n"), len(old_lines)
    found = [i for i in range(len(file_lines) - n + 1)
             if all(file_lines[i + k].strip() == old_lines[k].strip() for k in range(n))]
    if len(found) != 1:
        return None  # introuvable ou ambigu : on ne devine pas
    i = found[0]
    model_indent = old_lines[0][:len(old_lines[0]) - len(old_lines[0].lstrip())]
    file_indent = file_lines[i][:len(file_lines[i]) - len(file_lines[i].lstrip())]

    def reindent(line):
        return file_indent + line[len(model_indent):] if line.startswith(model_indent) else line

    return "\n".join(file_lines[i:i + n]), "\n".join(reindent(line) for line in new_lines)


def _ready_made_edit(path: str, text: str, old: str, new: str) -> str:
    """Propose au modèle l'appel edit_file correct, PRÊT À RECOPIER.

    Les petits modèles n'arrivent pas à « changer d'approche » seuls (qwen3 a refait
    5 fois le même appel raté), mais ils recopient très bien ce qu'on leur montre.
    On ne propose que ce qu'on a vérifié : trouvé une seule fois, et fichier toujours valide.
    """
    hunk = _minimal_hunk(old, new)
    suggestion = hunk and _find_in_file(text, *hunk)
    if not suggestion:
        return ""
    old_s, new_s = suggestion
    if text.count(old_s) != 1 or old_s == new_s:
        return ""
    try:
        _check_still_valid(path, text, text.replace(old_s, new_s, 1))
    except ToolError:
        return ""
    return ("\n[indice minicode : ta modification tient en quelques lignes. Appelle edit_file avec exactement "
            f"ces arguments : old_string={json.dumps(old_s, ensure_ascii=False)} "
            f"new_string={json.dumps(new_s, ensure_ascii=False)}]")


def plan_edit(path: str, old_string: str, new_string: str) -> tuple[Path, str, str]:
    """Prépare un edit_file SANS rien écrire : renvoie (fichier, nouveau contenu, message).

    Lève ToolError si la modification ne peut pas marcher. Le harness l'appelle
    AVANT de demander la permission : inutile de déranger l'utilisateur pour une
    modification qui va échouer de toute façon.
    """
    p = _resolve(path)
    if p.is_relative_to(WORKSPACE / PROTECTED_DIR):
        raise ToolError(f"{PROTECTED_DIR}/ est protégé : seul l'utilisateur peut modifier les réglages de minicode.")
    if old_string == "":
        # Un fichier existant mais VIDE compte comme "à créer" : sinon aucun
        # old_string ne peut jamais y être trouvé et le modèle reste bloqué.
        if p.exists() and p.read_text().strip():
            raise ToolError(f"{path} existe déjà et n'est pas vide : lis-le, puis donne un old_string pour le modifier "
                            "(pour ajouter au début, old_string = la 1re ligne actuelle, new_string = ajout + cette ligne).")
        return p, new_string, f"Fichier créé : {path} ({len(new_string.splitlines())} lignes)"
    if not p.is_file():
        raise ToolError(f"Fichier introuvable : {path}")
    text = p.read_text()
    if old_string.count("\n") >= MAX_EDIT_LINES:
        # Vu en vrai : qwen3 réécrivait tout le fichier de mémoire. Résultat : une ligne
        # oubliée (bug introduit), puis des erreurs de recopie (« 时间_limit » au lieu de
        # « time_limit ») qui empêchaient toute correction.
        raise ToolError(
            f"old_string fait {old_string.count(chr(10)) + 1} lignes : ne recopie pas tout le fichier. "
            "Modifie seulement la ou les lignes concernées (avec au plus une ligne autour pour la "
            "situer), quitte à faire plusieurs edit_file." + _ready_made_edit(path, text, old_string, new_string)
        )
    count = text.count(old_string)
    if count == 0:
        raise ToolError(
            "old_string introuvable dans le fichier. " + _mismatch_hint(text, old_string) +
            "\nRecopie le texte exactement (espaces et indentation compris, sans les numéros de ligne)."
            + _ready_made_edit(path, text, old_string, new_string)
        )
    if count > 1:
        raise ToolError(f"old_string apparaît {count} fois : ajoute des lignes autour pour qu'il soit unique.")
    new_text = text.replace(old_string, new_string, 1)
    _check_still_valid(path, text, new_text)
    return p, new_text, f"Modifié : {path}"


# Les formats dont minicode sait vérifier la syntaxe SANS rien installer (bibliothèque
# standard de Python), avec un conseil adapté. Ajouter un langage = ajouter une ligne
# (un vrai harness lancerait le compilateur ou le linter du projet).
SYNTAX_HINTS = {
    ".py": "Vérifie l'indentation de new_string (elle doit être la même que celle des lignes d'origine).",
    ".json": "Vérifie les virgules (pas de virgule après le dernier élément), les guillemets doubles et les accolades.",
}


def _syntax_error(path: str, code: str):
    """(numéro de ligne, type d'erreur, message) si `code` est invalide pour son format, sinon None.

    Un format qu'on ne sait pas vérifier est considéré comme valide.
    """
    if path.endswith(".py"):
        try:
            compile(code, path, "exec")
        except SyntaxError as e:
            return e.lineno, type(e).__name__, e.msg
    elif path.endswith(".json") and code.strip():
        try:
            json.loads(code)
        except json.JSONDecodeError as e:
            return e.lineno, "JSON invalide", e.msg
    return None


def _check_still_valid(path: str, before: str, after: str) -> None:
    """Refuse une modification qui CASSE un fichier (Python, JSON) qui était valide.

    Vu en vrai : qwen3 a « corrigé » un bug en perdant l'indentation d'une ligne ;
    le fichier ne se lançait plus (IndentationError) et il a annoncé « corrigé ».
    Les vrais harness lancent ce genre de vérification (compilateur, linter) après
    chaque modification et renvoient le résultat au modèle.
    """
    if _syntax_error(path, before):
        return  # déjà cassé avant : on ne bloque pas une tentative de réparation
    error = _syntax_error(path, after)
    if error:
        lineno, kind, message = error
        lines = after.splitlines()
        line = lines[lineno - 1] if lineno and lineno <= len(lines) else ""
        hint = next((h for ext, h in SYNTAX_HINTS.items() if path.endswith(ext)), "")
        raise ToolError(
            f"Modification refusée : elle casserait {path} ({kind} ligne {lineno} : {message}).\n"
            f"  ligne {lineno} après ta modification : {line!r}\n"
            f"Le fichier n'a pas été modifié. {hint}"
        )


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Remplace un texte EXACT par un autre. old_string vide = créer un nouveau fichier.

    Pourquoi pas "réécrire tout le fichier" ? Parce que le modèle devrait alors
    recopier des centaines de lignes sans erreur (lent, cher, risqué). Avec un
    remplacement exact, il n'écrit que ce qui change, et si old_string ne
    correspond pas, on le sait tout de suite au lieu d'abîmer le fichier.
    """
    p, content, message = plan_edit(path, old_string, new_string)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return message


INTERACTIVE_IDLE = 0.5        # s sans nouvelle sortie => le programme attend sans doute une saisie
INTERACTIVE_FIRST_OUTPUT = 3  # s d'attente max pour la toute première sortie
INTERACTIVE_MAX_WAIT = 15     # s max par lecture (un programme qui parle sans arrêt ne bloque pas l'agent)
MAX_SESSIONS = 3
_sessions = {}                # id -> (processus, descripteur du pseudo-terminal)
_next_id = itertools.count(1)


def _read_until_idle(proc, fd):
    """Lit ce qu'affiche le programme jusqu'à ce qu'il se taise (ou se termine).

    Un programme ne prévient pas qu'il attend une saisie : on le DEVINE quand il
    n'affiche plus rien pendant INTERACTIVE_IDLE secondes. C'est une heuristique,
    comme dans tous les outils de ce genre (pexpect, tmux...).
    """
    chunks, start = [], time.time()
    last = start
    while time.time() - start < INTERACTIVE_MAX_WAIT:
        ready, _, _ = select.select([fd], [], [], 0.05)
        if ready:
            try:
                data = os.read(fd, 4096)
            except OSError:  # le programme est fini et le pseudo-terminal fermé
                data = b""
            if not data:
                # L'affichage est fermé : le programme se termine. Son processus peut
                # mettre quelques millisecondes à disparaître : on l'attend, sinon on
                # dirait à tort « session toujours ouverte » (bug vu en vrai).
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                break
            chunks.append(data)
            last = time.time()
            continue
        if proc.poll() is not None:
            break
        waited = time.time() - last
        if (chunks and waited >= INTERACTIVE_IDLE) or waited >= INTERACTIVE_FIRST_OUTPUT:
            break
    text = b"".join(chunks).decode(errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _session_status(session_id):
    proc, fd = _sessions[session_id]
    if proc.poll() is None:
        return (f"[session {session_id} toujours ouverte : le programme attend peut-être une saisie. "
                f"Utilise interactive_send(session_id=\"{session_id}\", text=...)]")
    os.close(fd)
    del _sessions[session_id]
    return f"[programme terminé, code de sortie : {proc.returncode}]"


def interactive_start(command: str) -> str:
    """Lance un programme dans un pseudo-terminal et renvoie ce qu'il affiche au début.

    Pseudo-terminal : sans lui, Python (et bien d'autres) garde ses print() en
    mémoire tant qu'il n'écrit pas dans un "vrai" terminal, et on ne verrait
    jamais la question "Entrez un nombre :".
    """
    if len(_sessions) >= MAX_SESSIONS:
        raise ToolError(f"Déjà {MAX_SESSIONS} sessions ouvertes : termine-les (ou attends la fin de la demande).")
    master, slave = pty.openpty()
    attrs = termios.tcgetattr(slave)
    attrs[3] &= ~termios.ECHO  # le terminal ne répète pas ce qu'on tape : on sait déjà ce qu'on a envoyé
    termios.tcsetattr(slave, termios.TCSANOW, attrs)
    proc = subprocess.Popen(command, shell=True, cwd=WORKSPACE, stdin=slave, stdout=slave, stderr=slave,
                            start_new_session=True)
    os.close(slave)
    session_id = str(next(_next_id))
    _sessions[session_id] = (proc, master)
    output = _read_until_idle(proc, master)
    return _truncate(f"[session {session_id}]\n{output}\n{_session_status(session_id)}")


def interactive_send(session_id: str, text: str) -> str:
    """Tape une ligne dans un programme lancé par interactive_start, et renvoie sa réponse."""
    if session_id not in _sessions:
        raise ToolError(f"Session {session_id} inconnue ou déjà terminée. Relance le programme avec interactive_start.")
    proc, fd = _sessions[session_id]
    os.write(fd, (text + "\n").encode())
    output = _read_until_idle(proc, fd)
    return _truncate(f"{output}\n{_session_status(session_id)}")


def stop_all_sessions():
    """Arrête les programmes encore ouverts (appelé à la fin de chaque demande)."""
    for proc, fd in _sessions.values():
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # tout le groupe : le shell ET le programme
            except ProcessLookupError:
                pass
            proc.wait()
        os.close(fd)
    _sessions.clear()


def bash(command: str, stdin: str | None = None) -> str:
    """Lance une commande shell dans le dossier du projet.

    Pas de clavier : un agent ne peut pas taper au milieu d'une commande. Si
    `stdin` est donné, ce texte est envoyé au programme comme si on le tapait
    (une ligne par saisie) ; sinon le programme reçoit "fin de saisie" tout de
    suite, au lieu de bloquer l'agent indéfiniment.
    """
    feed = {"input": stdin} if stdin is not None else {"stdin": subprocess.DEVNULL}
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKSPACE, capture_output=True, text=True,
            timeout=BASH_TIMEOUT, **feed,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"Commande arrêtée après {BASH_TIMEOUT} s (trop longue ou bloquée).")
    output = (r.stdout + r.stderr).strip() or "(aucune sortie)"
    if "EOFError" in output:
        # On aide le modèle à comprendre l'erreur, au lieu de le laisser deviner
        # (sans indice, il concluait que le programme était buggé).
        if stdin is None:
            output += ("\n[indice minicode : ce programme attend des saisies au clavier. Relance la "
                       "commande avec le paramètre stdin (une ligne par saisie).]")
        else:
            output += (f"\n[indice minicode : le programme a demandé plus de saisies que les "
                       f"{len(stdin.splitlines())} lignes fournies dans stdin. Ce n'est pas un bug du "
                       "programme. Si tes réponses dépendent de ce qu'il affiche, utilise interactive_start "
                       "puis interactive_send (une réponse à la fois). Sinon, relance TOI-MÊME bash avec "
                       "assez de lignes pour aller jusqu'au bout.]")
    # Le code de sortie est crucial : c'est grâce à lui que le modèle sait si
    # ses tests passent ou non.
    return _truncate(f"{output}\n[code de sortie : {r.returncode}]")


# --- Ce que le modèle voit --------------------------------------------------
# La qualité des descriptions change énormément le comportement du modèle :
# c'est du "prompt engineering" appliqué aux outils.
TOOL_SCHEMAS = [
    {
        "name": "read_file",
        "description": (
            "Lit un fichier texte du projet et renvoie son contenu avec les numéros de ligne. "
            "Le chemin est relatif à la racine du projet. Un gros fichier est renvoyé par morceaux : "
            "la fin du résultat indique le start_line pour lire la suite."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Chemin relatif à la racine du projet"},
                "start_line": {"type": "integer", "description": "Optionnel : 1re ligne à lire (pour la suite d'un gros fichier)"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "list_dir",
        "description": "Liste les fichiers et sous-dossiers d'un dossier du projet (les dossiers finissent par /).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Chemin relatif, '.' pour la racine"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "grep",
        "description": (
            "Cherche une expression régulière (syntaxe Python) dans les fichiers du projet et renvoie "
            "les lignes trouvées sous la forme fichier:ligne: texte. Utilise-le pour trouver où une "
            "fonction ou un mot est défini/utilisé, au lieu de lire tous les fichiers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": r"Regex, ex: def run_tool ou TODO|FIXME"},
                "path": {"type": "string", "description": "Dossier ou fichier où chercher, '.' pour tout le projet"},
            },
            "required": ["pattern", "path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "edit_file",
        "description": (
            "Modifie un fichier en remplaçant old_string par new_string. old_string doit être recopié "
            "EXACTEMENT depuis le fichier (indentation comprise, sans les numéros de ligne de read_file) "
            "et n'apparaître qu'une seule fois. old_string doit être COURT : seulement la ou les lignes "
            f"à changer (au plus {MAX_EDIT_LINES - 1}), jamais tout le fichier. Pour AJOUTER une ligne, "
            "prends comme old_string la ligne voisine, et mets dans new_string cette ligne + la nouvelle. "
            "Lis toujours le fichier avant de le modifier. "
            "Pour créer un nouveau fichier : old_string vide et tout le contenu dans new_string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Chemin relatif du fichier"},
                "old_string": {"type": "string", "description": "Texte exact à remplacer ('' pour créer un fichier)"},
                "new_string": {"type": "string", "description": "Nouveau texte"},
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "interactive_start",
        "description": (
            "Lance un programme interactif (qui lit le clavier) et renvoie ce qu'il affiche jusqu'à sa "
            "première question, avec un numéro de session. Ensuite, réponds-lui ligne par ligne avec "
            "interactive_send, en lisant chaque réponse avant de choisir la suivante. À utiliser quand "
            "les saisies dépendent de ce que le programme répond ; sinon, bash avec stdin suffit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "La commande qui lance le programme"}},
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "interactive_send",
        "description": (
            "Envoie UNE ligne (comme si on la tapait puis appuyait sur Entrée) au programme d'une session "
            "ouverte par interactive_start, et renvoie ce qu'il affiche en réponse. Indique aussi si le "
            "programme s'est terminé."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Le numéro donné par interactive_start"},
                "text": {"type": "string", "description": "La ligne à taper (sans le retour à la ligne)"},
            },
            "required": ["session_id", "text"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "bash",
        "description": (
            "Exécute une commande shell dans le dossier du projet et renvoie sa sortie et son code de "
            f"sortie (0 = succès). Limite : {BASH_TIMEOUT} s. Utile pour lancer les tests du projet et "
            "vérifier ton travail après une modification. Il n'y a pas de clavier. Programme qui pose "
            "des questions : si tu connais toutes les réponses d'avance, donne-les dans stdin ; si tes "
            "réponses dépendent de ce que le programme affiche, N'UTILISE PAS bash : utilise "
            "interactive_start puis interactive_send."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                # Pas d'exemple concret dans ces deux descriptions : les petits modèles RECOPIENT
                # les exemples. Vu avec stdin (ils envoyaient "50\n75\n62\n" à n'importe quel
                # programme) puis aux évals (« ex: uv run pytest » → qwen3 lançait pytest même
                # quand on lui donnait une autre commande, puis tentait de l'installer).
                "command": {"type": "string", "description": "La commande shell à exécuter"},
                "stdin": {
                    "type": "string",
                    "description": "Optionnel : texte envoyé au programme comme s'il était tapé au clavier, "
                                   "une ligne par saisie. Prévois une ligne pour CHAQUE question que posera "
                                   "le programme, sinon il s'arrête faute de saisie.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

# --- Ce que le harness exécute ----------------------------------------------
TOOL_FUNCTIONS = {
    "read_file": read_file,
    "list_dir": list_dir,
    "grep": grep,
    "edit_file": edit_file,
    "bash": bash,
    "interactive_start": interactive_start,
    "interactive_send": interactive_send,
}


SCHEMAS_BY_NAME = {s["name"]: s["input_schema"] for s in TOOL_SCHEMAS}


def validate_input(name: str, tool_input) -> None:
    """Vérifie les arguments AVANT d'exécuter : ne jamais faire confiance à ce que
    renvoie le modèle (un petit modèle se trompe, et en streaming un JSON peut être tronqué)."""
    schema = SCHEMAS_BY_NAME[name]
    if not isinstance(tool_input, dict):
        raise ToolError(f"Arguments invalides pour {name} : un objet JSON est attendu.")
    unknown = set(tool_input) - set(schema["properties"])
    if unknown:
        raise ToolError(f"Arguments inconnus pour {name} : {', '.join(sorted(unknown))}")
    for key in schema["required"]:
        if not isinstance(tool_input.get(key), str):
            raise ToolError(f"Argument manquant ou invalide pour {name} : {key} (texte attendu)")
    for key, value in tool_input.items():  # les arguments optionnels aussi doivent avoir le bon type
        if schema["properties"][key].get("type") == "integer":
            # Un petit modèle écrit parfois "151" au lieu de 151 : on accepte les deux.
            if isinstance(value, bool) or not (isinstance(value, int) or str(value).strip().isdigit()):
                raise ToolError(f"Argument invalide pour {name} : {key} (nombre entier attendu)")
        elif not isinstance(value, str):
            raise ToolError(f"Argument invalide pour {name} : {key} (texte attendu)")


def precheck(name: str, tool_input) -> str | None:
    """Vérifie un appel AVANT de demander la permission. Renvoie l'erreur, ou None si l'appel peut marcher."""
    if name not in TOOL_FUNCTIONS:
        return f"Outil inconnu : {name}"
    try:
        validate_input(name, tool_input)
        if name == "edit_file":
            plan_edit(**tool_input)
    except ToolError as e:
        return str(e)
    return None


def run_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Exécute un outil. Renvoie (résultat, is_error)."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Outil inconnu : {name}", True
    try:
        validate_input(name, tool_input)
        return fn(**tool_input), False
    except ToolError as e:
        return str(e), True
    except Exception as e:  # un bug dans un outil ne doit pas tuer la session
        return f"{type(e).__name__}: {e}", True

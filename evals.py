"""ÉTAPE 10 : les évals — mesurer minicode au lieu de deviner.

Quand on change un prompt, un garde-fou ou un modèle, comment savoir si c'est MIEUX ?
Essayer « à la main » sur un exemple ne suffit pas : un modèle est aléatoire, et une
correction qui aide sur un cas peut en casser un autre. Une éval, c'est :

  1. une liste de tâches FIXES (ici 10, petites) : des fichiers de départ + une demande ;
  2. pour chacune, une VÉRIFICATION automatique (le programme corrigé tourne, la réponse
     contient la bonne valeur…) : pas d'avis humain, pas de « ça a l'air bien » ;
  3. des mesures : réussite, nombre d'appels au modèle, tokens consommés, durée.

Chaque tâche tourne dans un dossier temporaire NEUF, dans un sous-processus à part :
minicode y démarre comme si tu le lançais dans ce dossier (WORKSPACE = ce dossier, pas
d'historique, MINICODE_YOLO=1 pour ne rien demander). La vérification se fait APRÈS,
depuis ce programme-ci, en regardant les fichiers et la réponse.

Lancer :
  uv run evals.py                    # les 10 tâches
  uv run evals.py -k bug -k jeu      # seulement celles dont le nom contient « bug » ou « jeu »
  uv run evals.py --repeat 3         # chaque tâche 3 fois (le modèle est aléatoire)
  MINICODE_MODEL=qwen3:4b uv run evals.py    # comparer un autre modèle

Les résultats sont gardés dans .minicode/evals/ : le score précédent est affiché à côté.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_TIMEOUT = 600  # secondes par tâche (qwen3 local peut être lent)
# Les évals tournent en MINICODE_YOLO=1 (tout est accepté)… mais une règle deny l'emporte
# même sur YOLO (étape 5). Vu au 1er run : pytest absent, qwen3 a tenté `sudo pacman -S`.
EVAL_DENY = ["bash(*sudo *)", "bash(sudo*)", "bash(*pip install*)", "bash(*pacman *)",
             "bash(*apt *)", "bash(*npm install*)", "bash(*curl *)", "bash(*wget *)"]


def run(workdir, *args, stdin=""):
    """Lance une commande dans le dossier de la tâche. Renvoie (code de sortie, sortie)."""
    try:
        r = subprocess.run(args, cwd=workdir, input=stdin, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return -1, "[bloqué : plus de 20 s]"
    return r.returncode, r.stdout + r.stderr


def python(workdir, *args, stdin=""):
    return run(workdir, sys.executable, *args, stdin=stdin)


@dataclass
class Task:
    name: str
    step: str                 # l'étape du projet que la tâche met surtout à l'épreuve
    prompts: list             # une demande par tour (plusieurs = conversation)
    files: dict               # fichiers de départ : chemin -> contenu
    check: object             # check(dossier, réponses) -> (réussi, explication)
    solution: dict = field(default_factory=dict)  # une bonne solution, pour tester la vérification
    good_answer: str = ""


def read(workdir, path):
    p = Path(workdir) / path
    return p.read_text() if p.exists() else ""


# --- les 10 tâches -------------------------------------------------------------------


def check_contains(expected):
    def check(workdir, answers):
        last = answers[-1] if answers else ""
        return expected in last, f"« {expected} » {'trouvé' if expected in last else 'absent'} dans la réponse"
    return check


def check_bug(workdir, answers):
    if read(workdir, "test_calc.py") != TEST_CALC:
        return False, "le test a été modifié (interdit : il faut corriger le programme)"
    code, out = python(workdir, "test_calc.py")
    return code == 0 and "OK" in out, f"python3 test_calc.py → code {code}"


def check_hello(workdir, answers):
    code, out = python(workdir, "bonjour.py")
    return code == 0 and out.strip() == "Bonjour le monde", f"sortie : {out.strip()[:60]!r}"


def check_est_pair(workdir, answers):
    code, out = python(workdir, "-c", "from utils import est_pair, double\n"
                                      "assert est_pair(4) is True and est_pair(7) is False\n"
                                      "assert double(3) == 6\nprint('OK')")
    return code == 0, "est_pair et double fonctionnent" if code == 0 else out.strip().splitlines()[-1][:80]


def check_rename(workdir, answers):
    left = [f for f in ("panier.py", "main.py") if "calc_total" in read(workdir, f)]
    if left:
        return False, f"calc_total encore présent dans {', '.join(left)}"
    code, out = python(workdir, "main.py")
    return code == 0 and out.strip() == "6", f"python3 main.py → {out.strip()[:60]!r}"


def check_game(workdir, answers):
    if "random.randint" not in read(workdir, "jeu.py"):
        return False, "le hasard a été supprimé (triche)"
    code, out = python(workdir, "jeu.py", stdin="0\n" * 10)  # 0 n'est jamais le nombre secret
    if "Perdu" not in out:
        return False, "avec 10 mauvaises réponses, on ne perd toujours pas"
    return "restants : 4" in out, "le compteur diminue et la partie se termine"


def check_ends_with_fin(workdir, answers):
    last = (answers[-1] if answers else "").strip()
    return last.endswith("FIN"), f"fin de la réponse : {last[-30:]!r}"


TEST_CALC = "from calc import add, mul\n\nassert add(2, 3) == 5\nassert mul(2, 3) == 6\nprint('OK')\n"
GAME = '''import random

secret = random.randint(1, 100)
while True:
    essais_restants = 5
    reponse = input("Ton nombre : ")
    if int(reponse) == secret:
        print("Gagné !")
        break
    essais_restants -= 1
    print(f"Raté. Essais restants : {essais_restants}")
    if essais_restants == 0:
        print("Perdu !")
        break
'''

TASKS = [
    Task("lire_valeur", "2. lire un fichier",
         ["Sur quel port écoute le serveur, d'après la configuration du projet ?"],
         {"config.py": "HOST = 'localhost'\nPORT = 8731\nDEBUG = False\n"},
         check_contains("8731"), good_answer="Le port 8731."),
    Task("trouver_fichier", "4. grep",
         ["Dans quel fichier est définie la fonction calculer_tva ?"],
         {"factures/__init__.py": "", "factures/tva.py": "def calculer_tva(montant):\n    return montant * 0.2\n",
          "clients/clients.py": "def nom_client(c):\n    return c['nom']\n",
          "utils/format.py": "def euros(x):\n    return f'{x:.2f} €'\n"},
         check_contains("tva.py"), good_answer="Dans factures/tva.py."),
    Task("corriger_bug", "4. edit_file + bash",
         ["python3 test_calc.py échoue. Trouve le bug et corrige-le."],
         {"calc.py": "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n", "test_calc.py": TEST_CALC},
         check_bug, solution={"calc.py": "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"}),
    Task("creer_script", "4. créer un fichier",
         ["Crée un fichier bonjour.py qui affiche exactement : Bonjour le monde"],
         {}, check_hello, solution={"bonjour.py": "print('Bonjour le monde')\n"}),
    Task("ajouter_fonction", "4. modifier sans casser",
         ["Ajoute dans utils.py une fonction est_pair(n) qui renvoie True si n est pair, sinon False. "
          "Ne modifie pas la fonction double."],
         {"utils.py": "def double(n):\n    return n * 2\n"},
         check_est_pair, solution={"utils.py": "def double(n):\n    return n * 2\n\n\ndef est_pair(n):\n    return n % 2 == 0\n"}),
    Task("renommer", "4. modifier plusieurs fichiers",
         ["Renomme la fonction calc_total en total_panier partout dans le projet."],
         {"panier.py": "def calc_total(prix):\n    return sum(prix)\n",
          "main.py": "from panier import calc_total\n\nprint(calc_total([1, 2, 3]))\n"},
         check_rename, solution={"panier.py": "def total_panier(prix):\n    return sum(prix)\n",
                                 "main.py": "from panier import total_panier\n\nprint(total_panier([1, 2, 3]))\n"}),
    Task("compteur_jeu", "4. bug de logique",
         ["Dans jeu.py, le nombre d'essais restants ne diminue jamais : on ne peut pas perdre. Corrige le bug."],
         {"jeu.py": GAME}, check_game,
         solution={"jeu.py": GAME.replace("while True:\n    essais_restants = 5\n",
                                          "essais_restants = 5\nwhile True:\n")}),
    Task("memoire", "1. historique",
         ["Je m'appelle Camille et mon projet s'appelle Orion. Réponds juste OK.",
          "Comment s'appelle mon projet ?"],
         {}, check_contains("Orion"), good_answer="Ton projet s'appelle Orion."),
    Task("agents_md", "6. AGENTS.md",
         ["Que fait app.py ?"],
         {"AGENTS.md": "# Règles\n- Termine TOUJOURS ta réponse par le mot FIN, seul sur la dernière ligne.\n",
          "app.py": "print('salut')\n"},
         check_ends_with_fin, good_answer="app.py affiche « salut ».\nFIN"),
    Task("gros_fichier", "8. contexte",
         ["Quelle est la valeur de VERSION dans data.py ?"],
         {"data.py": "".join(f"VALEUR_{i} = {i * 7}\n" for i in range(600)) + 'VERSION = "4.2.7"\n'},
         check_contains("4.2.7"), good_answer="VERSION vaut 4.2.7."),
]


# --- exécution d'UNE tâche (dans un sous-processus) ---------------------------------------


def worker(name, out_path):
    """Lancé avec cwd = le dossier de la tâche : minicode le prend pour son projet."""
    import minicode  # importé ICI : WORKSPACE = le dossier courant = celui de la tâche

    task = next(t for t in TASKS if t.name == name)
    ui = minicode.get_ui()
    client = minicode.make_client()
    minicode.load_project_context()
    trace = minicode.Trace(minicode.WORKSPACE / ".minicode" / "traces", provider=minicode.PROVIDER,
                           model=minicode.MODEL, system=minicode.SYSTEM_PROMPT, tools=minicode.MAIN_TOOLS)
    messages, answers, error = [], [], None
    start = time.time()
    for prompt in task.prompts:
        ui.console.print(f"> {prompt}")
        try:
            answers.append(minicode.run_turn(client, messages, prompt, trace=trace, ui=ui))
        except Exception as e:  # une erreur = tâche ratée, mais on garde les mesures
            error = f"{type(e).__name__}: {e}"
            break
    Path(out_path).write_text(json.dumps({
        "answers": answers, "error": error, "calls": ui.calls, "seconds": round(time.time() - start, 1),
        "tokens_in": ui.tokens_in, "tokens_out": ui.tokens_out, "trace": str(trace.path),
    }, ensure_ascii=False))


def setup(task, workdir):
    for path, content in task.files.items():
        target = Path(workdir) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


def run_task(task, root):
    """Prépare le dossier, lance minicode dessus, puis vérifie. Renvoie un dict de résultats."""
    workdir = Path(tempfile.mkdtemp(prefix=f"{task.name}-", dir=root))
    setup(task, workdir)
    rules = workdir / ".minicode" / "permissions.json"
    rules.parent.mkdir(exist_ok=True)
    rules.write_text(json.dumps({"allow": [], "deny": EVAL_DENY}, indent=1))
    result_file, log_file = workdir / ".resultat.json", workdir / ".minicode-sortie.txt"
    env = {**os.environ, "MINICODE_YOLO": "1", "MINICODE_TRACE": "1"}
    with log_file.open("w") as log:
        try:
            subprocess.run([sys.executable, str(HERE / "evals.py"), "--worker", task.name, str(result_file)],
                           cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=TASK_TIMEOUT)
        except subprocess.TimeoutExpired:
            pass
    if result_file.exists():
        result = json.loads(result_file.read_text())
    else:
        result = {"answers": [], "error": f"pas de résultat (délai de {TASK_TIMEOUT} s dépassé ou plantage)",
                  "calls": 0, "seconds": TASK_TIMEOUT, "tokens_in": 0, "tokens_out": 0}
    ok, why = task.check(workdir, result["answers"])
    if result["error"]:
        ok, why = False, result["error"]
    return {**result, "task": task.name, "ok": ok, "why": why, "dir": str(workdir), "log": str(log_file)}


# --- le programme principal ---------------------------------------------------------------


def previous_score(results_dir):
    files = sorted(results_dir.glob("*.json"))
    if not files:
        return None
    data = json.loads(files[-1].read_text())
    return f"{data['passed']}/{data['total']} ({data['model']}, {data['date']})"


def main():
    parser = argparse.ArgumentParser(description="Évals de minicode (étape 10)")
    parser.add_argument("-k", action="append", default=[], help="filtre sur le nom des tâches")
    parser.add_argument("--repeat", type=int, default=1, help="nombre d'essais par tâche")
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(*args.worker)

    from rich.console import Console
    from rich.table import Table

    import minicode
    console = Console(highlight=False)
    tasks = [t for t in TASKS if not args.k or any(k in t.name for k in args.k)]
    root = Path(tempfile.mkdtemp(prefix="minicode-evals-"))
    console.print(f"[bold]{len(tasks)} tâche(s) × {args.repeat}[/] · {minicode.PROVIDER} / {minicode.MODEL} · "
                  f"dossiers : {root}\n")

    rows = []
    for task in tasks:
        for attempt in range(args.repeat):
            if console.is_terminal:  # animation seulement dans un vrai terminal (pas dans un fichier)
                with console.status(f"{task.name} (essai {attempt + 1}/{args.repeat})…"):
                    r = run_task(task, root)
            else:
                r = run_task(task, root)
            mark = "[green]✓[/]" if r["ok"] else "[red]✗[/]"
            console.print(f"{mark} {task.name:<17} {r['calls']:>3} appels · {r['tokens_in'] + r['tokens_out']:>7} tokens · "
                          f"{r['seconds']:>5.0f} s · [dim]{r['why']}[/]")
            rows.append({**r, "step": task.step})

    table = Table(title="Résultats", title_justify="left")
    for column in ("tâche", "étape", "réussite", "appels (moy.)", "tokens (moy.)", "durée (moy.)"):
        table.add_column(column, justify="left" if column in ("tâche", "étape") else "right")
    for task in tasks:
        runs = [r for r in rows if r["task"] == task.name]
        passed = sum(r["ok"] for r in runs)
        style = "green" if passed == len(runs) else "red" if passed == 0 else "yellow"
        table.add_row(task.name, task.step, f"[{style}]{passed}/{len(runs)}[/]",
                      f"{sum(r['calls'] for r in runs) / len(runs):.1f}",
                      f"{sum(r['tokens_in'] + r['tokens_out'] for r in runs) / len(runs):.0f}",
                      f"{sum(r['seconds'] for r in runs) / len(runs):.0f} s")
    console.print()
    console.print(table)

    passed, total = sum(r["ok"] for r in rows), len(rows)
    results_dir = Path.cwd() / ".minicode" / "evals"
    results_dir.mkdir(parents=True, exist_ok=True)
    before = previous_score(results_dir)
    console.print(f"\n[bold]Score : {passed}/{total}[/] ({100 * passed // max(1, total)} %) · "
                  f"{sum(r['calls'] for r in rows)} appels · "
                  f"{sum(r['tokens_in'] + r['tokens_out'] for r in rows)} tokens · "
                  f"{sum(r['seconds'] for r in rows) / 60:.1f} min")
    if before:
        console.print(f"[dim]précédent : {before}[/]")
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = results_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"date": date, "model": f"{minicode.PROVIDER}/{minicode.MODEL}", "passed": passed,
                               "total": total, "runs": rows}, ensure_ascii=False, indent=1))
    console.print(f"[dim]détails : {out}  (pour chaque tâche : dossier, sortie de minicode, journal)[/]")


if __name__ == "__main__":
    main()

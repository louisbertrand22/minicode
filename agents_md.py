"""ÉTAPE 6 : le contexte projet (AGENTS.md).

Le modèle ne sait rien de ton projet au début d'une conversation : ni la commande
des tests, ni les conventions, ni les fichiers à ne pas toucher. Il pourrait le
redécouvrir en explorant à chaque fois… ou on le lui DIT : c'est le rôle d'AGENTS.md,
un simple fichier Markdown écrit pour les agents (comme un README écrit pour les humains).
C'est un format partagé par plusieurs outils ; Claude Code lit le même genre de
fichier sous le nom CLAUDE.md.

Le harness le lit et l'ajoute au PROMPT SYSTÈME. Pourquoi là, et pas dans un message ?
  - il est envoyé à chaque appel, AVANT la conversation : le modèle le traite comme
    une consigne, pas comme une demande parmi d'autres ;
  - l'étape 8 ne le résume ni ne l'efface jamais (elle ne touche qu'à `messages`) ;
  - il ne change pas d'un appel à l'autre : le serveur le garde en cache.

On cherche AGENTS.md dans le dossier du projet ET dans ses dossiers parents (jusqu'au
dossier personnel) : un AGENTS.md général dans ~/projets s'applique à tous les
projets en dessous. Le plus général vient d'abord, le plus précis en dernier
(en cas de contradiction, le modèle suit plutôt la dernière consigne lue).
"""

from pathlib import Path

FILENAME = "AGENTS.md"


def find_files(workspace: Path, stop: Path | None = None) -> list[Path]:
    """Les AGENTS.md qui s'appliquent à `workspace`, du plus général au plus précis."""
    stop = (stop or Path.home()).resolve()
    found = []
    for directory in [workspace, *workspace.parents]:
        candidate = directory / FILENAME
        if candidate.is_file():
            found.append(candidate)
        if directory == stop:  # on ne remonte pas au-dessus du dossier personnel
            break
    return found[::-1]


def load(workspace: Path, max_chars: int, stop: Path | None = None) -> tuple[str, list[Path]]:
    """Renvoie (texte à ajouter au prompt système, fichiers lus).

    `max_chars` : le contexte projet est envoyé à CHAQUE appel. Un AGENTS.md énorme
    mangerait la fenêtre (étape 8) : au-delà, on coupe et on le signale.
    """
    files = find_files(workspace, stop)
    if not files:
        return "", []
    sections = []
    for path in files:
        text = path.read_text(errors="replace").strip()
        if text:
            shown = path.relative_to(workspace) if path.is_relative_to(workspace) else path
            sections.append(f"## {shown}\n\n{text}")
    body = "\n\n".join(sections)
    if len(body) > max_chars:
        body = body[:max_chars] + f"\n\n[… AGENTS.md tronqué par minicode : {len(body)} caractères, limite {max_chars}]"
    if not body:
        return "", files
    return (
        "\n\n# Instructions du projet (AGENTS.md)\n\n"
        "Ces instructions viennent de l'utilisateur. Suis-les : elles passent avant tes habitudes "
        "(mais pas avant ses demandes explicites dans la conversation).\n\n" + body
    ), files


def fingerprint(workspace: Path, stop: Path | None = None):
    """Une « empreinte » (fichiers + dates de modification) pour savoir s'il faut recharger."""
    return tuple((str(p), p.stat().st_mtime_ns) for p in find_files(workspace, stop))


INIT_REQUEST = f"""Crée le fichier {FILENAME} à la racine du projet, pour les futurs agents qui travailleront ici.
Étapes, dans l'ordre :
1. list_dir, puis read_file sur les fichiers de code et de configuration (README, pyproject.toml,
   package.json…) : au moins les principaux. Ne te contente PAS de la liste des fichiers.
2. Seulement ensuite, écris le fichier avec edit_file. S'il existe déjà, améliore-le au lieu de le remplacer.

Règle stricte : n'écris QUE ce que tu as lu dans un fichier ou vérifié avec bash. Si tu ne sais
pas (ex : aucune commande de test trouvée), écris « aucune trouvée » au lieu d'inventer.
Les dossiers .minicode et .git appartiennent aux outils, pas au projet : ne les décris pas.

Contenu attendu, court (moins de 40 lignes), en Markdown :
- en une phrase : ce que fait le projet ;
- les commandes utiles (lancer, tester, installer) — vérifie-les avec bash si possible ;
- l'organisation : les fichiers ou dossiers importants et leur rôle ;
- les conventions de code observées (langue, style, nommage) ;
- les pièges ou règles à respecter s'il y en a.
Pas de généralités valables pour n'importe quel projet."""

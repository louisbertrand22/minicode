# minicode

Un mini agent de code (façon Claude Code) écrit **à la main**, sans framework d'agent,
pour comprendre ce qu'est un *harness* : tout le code autour du modèle (boucle, outils,
contexte, permissions, mémoire).

## Lancer (gratuit, avec Ollama)

Le modèle tourne sur ta machine : pas de clé, pas de frais.

```bash
sudo pacman -S ollama-cuda                 # Arch + carte NVIDIA (sinon : https://ollama.com/download)
OLLAMA_CONTEXT_LENGTH=8192 ollama serve    # dans un terminal à part, laisse-le tourner
ollama pull qwen3:8b                       # ~5 Go, une seule fois
uv run minicode.py                         # dans un autre terminal
```

Sans sudo : l'archive officielle s'installe dans ton dossier perso.

```bash
mkdir -p ~/.local/ollama
curl -fL -C - -o /tmp/ollama.tar.zst https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst
tar --zstd -xf /tmp/ollama.tar.zst -C ~/.local/ollama
export PATH="$HOME/.local/ollama/bin:$PATH"   # à ajouter dans ~/.zshrc
```

⚠️ **`OLLAMA_CONTEXT_LENGTH` est important.** Avec moins de 24 Go de VRAM, Ollama limite
le contexte à 4k tokens par défaut, et **coupe l'historique en silence** quand ça déborde :
le modèle "oublie" alors le début de la conversation ou les résultats d'outils.

Plus de contexte = plus de VRAM. Mesuré avec `qwen3:8b` sur une RTX 5060 (8 Go), même
test de 3 questions :

| Contexte | Mémoire | Réparti CPU/GPU | Durée |
|---|---|---|---|
| 8192  | 6,6 Go | 11 % / 89 % | 42 s |
| 16384 | 7,8 Go | 24 % / 76 % | 77 s |

Vérifie la répartition avec `ollama ps`. 8k suffit pour les étapes 1–3 ; il deviendra
trop petit quand l'agent lira de gros fichiers : c'est exactement le problème que
l'étape 8 (gestion du contexte) apprend à résoudre.

Autres modèles qui savent utiliser des outils : `qwen3:4b` (plus léger),
`qwen3-coder:30b` (bien meilleur en code, mais ~19 Go : lent sur 8 Go de VRAM).
→ `MINICODE_MODEL=qwen3:4b uv run minicode.py`

## Lancer (payant, avec Claude)

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # https://console.anthropic.com
MINICODE_PROVIDER=anthropic uv run minicode.py
```

Variables optionnelles : `MINICODE_MODEL` (défaut `claude-opus-5-5`, `claude-haiku-4-5`
pour dépenser moins), `MINICODE_EFFORT` (`low` … `max`, défaut `medium`).

Pour explorer un autre projet que minicode lui-même, lance depuis son dossier :
`uv run --project ~/Documents/GITHUB-PROJECTS/minicode ~/Documents/GITHUB-PROJECTS/minicode/minicode.py`

Tests (sans appel API) : `uv run pytest`

## Les 3 idées à retenir (étapes 1–3)

1. **Le modèle n'a pas de mémoire.** `messages` (une simple liste) *est* la conversation ;
   on la renvoie en entier à chaque appel. → `main()` dans `minicode.py`
2. **Le modèle n'exécute rien.** Il répond `stop_reason="tool_use"` avec des blocs
   `tool_use` ; le harness exécute, puis renvoie des `tool_result` (même `id`) dans un
   message `user`. → `tools.py`
3. **Un « agent », c'est une boucle `while`.** On rappelle le modèle tant qu'il demande
   des outils. → `run_turn()` dans `minicode.py`

## Feuille de route

- [x] 1. REPL de chat (historique complet renvoyé à chaque appel)
- [x] 2. Outils `read_file` / `list_dir` (schéma + exécution + erreurs `is_error`)
- [x] 3. Boucle d'agent (plusieurs appels d'outils, en parallèle, limite `MAX_STEPS`)
- [x] 4. Outils qui agissent : `grep`, `edit_file` (remplacement exact), `bash` (+ relance si réponse vide)
- [x] 5. Permissions : o/t/N avant `bash` et `edit_file`, règles allow/deny dans `.minicode/permissions.json`
- [x] 6. Contexte projet : les `AGENTS.md` du projet (et des dossiers parents) dans le prompt système, `/init`, `/agents`
- [x] 7. Streaming (réflexion 💭 + réponse en direct, tokens par appel) + journal JSONL + `show_trace.py`
- [x] 8. Gestion du contexte : compter les tokens, effacer les vieux résultats d'outils, résumer les vieux tours (`/context`, `/compact`)
- [x] 9. Sous-agents : un outil `task` qui lance une boucle avec son propre contexte (lecture seule, seul le rapport revient)
- [ ] 10. Évals : 10 petites tâches, score de réussite / nombre de tours / coût

## Étapes 4–5 : l'agent agit, le harness contrôle

- `grep` évite de lire tous les fichiers ; `edit_file` remplace un texte **exact** (le modèle
  n'écrit que ce qui change, et une erreur de recopie est détectée au lieu d'abîmer le fichier) ;
  `bash` renvoie la sortie **et le code de sortie**, ce qui permet à l'agent de vérifier son travail.
- Avant `edit_file` et `bash`, minicode montre ce qui va se passer et demande `o/N`.
  Un refus n'exécute rien, mais renvoie quand même un `tool_result` au modèle (sinon l'API
  rejette l'historique) avec la consigne de ne pas réessayer.
- `MINICODE_YOLO=1` accepte tout sans demander. Pratique pour les tests, dangereux ailleurs.

### Les règles (`.minicode/permissions.json`)

Réponds `t` (toujours) à une question et minicode enregistre une règle pour ne plus la poser.
Tu peux aussi écrire le fichier toi-même :

```json
{
  "allow": ["bash(uv run pytest*)", "bash(git status)", "edit_file(src/*)"],
  "deny":  ["bash(rm -rf*)", "bash(git push*)"]
}
```

`outil(motif)` : le motif (glob, `*` = n'importe quoi) est comparé à la commande pour `bash`,
au chemin pour `edit_file`. Ordre de décision (voir `permissions.py`) :

1. **deny** l'emporte sur tout, même sur `MINICODE_YOLO=1` ;
2. une **commande composée** (`;` `&&` `|` `>` `$(`…) est **toujours demandée** :
   sans ça, `bash(uv run pytest*)` autoriserait `uv run pytest; rm -rf ~` ;
3. **allow** ;
4. sinon, on demande.

L'agent ne peut pas écrire dans `.minicode/` avec `edit_file`, sinon il pourrait s'autoriser
lui-même. **Limite honnête** : via `bash` (si tu l'autorises), un programme peut toujours écrire
n'importe où. Des règles de texte ne suffisent pas à sécuriser un agent : les vrais outils
ajoutent un **bac à sable** (conteneur, VM) qui limite ce que les commandes peuvent toucher.

**Défi testé avec qwen3:8b** : sur un `calc.py` où `add` soustrait, la demande
« lance les tests ; si un test échoue, corrige le bug et relance » donne
`bash → grep → read_file → edit_file → bash` et des tests verts, sans aide.

**Exercices :**
- Crée un petit projet avec un bug et refais le défi. Refuse l'`edit_file` : que dit le modèle ?
- Demande « supprime tous les fichiers .txt ». Lis bien la commande avant de répondre `o` !
- Ajoute `"deny": ["edit_file(*)"]` puis demande une correction : comment réagit le modèle ?
- Trouve une commande qui passe la règle `bash(python3*)` et fait autre chose que lancer
  un script (indice : `python3 -c`). Que faudrait-il changer pour la bloquer ?
- Mets `MAX_NUDGES = 0` et répète une demande de correction plusieurs fois : combien de
  réponses vides obtiens-tu ?

## L'interface (façon Claude Code)

Tout l'affichage est dans `ui.py` (bibliothèques `rich` et `prompt_toolkit`) ; `minicode.py`
ne contient que le harness et appelle `ui.tool_call(...)`, `ui.permission(...)`… On pourrait
remplacer `ui.py` par une page web sans toucher à la boucle d'agent.

```
⏺ Search("Félicitations!" dans .)
  ⎿  1 résultats
⏺ Update(jeu.py)
╭─ Modifier jeu.py ─────────────────────────────────────────╮
│   18       else:                                          │
│   19 -         print("Félicitations!")                    │
│   19 +         print("Bravo, tu as trouvé !")             │
╰───────────────────────────────────────────────────────────╯
 Voulez-vous continuer ?
   >  1. Oui
      2. Oui, et ne plus demander pour edit_file(jeu.py)
      3. Non, et dire à minicode quoi faire autrement
⏺ Le message a été remplacé…
✻ Terminé en 41 s · 7 appel(s) au modèle · contexte 3761 tokens (3689 en cache)
```

- Saisie encadrée avec historique (↑/↓, gardé dans `.minicode/history`) et complétion des
  commandes : `/help`, `/clear` (nouvelle conversation : on vide simplement `messages`),
  `/trace`, `/exit`. La barre du bas affiche le modèle et la taille du contexte.
- Pendant un appel : `✻ Réflexion… (12 s · ↓ ~300 tokens)` avec un aperçu de la réflexion du
  modèle (`MINICODE_THINKING=0` pour le cacher), puis la réponse en Markdown.
- Permission : flèches + Entrée. « 3. Non » demande quoi faire à la place, et cette consigne
  est renvoyée au modèle dans le `tool_result`.
- **Ctrl-C** interrompt le modèle : la demande en cours est annulée (l'historique est remis
  dans son état d'avant, sinon un `tool_use` sans réponse ferait échouer l'appel suivant).

Deux protections ajoutées en testant l'interface avec qwen3:8b :
- **vérifier avant de demander** (`tools.precheck`) : un `edit_file` qui échouera de toute
  façon (texte introuvable, fichier existant…) renvoie l'erreur au modèle sans te déranger ;
- **détecter les boucles** : qwen3 a refait 6 fois exactement le même appel raté. Le harness
  signale maintenant « tu as déjà fait exactement cet appel et il a échoué ».

## Étape 6 : le contexte projet (`AGENTS.md`)

Au début d'une conversation, le modèle ne sait rien de ton projet : ni comment lancer les
tests, ni tes conventions, ni ce qu'il ne doit pas toucher. `AGENTS.md` est un simple fichier
Markdown **écrit pour les agents** (comme un README pour les humains). Le même format est lu
par plusieurs outils ; Claude Code utilise le même principe avec `CLAUDE.md`.

```markdown
# AGENTS.md
- Tests : `uv run pytest` (à lancer après chaque modification)
- Code et messages en français.
- Ne touche jamais à `uv.lock`.
```

minicode cherche `AGENTS.md` dans le dossier du projet **et dans ses parents**, jusqu'à ton
dossier personnel. Un `~/projets/AGENTS.md` s'applique donc à tous tes projets. Il les ajoute
à la fin du **prompt système**, du plus général au plus précis (→ `agents_md.py`).

Pourquoi dans le prompt système, et pas comme premier message ?
- le modèle le traite comme une **consigne permanente**, pas comme une demande parmi d'autres ;
- l'étape 8 ne le résume ni ne l'efface jamais : elle ne touche qu'à `messages` ;
- il est identique d'un appel à l'autre, donc le serveur le garde **en cache**.

La contrepartie : il est envoyé à **chaque** appel. minicode le limite donc à ~10 % de la
fenêtre (≈ 2 400 caractères avec 8k tokens) et prévient s'il le coupe. Un bon AGENTS.md est court.

- `/init` : l'agent explore le projet et écrit (ou améliore) `AGENTS.md`. Il n'y a rien de
  magique : c'est juste une demande toute prête (`agents_md.INIT_REQUEST`).
- `/agents` : les fichiers chargés, et combien de caractères ils ajoutent au prompt.
- Avant chaque demande, minicode vérifie si un `AGENTS.md` a changé (date de modification).
  Si oui, il reconstruit le prompt système et le note dans le journal (`"type": "system"`).
  Sinon il ne touche à rien, pour garder le cache.

**Exercices :**
- Écris une règle visible (« termine chaque réponse par 🦊 ») et regarde si qwen3 la suit.
- `uv run show_trace.py --call 1` : retrouve ton AGENTS.md dans le `system` envoyé.
- Mets une règle dans `~/AGENTS.md` et une règle contraire dans le projet : laquelle gagne ?

## Étape 7 : voir ce qui se passe sous le capot

**Streaming.** La réponse arrive en petits morceaux (événements `thinking`, `text`) affichés
dès qu'ils arrivent : la réflexion du modèle en gris après 💭, la réponse en normal. À la fin,
`stream.get_final_message()` reconstitue le message complet, donc le reste de la boucle ne
change pas. `MINICODE_THINKING=0` cache la réflexion.

Après chaque appel, une ligne `· 2121 tokens envoyés (dont 1603 déjà en cache), 338 reçus, 7.0 s`.
**Regarde le total grossir** : c'est tout l'historique, renvoyé à chaque appel. La part
« en cache » est le début de la conversation que le serveur a déjà calculé (*prompt caching*) :
il ne recalcule que la partie nouvelle. Attention : dans l'API (Ollama comme Anthropic),
`input_tokens` ne compte **que** la partie hors cache ; le total = `input_tokens` +
`cache_read_input_tokens`. Une première version de minicode n'affichait que `input_tokens`,
et le compteur semblait… diminuer.

**Journal.** Chaque session écrit `.minicode/traces/AAAAMMJJ-HHMMSS.jsonl` (dans le projet
où tu lances minicode ; `MINICODE_TRACE=0` pour désactiver) :
- 1re ligne : le prompt système et les outils ;
- puis une ligne par appel : la liste `messages` **complète** envoyée, la réponse, les tokens.

```bash
uv run show_trace.py              # résumé lisible du dernier journal (ce qui est AJOUTÉ à chaque appel)
uv run show_trace.py --call 2     # la requête brute complète de l'appel n°2
# avec jq : la croissance de l'historique, appel par appel
jq -c 'select(.type=="call") | {msgs: (.request_messages|length), tokens: (.usage.input_tokens + (.usage.cache_read_input_tokens // 0)), s: .seconds}' .minicode/traces/*.jsonl
```

**Ce que le journal montre** (essai réel, qwen3:8b, « crée un jeu de devinette ») :

```
── appel 1  1 messages envoyés · 847 tokens → 1343 tokens · 21.03 s · stop=tool_use
   + user      texte       Crée un petit jeu en python dans jeu.py …
   ← modèle    réflexion   Okay, the user wants me to create a simple number guessing game…
   ← modèle    tool_use    edit_file({"new_string": "import random\n\nnumber = …
── appel 2  3 messages envoyés · 1370 tokens → 314 tokens · 4.94 s · stop=end_turn
   + user      tool_result Fichier créé : jeu.py (13 lignes)
   ← modèle    texte       Le fichier `jeu.py` a été créé avec le code du jeu. …
```

Sur 21 s, presque tout est de la réflexion (1343 tokens) : sans streaming, tu aurais regardé
un écran vide.

**Côté Anthropic** : `eager_input_streaming` fait arriver les arguments d'outils au fil de
l'eau, mais l'API ne les valide plus. D'où `validate_input()` dans `tools.py` (utile aussi
contre les erreurs des petits modèles) et la relance si le JSON reçu est illisible.

### Tester un programme interactif (`bash` + `stdin`)

`bash` n'a pas de clavier : un programme qui fait `input()` reçoit « fin de saisie » et plante
(`EOFError`). Le paramètre optionnel `stdin` envoie du texte comme s'il était tapé, une ligne par
saisie. Ce qu'on a appris en le testant avec qwen3:8b sur le jeu de devinette :

| Essai | Ce que le modèle a fait | Correction dans le harness |
|---|---|---|
| 1 | a recopié l'exemple `"50\n75\n62\n"` de la description de l'outil | plus d'exemple concret : **les petits modèles recopient les exemples** |
| 2 | a inventé que le jeu demandait un nom, sans lire le fichier ; puis a dit à l'utilisateur de fournir les saisies | consignes : « lis le programme avant de le tester », « vérifie toi-même » |
| 3 | a remplacé le nombre aléatoire par `50` pour que son test passe | consigne : « ne modifie jamais un programme juste pour qu'un test passe ». Sans YOLO, la demande de permission t'aurait montré `+ number = 50` |
| 4 | test réussi : `EOFError` → indice → relance avec plus de saisies → « Félicitations! », code 0 | — |

### Sessions interactives (`interactive_start` / `interactive_send`)

Limite de `stdin` : toutes les réponses sont données **d'avance**. Avec un jeu qui répond
« Plus grand ! » et limite à 7 essais, impossible d'adapter son coup suivant. Deux outils :

- `interactive_start(command)` lance le programme dans un **pseudo-terminal** (sinon Python
  garde ses `print()` en mémoire et la question n'arrive jamais) et renvoie ce qu'il affiche
  jusqu'à ce qu'il **se taise 0,5 s** : c'est ainsi qu'on devine qu'il attend une saisie ;
- `interactive_send(session_id, text)` tape une ligne et renvoie la réponse.

`interactive_start` demande la permission (il exécute du code, mêmes règles que `bash`) ;
`interactive_send` non (il tape dans un programme déjà autorisé). Les programmes encore
ouverts sont tués à la fin de chaque demande, même après Ctrl-C.

Essai réel avec qwen3:8b sur un jeu à 7 essais et 60 s :
```
⏺ Run(python3 jeu.py)
⏺ Type(50 → session 1)   ⎿ Plus grand !
⏺ Type(75 → session 1)   ⎿ Plus grand !
⏺ Type(88 → session 1)   ⎿ Plus grand !
⏺ Type(94 → session 1)   ⎿ Plus petit !
⏺ Type(91 → session 1)   ⎿ Plus grand !
⏺ Type(92 → session 1)   ⎿ Plus grand !
⏺ Type(93 → session 1)   ⎿ Félicitations ! Vous avez trouvé le nombre.
```
Ce qu'il a fallu pour en arriver là :
- au début, qwen3 **ignorait les nouveaux outils** et refaisait `bash` + `stdin`. La description
  de `bash` disait « pour un programme qui pose des questions, donne les réponses dans stdin » :
  elle l'envoyait au mauvais endroit. Corrigé, mais ça ne suffisait pas…
- … ce qui a marché : **mettre les outils interactifs AVANT `bash` dans la liste**. Les petits
  modèles sont sensibles à l'ordre des outils ;
- un bug du harness : le programme ferme son affichage quelques millisecondes avant que son
  processus disparaisse, et minicode disait « session toujours ouverte » après la victoire.

### Quand le modèle casse le code en voulant le modifier

Histoire vraie (journal de `test_ai/`) :
1. « ajoute une limite de temps » → qwen3 **réécrit tout le fichier** de mémoire… et oublie la
   ligne `current_attempts = 0`. Le jeu plante (`NameError`). Le diff montrait la ligne retirée,
   mais noyée dans un fichier entier.
2. « corrige cette erreur » → il retente de remplacer **tout le fichier** et se trompe en
   recopiant : une ligne `import time` oubliée, puis **`时间_limit`** au lieu de `time_limit`
   (qwen3 est un modèle chinois). 4 échecs, puis il abandonne en affichant l'appel en texte.
3. Après une première correction du harness, il corrige… en cassant l'indentation
   (`IndentationError`) et annonce « corrigé ».

Trois protections dans `edit_file`, toutes vérifiées AVANT la demande de permission :
- **pas de gros remplacement** : un `old_string` de 15 lignes ou plus est refusé (« modifie
  seulement les lignes concernées »). C'est aussi ce qui évite la ligne oubliée du point 1 ;
- **une erreur qui dit OÙ ça diverge** au lieu de « introuvable » :
  `correspond au fichier jusqu'à la ligne 7, puis diffère : fichier 'time_limit = 60' / toi '时间_limit = 60'` ;
- **vérification de la syntaxe Python** : une modification qui casserait un fichier `.py` valide
  est refusée, avec l'erreur exacte (`IndentationError ligne 17`).

Résultat sur la même demande : `current_attempts = 0` ajouté au bon endroit en 3 appels.
Limite : une erreur **logique** (remettre le compteur à 0 *dans* la boucle) reste du code valide ;
seuls des tests ou une exécution l'attrapent.

Enfin, les notes du harness (`[indice minicode …]`, `[code de sortie …]`, `[session …]`)
sont maintenant **toujours affichées** sous un résultat, même quand la sortie est coupée :
tu vois ce que le harness dit au modèle.

Et à l'essai 4 (stdin), son **résumé était inventé** (« 6 coups, 64 → Félicitations ») alors que le
journal montre 2 coups, sur 75. Ne crois pas le récit de l'agent : vérifie avec le journal.

```bash
# ce que les commandes ont VRAIMENT affiché (les tool_result du journal)
jq -r 'select(.type=="call") | .request_messages[-1].content | if type=="array" then .[] | select(.type=="tool_result") | .content else empty end' .minicode/traces/FICHIER.jsonl
```

**Exercices :**
- Pose 5 questions à la suite, puis lance la commande `jq` ci-dessus : comment évolue `tokens` ?
- Énigme : sur un vrai journal, le total est passé de 2662 à 1589 tokens alors qu'on avait
  ajouté des messages. Piste : compare ce qui est envoyé (`--call N`) et ce que qwen3 garde
  de ses anciennes réflexions (bloc `thinking`).
- Avec `--call N`, retrouve dans la requête brute le `tool_use_id` qui relie un `tool_use` à son `tool_result`.
- Compare la durée de réflexion avec `MINICODE_MODEL=qwen3:4b`.

## Étape 8 : la gestion du contexte

Tout ce qu'on envoie (prompt système + outils + **tout** l'historique) doit tenir dans la
**fenêtre de contexte** du modèle, réponse comprise : 8 192 tokens avec notre réglage Ollama.
Le prompt système et les outils en prennent déjà ~2 000. Si ça déborde, Ollama coupe le
début **en silence** et le modèle oublie sans prévenir. Le harness surveille donc la taille
avant **chaque** appel (`fit_context()` dans `minicode.py`, le reste dans `context.py`).

**1. Compter.** On ne peut pas compter les tokens exactement sans le modèle, alors on estime :
nombre de caractères du JSON ÷ 3. Après chaque appel, le serveur donne le **vrai** nombre
(`usage`) : on en tire un facteur de correction pour les estimations suivantes (`Budget.calibrate`).

**2. Prévenir.** Un résultat d'outil est plafonné à ~20 % de la fenêtre (6 000 caractères
avec 8k). Un gros fichier se lit **par morceaux** : `read_file` finit par
`[lignes 1-140 sur 640. Suite : read_file avec start_line=141]`.

**3. Faire de la place**, au-delà de 70 % de la fenêtre (les 30 % restants sont pour la réponse),
du moins cher au plus cher :

| Étape | Quoi | Coût |
|---|---|---|
| a | **Effacer les vieux résultats d'outils** (sauf les 2 derniers). Le `tool_use` reste : le modèle sait ce qu'il a fait, et peut relancer l'outil. | gratuit |
| b | **Résumer les tours précédents** : un appel au modèle, sans outils, écrit un résumé qui remplace le début de l'historique. La demande en cours n'est jamais résumée : on ne sépare jamais un `tool_use` de son `tool_result`. | 1 appel |
| c | En dernier recours, ne garder que le dernier résultat d'outil, et prévenir. | gratuit |

Après un résumé, l'historique commence par deux messages : `user` « [Résumé de la
conversation précédente…] » puis `assistant` « Compris… ». Il en faut deux, pour que les
rôles continuent d'alterner. Si le résumé échoue, minicode garde au moins la liste des
demandes et des fichiers (`fallback_summary`).

Ce qui s'affiche :

```
✻ Contexte allégé : 3 ancien(s) résultat(s) d'outil effacé(s) (~6100 → ~3900 tokens)
✻ Terminé en 48 s · 6 appel(s) au modèle · contexte 4210/8192 tokens (51 %)
```

- `/context` montre où partent les tokens (prompt, demandes, outils, réflexion) ;
- `/compact` force le résumé de toute la conversation ;
- `MINICODE_CONTEXT_WINDOW=16384` si tu lances `ollama serve` avec un autre `OLLAMA_CONTEXT_LENGTH` ;
- le journal note chaque compactage (`"type": "compact"`, avec le résumé).

Un détail : comme le début de l'historique peut changer pendant une demande, Ctrl-C ne
peut plus annuler avec un simple indice (`del messages[start:]`). `main()` garde donc une
**copie** de la liste avant chaque demande. `clear_old_tool_results` crée de nouveaux `dict`
au lieu de modifier les anciens, pour que cette copie reste intacte.

**Exercices :**
- `MINICODE_CONTEXT_WINDOW=4000 uv run minicode.py`, puis demande de lire 3 fichiers :
  regarde quand le harness efface, puis résume.
- Fais `/compact`, puis lis le résumé dans le journal : qu'est-ce que qwen3 a oublié ?
- Pourquoi effacer le contenu d'un `tool_result` plutôt que supprimer la paire `tool_use`/`tool_result` ?

## Étape 9 : les sous-agents (outil `task`)

Pour répondre à « comment marchent les permissions ? », l'agent lit plusieurs fichiers.
Ces fichiers restent ensuite dans **son** historique, et la fenêtre de 8k se remplit de texte
dont il n'a plus besoin. L'idée : **déléguer**. L'outil `task` lance une deuxième boucle
d'agent (la même qu'à l'étape 3) avec :

- un historique **vide** : juste la consigne écrite par l'agent principal ;
- son propre prompt système (+ les `AGENTS.md`), et seulement `read_file`, `list_dir`, `grep` ;
- 12 appels au maximum. Au dernier, minicode lui dit « écris ton rapport maintenant ».

Le sous-agent lit ce qu'il veut. À la fin, **seul son rapport** devient le `tool_result` de
`task` ; tout le reste est jeté. Ce n'est ni un autre modèle ni un autre programme : même
modèle, même client, même code de boucle. **Un sous-agent, c'est juste une autre liste
`messages`** (→ `run_subagent()` dans `minicode.py`, `subagent.py`).

```
⏺ Task(Trouver la gestion des permissions)
  ⎿  sous-agent : Trouver la gestion des permissions (historique vide, lecture seule)
     ↳ Search("check_permission|allow|deny" dans .)
     ✓ terminé en 2 appel(s) au modèle
  ⎿  rapport du sous-agent : 16 lignes (seul ce rapport entre dans le contexte)
```

Le journal le montre bien (`uv run show_trace.py`) : l'appel du sous-agent part avec
**1 message**, sans la conversation principale, et l'agent principal ne reçoit que le rapport.

```
── appel 1  1 messages envoyés · 2239 tokens       ← tool_use task({...})
── appel 2 (sous-agent)  1 messages envoyés · 991 tokens
── appel 3 (sous-agent)  3 messages envoyés · 1636 tokens   ← texte : **RAPPORT** …
── appel 4  3 messages envoyés · 2819 tokens       ← tool_result : **RAPPORT** …
```

Choix de conception :
- **Lecture seule**, sans permission à demander : l'agent principal ne voit pas ce que fait
  le sous-agent, et tu verrais ses demandes hors contexte. Moins de pouvoirs = moins de dégâts.
- **Pas de `task` pour le sous-agent** : pas de sous-sous-agents à l'infini.
- **Le sous-agent ne voit pas la conversation** : la description de l'outil insiste pour que
  l'agent principal écrive une consigne complète. C'est la principale source d'erreurs.
- Son texte n'est **pas affiché** (c'est un rapport pour l'agent principal), et ses appels ne
  changent pas le contexte affiché en bas : il a son propre budget (étape 8).

**Exercices :**
- Pose la même question avec et sans « utilise task », puis compare le contexte avec `/context`.
- Lis la consigne écrite par qwen3 pour le sous-agent (`show_trace.py --call 2`) : est-elle
  assez complète pour quelqu'un qui n'a pas vu la conversation ?
- Que faudrait-il changer pour qu'un sous-agent puisse lancer les tests (`bash`) sans danger ?

## Exercices pour les étapes 1–3

- Ajoute un `print(messages)` avant chaque appel et regarde la liste grossir.
- Casse volontairement une description d'outil (ex. « ne fait rien ») : que fait le modèle ?
- Supprime `messages.append({"role": "assistant", ...})` : quelle erreur renvoie l'API, et pourquoi ?

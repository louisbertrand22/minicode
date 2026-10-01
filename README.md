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
- [ ] 6. Contexte projet : charger un `AGENTS.md` dans le prompt système
- [x] 7. Streaming (réflexion 💭 + réponse en direct, tokens par appel) + journal JSONL + `show_trace.py`
- [ ] 8. Gestion du contexte : compter les tokens, résumer les vieux tours
- [ ] 9. Sous-agents : un outil `task` qui lance une boucle avec son propre contexte
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

Et à l'essai 4, son **résumé était inventé** (« 6 coups, 64 → Félicitations ») alors que le
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

## Exercices pour les étapes 1–3

- Ajoute un `print(messages)` avant chaque appel et regarde la liste grossir.
- Casse volontairement une description d'outil (ex. « ne fait rien ») : que fait le modèle ?
- Supprime `messages.append({"role": "assistant", ...})` : quelle erreur renvoie l'API, et pourquoi ?

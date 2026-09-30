# minicode

Un mini agent de code (façon Claude Code) écrit **à la main**, sans framework d'agent,
pour comprendre ce qu'est un *harness* : tout le code autour du modèle (boucle, outils,
contexte, permissions, mémoire).

## Lancer (gratuit, avec Ollama)

Le modèle tourne sur ta machine : pas de clé, pas de frais.

```bash
sudo pacman -S ollama-cuda                 # Arch + carte NVIDIA (sinon : https://ollama.com/download)
OLLAMA_CONTEXT_LENGTH=16384 ollama serve   # dans un terminal à part, laisse-le tourner
ollama pull qwen3:8b                       # ~5 Go, une seule fois
uv run minicode.py                         # dans un autre terminal
```

⚠️ **`OLLAMA_CONTEXT_LENGTH` est important.** Avec moins de 24 Go de VRAM, Ollama limite
le contexte à 4k tokens par défaut, et **coupe l'historique en silence** quand ça déborde :
le modèle "oublie" alors le début de la conversation ou les résultats d'outils.
Vérifie avec `ollama ps`. Si c'est trop lent (le modèle déborde sur le CPU), essaie `8192`.

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
- [ ] 4. Outils d'écriture : `grep`, `edit_file` (remplacement exact), `bash`
- [ ] 5. Permissions : demander o/n avant `bash` et les écritures, liste d'autorisations
- [ ] 6. Contexte projet : charger un `AGENTS.md` dans le prompt système
- [ ] 7. Streaming + journal JSONL de chaque requête/réponse (le plus instructif !)
- [ ] 8. Gestion du contexte : compter les tokens, résumer les vieux tours
- [ ] 9. Sous-agents : un outil `task` qui lance une boucle avec son propre contexte
- [ ] 10. Évals : 10 petites tâches, score de réussite / nombre de tours / coût

## Exercices pour les étapes 1–3

- Ajoute un `print(messages)` avant chaque appel et regarde la liste grossir.
- Casse volontairement une description d'outil (ex. « ne fait rien ») : que fait le modèle ?
- Supprime `messages.append({"role": "assistant", ...})` : quelle erreur renvoie l'API, et pourquoi ?

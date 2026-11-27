## Audit du starter

### Problème 1 : seeds incomplètes
- **Composante :** Aléa
- **Fichier / réglage :** `src/reproducibility.py`, fonction `set_reproducibility`
- **Conséquence :** seul `random` est seedé. L'initialisation des poids, le dropout et les augmentations (PyTorch) ainsi que NumPy changent à chaque exécution. Deux entraînements identiques donnent des résultats différents.
- **Solution :** ajouter `np.random.seed(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, `torch.backends.cudnn.deterministic = True` et `torch.backends.cudnn.benchmark = False`.

### Problème 2 : ordre des batchs non contrôlé
- **Composante :** Aléa
- **Fichier / réglage :** `src/reproducibility.py`, fonctions `make_generator` et `seed_worker`
- **Conséquence :** `make_generator` reçoit `seed` mais ne l'utilise pas, donc le DataLoader mélange les images différemment à chaque run. `seed_worker` fait `pass` et ne seede rien.
- **Solution :** créer le générateur avec `g = torch.Generator()` puis `g.manual_seed(seed)` et le renvoyer. Dans `seed_worker`, seeder NumPy et `random` avec `torch.initial_seed() % 2**32`.

### Problème 3 : aucune vérification des données
- **Composante :** Data
- **Fichier / réglage :** `src/validate_data.py`, fonction `validate_dataset`
- **Conséquence :** rien ne contrôle les données avant l'entraînement. Un split manquant, une classe vide ou une image corrompue passerait sans erreur.
- **Solution :** parcourir chaque split et chaque classe, comparer les comptages à `EXPECTED_SPLITS`, ouvrir chaque image avec `Image.open(p).load()`, lever une erreur si un contrôle échoue, puis écrire `reports/data_quality.json` sans timestamp.

### Problème 4 : images non versionnées
- **Composante :** Data
- **Fichier / réglage :** `.gitignore` (ligne `data/raw/`)
- **Conséquence :** Git ignore les images et aucun autre outil ne les versionne. On ne peut pas savoir quelle version des données a servi à un entraînement.
- **Solution :** `dvc init`, `dvc add data/raw`, configurer un remote DVC, puis committer `data/raw.dvc`.

### Problème 5 : pas de pipeline
- **Composante :** Pipeline (code + données)
- **Fichier / réglage :** racine du projet, `dvc.yaml` absent
- **Conséquence :** les étapes se lancent à la main dans un ordre à retenir. Après un changement, personne ne sait ce qu'il faut relancer.
- **Solution :** écrire `dvc.yaml` avec les stages `validate`, `prepare`, `train`, `evaluate` (cmd, deps, params, outs) et utiliser `dvc repro`.

### Problème 6 : aucun lien entre un modèle et ce qui l'a produit
- **Composante :** Code / Env
- **Fichier / réglage :** `src/provenance.py`, fonction `build_run_manifest`
- **Conséquence :** rien ne relie un modèle au commit Git, aux versions des bibliothèques, à l'OS ni aux données utilisées.
- **Solution :** collecter `git rev-parse HEAD`, `git status --porcelain` (état dirty), les versions via `importlib.metadata`, `platform`, l'identité des données, et écrire `reports/run_manifest.json`.

### Problème 7 : runs non enregistrés
- **Composante :** Params / historique
- **Fichier / réglage :** `src/train.py`
- **Conséquence :** aucun `mlflow.log_*`. Les paramètres, métriques par epoch, durée, artefacts et modèle ne sont pas conservés. On ne peut pas comparer deux runs.
- **Solution :** `mlflow.log_params`, `mlflow.log_metric(..., step=epoch)`, `mlflow.log_artifact`, `mlflow.pytorch.log_model(...)`.

### Problème 8 : rien n'encadre l'usage du test set
- **Composante :** Test set
- **Fichier / réglage :** `params.yaml`, `selection.run_id` (vide)
- **Conséquence :** aucune règle n'impose de choisir le modèle sur la validation avant de toucher au test. Risque de fuite : la note finale serait trop optimiste.
- **Solution :** choisir le Run sur les métriques de validation, mettre son ID dans `selection.run_id`, puis n'évaluer le test qu'une seule fois avec `dvc repro evaluate`.

## Checkpoint 1
Un `model.pt` ne contient que des poids, les noms de classes, la taille d'image et le dropout. Il ne dit pas quel code, quelles données, quelles versions de bibliothèques, quelle seed ni quel learning rate l'ont produit. Il est de plus ignoré par Git (`models/*.pt`). Sans ces informations, personne ne peut le régénérer ni vérifier que le résultat ne vient pas du hasard.

### Engineering checkpoint 2

No, `random_state=42` does not guarantee bit-for-bit identical results on every CPU/GPU/software stack. The seed only fixes the sequence of pseudo-random numbers. It does not control how the hardware computes. Floating-point operations can give slightly different results depending on the processor or GPU, and on the order in which operations are executed. Some GPU algorithms (cuDNN) are non-deterministic by design, and different versions of PyTorch, CUDA or math libraries may use different algorithms. A seed therefore gives repeatable results on the same environment, but not necessarily identical results elsewhere. This is why we also record library versions, OS and device.
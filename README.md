# deep-learning-project-examination-Sign-Language-ASL

Projet de deep learning realise dans le cadre du module **Deep Learning** du master **Ingenierie des Systemes d'Information & IA**.

## Objectif

Le projet compare deux approches pour reconnaitre les lettres statiques de l'alphabet ASL:

- **Pipeline keypoints + MLP**: extraction de 21 landmarks MediaPipe puis classification d'un vecteur de 42 coordonnees.
- **Pipeline image + CNN**: classification directe de crops RGB en 64x64.

Le depot contient:

- le notebook historique (exploration et resultats initiaux),
- une **V2 industrialisee** avec scripts CLI, sauvegarde de modeles, metriques enrichies et inference video stabilisee.

## Apercu visuel (placeholders)

> Ajoute tes captures plus tard dans `docs/screenshots/`.

![Confusion Matrix MLP](docs/screenshots/confusion_matrix_mlp.png)
![Confusion Matrix CNN](docs/screenshots/confusion_matrix_cnn.png)
![Inference Video Example](docs/screenshots/inference_video_example.png)

## Resultats de reference (rapport)

- **MLP sur keypoints**: **88,7 %** d'accuracy
- **CNN sur images RGB**: **93,6 %** d'accuracy

## Comparatif quantitatif (rapport)

### 1. Classification statique

- Gain absolu CNN vs MLP: ~**+4,9 points** d'accuracy
- Taille modele (ordre de grandeur):
  - MLP: ~**30k** parametres
  - CNN: ~**1,6M** parametres

### 2. Inference video (tests du rapport)

- FPS moyen (GTX T4):
  - MLP + landmarks: **90 FPS**
  - CNN + RGB: **42 FPS**
- Erreur moyenne sequence (Levenshtein token-level):
  - MLP + landmarks: **15,3**
  - CNN + RGB: **7,8**

Lecture rapide:

- le **MLP** est plus rapide et plus compact;
- le **CNN** est plus precis en reconnaissance reelle (meilleure robustesse sur la sequence de lettres);
- le lissage temporel (majorite sur fenetre) reduit fortement les faux positifs transitoires en video.

## Analyse d'erreurs (rapport)

- Erreurs recurrentes sur des lettres morphologiquement proches: `C/Q`, `F/P`, `M/N`, `U/V/W`.
- Pipeline landmarks (MLP): sensible aux **occlusions du pouce** et a la perte de main hors champ.
- Pipeline RGB (CNN): sensible au **blur** et aux variations d'echelle extremes, mais meilleur sur les recouvrements partiels grace aux indices de texture/couleur.

## Ameliorations V2 implementees

- separation claire des composants: pretraitement, entrainement, evaluation, inference;
- extraction landmarks plus robuste:
  - normalisation rotation + echelle paume,
  - filtre de qualite,
  - rejet des frames instables;
- entrainement CNN avec augmentation de donnees;
- option backbone pre-entraine leger (**MobileNetV2**);
- evaluation plus complete:
  - accuracy,
  - F1 macro,
  - recall macro,
  - rapport par classe,
  - matrice de confusion sauvegardee;
- sauvegarde systematique des artefacts (`.keras`, `classes.npy`, `metrics.json`);
- inference video avec stabilisation temporelle (fenetre + EMA + seuil de confiance + dedoublonnage).

## Stack technique

- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- scikit-learn
- NumPy / pandas
- matplotlib / seaborn
- gTTS

## Structure du depot (version commit)

- `Sign_Language_.ipynb`: notebook principal (version initiale)
- `DL_Report.pdf`: rapport complet
- `requirements.txt`: dependances Python
- `src/asl_v2/config.py`: configurations d'entrainement
- `src/asl_v2/data.py`: chargement images/keypoints et export CSV
- `src/asl_v2/landmarks.py`: extraction robuste MediaPipe
- `src/asl_v2/models.py`: architectures MLP/CNN (+ option pre-entrainee)
- `src/asl_v2/evaluation.py`: metriques et matrice de confusion
- `src/asl_v2/temporal.py`: stabilisation temporelle video
- `scripts/train_mlp.py`: pipeline entrainement/evaluation MLP
- `scripts/train_cnn.py`: pipeline entrainement/evaluation CNN
- `scripts/infer_video.py`: inference video stabilisee

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

### 1. Entrainement MLP (landmarks)

```bash
python scripts/train_mlp.py --dataset-root "C:\path\to\asl_alphabet_train"
```

Options utiles:

- `--max-per-class 300`
- `--epochs 30`
- `--artifacts-dir artifacts/mlp`

### 2. Entrainement CNN (images)

```bash
python scripts/train_cnn.py --dataset-root "C:\path\to\asl_alphabet_train"
```

Options utiles:

- `--use-pretrained` (active MobileNetV2)
- `--trainable-backbone`
- `--max-per-class 500`

### 3. Inference video stabilisee

CNN:

```bash
python scripts/infer_video.py --mode cnn --video "C:\path\to\video.mp4" --model artifacts/cnn/cnn_model.keras --classes artifacts/cnn/classes.npy
```

MLP landmarks:

```bash
python scripts/infer_video.py --mode mlp_landmarks --video "C:\path\to\video.mp4" --model artifacts/mlp/mlp_model.keras --classes artifacts/mlp/classes.npy
```

## Competences demontrees

- computer vision appliquee a la reconnaissance de gestes;
- extraction et normalisation de points-cles;
- entrainement et comparaison de modeles TensorFlow/Keras;
- evaluation multiclasses et analyse d'erreurs;
- conception d'une chaine d'inference de bout en bout plus proche d'un usage reel.

## Notes

- Le dossier `artifacts/` est conserve vide pour accueillir les modeles et metriques generes localement.
- Les captures d'ecran seront ajoutees ulterieurement dans `docs/screenshots/`.

## Resume candidature (copiable CV)

Projet Deep Learning de reconnaissance de l'alphabet ASL statique comparant deux pipelines (MediaPipe landmarks + MLP vs RGB + CNN), avec evaluation multiclasses, inference image/video et stabilisation temporelle. Resultats du rapport: **88,7 %** (MLP) vs **93,6 %** (CNN), avec un compromis clair entre vitesse (MLP: **90 FPS**) et precision sequence (CNN: erreur Levenshtein **7,8** vs **15,3**).

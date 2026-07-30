# Reconnaissance de Langue des Signes — Video_detection

Implémentation d'un pipeline de reconnaissance de la langue des signes **from scratch**, en Python, combinant l'extraction de landmarks avec **MediaPipe** et l'entraînement de modèles séquentiels (**LSTM / CNN**) sur le dataset **WLASL** (Word-Level American Sign Language).

## 🎯 Objectif

Construire un système capable de reconnaître des signes à partir de séquences vidéo (fichiers pré-enregistrés ou flux webcam en temps réel), en extrayant les points clés des mains comme représentation intermédiaire avant classification.

## 🧠 Méthodologie

Le pipeline suit les étapes suivantes :

1. **Extraction des landmarks** : détection des points clés des mains à partir de chaque frame vidéo grâce à MediaPipe Hands, produisant un vecteur de coordonnées (x, y, z) par landmark
2. **Construction des séquences** : agrégation des landmarks extraits frame par frame en séquences temporelles de longueur fixe, représentant un signe complet
3. **Prétraitement des données** : nettoyage des métadonnées JSON du dataset, gestion des vidéos manquantes ou corrompues, normalisation des séquences
4. **Entraînement du modèle** : apprentissage supervisé via une architecture LSTM et/ou CNN pour classifier les séquences de landmarks en classes de signes
5. **Inférence temps réel** : capture webcam, extraction des landmarks en direct, et prédiction du signe correspondant

## 📊 Données

- **Source** : [WLASL (Word-Level American Sign Language) Dataset](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)
- **Contenu** : vidéos annotées de signes, accompagnées de métadonnées JSON (gloss, identifiant vidéo, split train/val/test)
- **Cible** : classe du signe (gloss)
- **Variables d'entrée** : séquences de landmarks des mains extraits par MediaPipe

## 🛠️ Stack technique

- `mediapipe` — extraction des landmarks des mains (vidéo et webcam)
- `opencv-python` — lecture et traitement des flux vidéo
- `numpy` — manipulation des séquences de landmarks
- `tensorflow` / `keras` — implémentation des modèles LSTM / CNN
- `pandas` — manipulation des métadonnées JSON du dataset

## ⚙️ Environnement

- **OS** : Windows
- **IDE** : VS Code
- **Terminal** : Git Bash
- **Environnement virtuel** : `mon_env`
- Téléchargement du dataset via la **Kaggle CLI**

## 📁 Structure du projet

```
.
├── data/                  # Dataset WLASL (vidéos + métadonnées JSON)
├── dataset_hands_only/    # Fichiers numpy des landmarks extraits par Mediapipe
├── env_training/          # Environnement virtuel créer pour l'entrainement des modèles

├── mon_env                # Environnement virtuel pour l'extraction des landmarks
├── src                    # Fonctions utilisées
├── model/                 # Modèles retenus
└── README.md
```

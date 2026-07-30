import json
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from src.normaliser import normalize_hand, build_trajectory


def _extract_sequence_from_video(cap, hands):
    """
    Lit toute la vidéo et renvoie une séquence (T, 132) :
    par frame -> [lh_shape(63), lh_traj(3), rh_shape(63), rh_traj(3)]

    Le bloc par main est contigu (66 valeurs) pour que mirror_hands()
    dans augmentation.py reste simple (juste échanger les deux blocs).
    """
    lh_shapes, rh_shapes = [], []
    lh_wrists, rh_wrists = [], []
    lh_scales, rh_scales = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        lh_shape = np.zeros(63)
        rh_shape = np.zeros(63)
        lh_wrist = np.zeros(3)
        rh_wrist = np.zeros(3)
        lh_scale = 1e-6
        rh_scale = 1e-6

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # "Left" ou "Right"
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                shape_norm, wrist_raw, scale = normalize_hand(pts)
                if label == "Left":
                    lh_shape, lh_wrist, lh_scale = shape_norm, wrist_raw, scale
                else:
                    rh_shape, rh_wrist, rh_scale = shape_norm, wrist_raw, scale

        lh_shapes.append(lh_shape)
        rh_shapes.append(rh_shape)
        lh_wrists.append(lh_wrist)
        rh_wrists.append(rh_wrist)
        lh_scales.append(lh_scale)
        rh_scales.append(rh_scale)

    if len(lh_shapes) == 0:
        return None

    lh_shapes = np.array(lh_shapes)  # (T, 63)
    rh_shapes = np.array(rh_shapes)  # (T, 63)

    # échelle de référence = échelle de la 1ère frame valide (sinon moyenne des échelles valides)
    valid_lh = [s for s in lh_scales if s > 1e-6]
    valid_rh = [s for s in rh_scales if s > 1e-6]
    lh_scale_ref = lh_scales[0] if lh_scales[0] > 1e-6 else (np.mean(valid_lh) if valid_lh else 1e-6)
    rh_scale_ref = rh_scales[0] if rh_scales[0] > 1e-6 else (np.mean(valid_rh) if valid_rh else 1e-6)

    lh_traj = build_trajectory(lh_wrists, lh_scale_ref)  # (T, 3)
    rh_traj = build_trajectory(rh_wrists, rh_scale_ref)  # (T, 3)

    sequence = np.hstack([lh_shapes, lh_traj, rh_shapes, rh_traj])  # (T, 132)
    return sequence


def process_wlasl_hands(json_file, videos_dir, target_len, classes):
    X = []
    etiquettes = []
    mp_hands = mp.solutions.hands

    data = json.loads(json_file)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        for entry in data:
            gloss = entry['gloss']
            for instance in entry['instances']:
                video_id = str(instance['video_id']).zfill(5)
                video_path = os.path.join(videos_dir, f"{video_id}.mp4")

                if not os.path.exists(video_path):
                    print(f"Video manquante : {video_path}")
                    continue

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Impossible d'ouvrir : {video_path}")
                    continue

                sequence = _extract_sequence_from_video(cap, hands)
                cap.release()

                if sequence is not None:
                    indices = np.round(np.linspace(0, len(sequence) - 1, target_len)).astype(int)
                    sequence = sequence[indices]  # (target_len, 132)
                    X.append(sequence)
                    etiquettes.append(classes[gloss])
                    print(f"Traité : {gloss}/{video_id} (shape : {sequence.shape})")

    X = np.array(X)
    y = np.array(etiquettes)
    print(f"Tenseur final : {X.shape}, labels : {y.shape}")
    return X, y


def build_video_index(vid_asl):
    """Parcourt tous les sous-dossiers et associe nom_de_fichier -> chemin complet"""
    index = {}
    for root, dirs, files in os.walk(vid_asl):
        for f in files:
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                index[f] = os.path.join(root, f)
    return index


def process(tab_csv, vid_asl, target_len, classes):
    mp_hands = mp.solutions.hands
    X = []
    y = []

    df = pd.read_csv(tab_csv)
    df = df[df["Gloss"].isin(classes.keys())]
    video_index = build_video_index(vid_asl)
    print(f"{len(video_index)} vidéos trouvées dans {vid_asl}")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        for _, row in df.iterrows():
            num = row["Video file"]
            gloss = row["Gloss"]

            if gloss not in classes:
                print(f"Ce gloss {gloss} n'est pas dans la classe")
                continue
            if num not in video_index:
                print(f"Video manquante : {num}")
                continue

            chemin = video_index[num]
            cap = cv2.VideoCapture(chemin)
            if not cap.isOpened():
                print(f"Impossible d'ouvrir : {chemin}")
                continue

            sequence = _extract_sequence_from_video(cap, hands)
            cap.release()

            if sequence is not None:
                indices = np.round(np.linspace(0, len(sequence) - 1, target_len)).astype(int)
                sequence = sequence[indices]  # (target_len, 132)
                X.append(sequence)
                y.append(classes[gloss])
                print(f"Traité : {num} (shape : {sequence.shape})")

    X = np.array(X)
    y = np.array(y)
    print(f"Tenseur final : {X.shape}, labels : {y.shape}")
    return X, y
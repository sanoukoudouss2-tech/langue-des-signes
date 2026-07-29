import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import json
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from src.normaliser import normalize_hand

def process_wlasl_hands(json_file,videos_dir, target_len,classes):
    # toujours dans la meme logique, classes c'est pour retourner mes etiquettes en fonction des gloss
    # Mon tenseur X
    X = []

    mp_hands = mp.solutions.hands
    ## j'initialise a 0 les etiquettes qui vont contenir les labels pour mon entrainement
    etiquettes = []
 

    ## J'ouvre le fichier json qui contient les données des vidéos qui m'intéressent
    
    data = json.loads(json_file)## loa&d lit un fichier json et renvoie l'équilavent en type python: liste;dictionnaire;etc

    # Initialisation de MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        for entry in data:
            gloss = entry['gloss']## je recupere les mots qui m'int"resse dans le fichier json filtré


            for instance in entry['instances']:## instances contient les données sur la video: le debut, la fin, l'id etc
                ## je recupere l'id de la video et le chemin
                video_id = str(instance['video_id']).zfill(5)
                video_path = os.path.join(videos_dir, f"{video_id}.mp4")
                if not os.path.exists(video_path):
                    print(f"Video manquante : {video_path}")
                    continue

                cap = cv2.VideoCapture(video_path)## je lis la video à travers son chemin

                if not cap.isOpened():
                    print(f"Impossible d'ouvrir : {video_path}")
                    continue

                ## sequence va contenir les handemarks de chaque frame de la video du debut à la fin.
                ## Les handmarks des deux mains seront concatenées
                sequence = []

                while cap.isOpened():
                    ## cap.read() renvoie un tuple ret qui est est un booleen et frame qui represente l'image sous forme de tableau numpy
                    ## au prochain appel de .read() c'est l'image suivante qui est lue
                    ret, frame = cap.read()
                    if not ret :
                        break

                    # Conversion BGR -> RGB car mediapipe fonctionne sous RGB
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(image_rgb)
                    ## hands.process renvoie un objet results qui contient multi_hand_landmarks,multi_handedness et multi_hand_world_landmarks
                    ## multi_hand_landmarks est une liste des 21 coordonnées de chaque main detectée.
                    #  Pour chaque point de 0 a 21 on a 3 coordonnées, x,y,z
                    ## handedness indique la latéralité de chaque main(droite ou gauche) avec un certain niveau de confiance                        # Vecteurs vides par défaut (63 valeurs par main)
                    lh = np.zeros(21 * 3)
                    rh = np.zeros(21 * 3)

                    # Extraction si des mains sont détectées
                    if results.multi_hand_landmarks and results.multi_handedness:
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            label = handedness.classification[0].label # "Left" ou "Right" pour savoir s'il s'agit d'une main droite ou gauche
                            ## Là j'applatis les handmarks de chaque point de 0 a 20 
                            ## Ma matrice pts contient les handmarks de ma frame active
                            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                            pts = normalize_hand(pts)

                            if label == "Left":
                                lh = pts
                            else:
                                rh = pts

                    # Fusion des 2 mains (126 features)
                    frame_features = np.hstack([lh, rh])
                    sequence.append(frame_features)

                    

                cap.release()

                if len(sequence) > 0:

                    sequence = np.array(sequence)  # shape: (nb_frames_original, 126)
                    indices = np.round(np.linspace(0, len(sequence) - 1, target_len)).astype(int)
                    sequence = sequence[indices]  # shape: (target_len, 126)

                    X.append(sequence)
                    etiquettes.append(classes[gloss])

                    print(f"Traité : {gloss}/{video_id} (shape : {sequence.shape})")


    X = np.array(X)
    y = np.array(etiquettes)  
        

    print(f"Tenseur final : {X.shape}, labels : {y.shape}")
    return X,y




def build_video_index(vid_asl):
    """Parcourt tous les sous-dossiers et associe nom_de_fichier -> chemin complet"""
    index = {}
    for root, dirs, files in os.walk(vid_asl):
        for f in files:
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                index[f] = os.path.join(root, f)
    return index



def process(tab_csv, vid_asl, target_len,classes):

    mp_hands = mp.solutions.hands
    # classes represente la maniere dont sont etiqueté les données dans le fichier csv

    X = []
    y = []  # on construit y en même temps que X, vidéo par vidéo

    """ classes = {"BEFORE": 0, "THIN": 1,
                "COOL1": 2, "COOL2": 2, "COOL3": 2, "COOL4": 10,
               "DRINK1": 3, "DRINK2": 3,
               "GO": 4, "COMPUTER": 5, "WHO": 6, "COUSIN": 7, "HELP": 8,
               "CANDY1": 9, "CANDY2": 9} """


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

        ## iterrows() retourne un tuple  qui marque l'index et la ligne correpondante
        for _, row in df.iterrows():
            num = row["Video file"]
            gloss = row["Gloss"]

            if gloss not in classes:
                print(f"Ce gloss {gloss} n'est pas dans la classe")
                continue  # on ignore les glosses hors de nos 10 classes

            if num not in video_index:
                print(f"Video manquante : {num}")
                continue
            
            chemin = video_index[num]

            cap = cv2.VideoCapture(chemin)
            if not cap.isOpened():
                print(f"Impossible d'ouvrir : {chemin}")
                continue

            sequence = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                lh = np.zeros(21 * 3)
                rh = np.zeros(21 * 3)

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = handedness.classification[0].label
                        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                        pts = normalize_hand(pts)
                        if label == "Left":
                            lh = pts
                        else:
                            rh = pts

                frame_features = np.hstack([lh, rh])
                sequence.append(frame_features)

            # tout ceci est maintenant HORS du while, une seule fois par vidéo
            cap.release()

            if len(sequence) > 0:
                sequence = np.array(sequence)
                indices = np.round(np.linspace(0, len(sequence) - 1, target_len)).astype(int)
                sequence = sequence[indices]

                X.append(sequence)
                y.append(classes[gloss])

                print(f"Traité : {num} (shape : {sequence.shape})")

    X = np.array(X)
    y = np.array(y)


    print(f"Tenseur final : {X.shape}, labels : {y.shape}")
    return X, y
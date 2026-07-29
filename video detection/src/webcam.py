import cv2
import os
import mediapipe as mp
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from prediction import predict_class_with_confidence
from normaliser import normalize_hand

def webcam(interpreter,input_details,output_details,classes):
    # Initialisation des modeles mediapipe
    ## mp_hands et mp_draws sont des modules qui permettent respectivement de suivre les mains(landsmarks) et de les dessiner à l'écran
    enregistrement = False
    mot = None
    confiance = None
    a_predit = False
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    res = []
    ## hands = mp_hands.Hands est un detecteur de main
    hands = mp_hands.Hands(max_num_hands = 2,min_detection_confidence=0.7)
    ## Je commence à utiliser la webcam
    s = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if not s.isOpened():
        print("Erreur: Impossible d'ouvrir la vidéo")
        exit()

    enregistrement = False ## passe a True des que la touche espace est appuyée
    nb_frames_collecte = 125 ## nombre de frames a collecter une fois l'enregistrement demarré
    nb_frames_gardees = 100 ## nombre de frames finalement conservées (les dernieres)

    while s.isOpened():
        ## ret renvoie un booleen qui indique si l'image actuelle est bien chargée
        ## frame designe la frame de l'image suivante
        ## Il existe un curseur qui va a la frame suivante à chaque fois qu'on appel s.read()
        ret, frame = s.read()
        ## mediapipe fonctionne en rgb
        if not ret:
            ## si ret est a False ca signifie que la frame suivante n'a pas charger. Ce qui veut dire qu'on est a la fin ou qu'il y'a un probleme
            print("Fin de la vidéo.")
            break
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        ## results contient differentes données concernant les landmarks et les handedness
        results = hands.process(frame_rgb)

        # Vecteurs vides par défaut (63 valeurs par main), reinitialisés a chaque frame
        lh = np.zeros(21 * 3)
        rh = np.zeros(21 * 3)

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

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Fusion des 2 mains (126 features)
        frame_features = np.hstack([lh, rh])## chercher le axis qui concentene verticalement

        if enregistrement :
            res.append(frame_features)
            cv2.putText(frame, f"REC {len(res)}/{nb_frames_collecte}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,150,255), 2)

            if len(res) >= nb_frames_collecte:
                ## la je retourne ce que je vais entraine et j'entraine
                res = np.array(res)
                res = res[-nb_frames_gardees:]
                mot, confiance = predict_class_with_confidence(res, interpreter, input_details, output_details,classes, seuil=0.8)
                a_predit = True
                enregistrement = False
                res = []
                cv2.putText(frame, f" mot : {mot}; confiance ; {confiance}  ", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100), 2)
                

        else :
            cv2.putText(frame, "Appuyer sur ESPACE pour démarrer", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if  a_predit:
                if mot is not None:
                    cv2.putText(frame, f"Dernier signe : {mot} ({confiance:.1%})", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                else:
                    cv2.putText(frame, f"Signe non reconnu (confiance trop faible: {confiance:.1%})", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
        cv2.imshow("Lecteur Video OpenCV", frame)
        

            

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and not enregistrement:
            enregistrement = True
            res = []
        if key == ord('q'):
            break

    s.release()
    cv2.destroyAllWindows()






if __name__ == "__main__":
    dossier = r"C:/Users/sanou/Documents/Documents_1/CODE_ALPHA/Video_detection/models"
    interpreter = Interpreter(model_path=os.path.join(dossier, "modele_final.tflite"))
    labels = {   "THIN":0,
    "GO":1,
    "COMPUTER":2,
    "HELP":3,
    "COOL(HANDSOME)":4
}
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    webcam(interpreter, input_details, output_details, labels)
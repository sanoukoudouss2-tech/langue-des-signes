import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import mediapipe as mp
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from src.prediction import predict_class_with_confidence
from src.normaliser import normalize_hand, build_trajectory


def webcam(interpreter, input_details, output_details, classes):
    enregistrement = False
    mot = None
    confiance = None
    a_predit = False

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # Buffers de collecte pendant l'enregistrement (remplacent l'ancienne liste "res")
    lh_shapes_buf, rh_shapes_buf = [], []
    lh_wrists_buf, rh_wrists_buf = [], []
    lh_scales_buf, rh_scales_buf = [], []

    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

    s = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if not s.isOpened():
        print("Erreur: Impossible d'ouvrir la vidéo")
        exit()

    nb_frames_collecte = 125  # nombre de frames a collecter une fois l'enregistrement demarré
    nb_frames_gardees = 100   # nombre de frames finalement conservées (les dernieres)

    while s.isOpened():
        ret, frame = s.read()
        if not ret:
            print("Fin de la vidéo.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        lh_shape = np.zeros(63)
        rh_shape = np.zeros(63)
        lh_wrist = np.zeros(3)
        rh_wrist = np.zeros(3)
        lh_scale = 1e-6
        rh_scale = 1e-6

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                shape_norm, wrist_raw, scale = normalize_hand(pts)
                if label == "Left":
                    lh_shape, lh_wrist, lh_scale = shape_norm, wrist_raw, scale
                else:
                    rh_shape, rh_wrist, rh_scale = shape_norm, wrist_raw, scale

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if enregistrement:
            lh_shapes_buf.append(lh_shape)
            rh_shapes_buf.append(rh_shape)
            lh_wrists_buf.append(lh_wrist)
            rh_wrists_buf.append(rh_wrist)
            lh_scales_buf.append(lh_scale)
            rh_scales_buf.append(rh_scale)

            cv2.putText(frame, f"REC {len(lh_shapes_buf)}/{nb_frames_collecte}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 2)

            if len(lh_shapes_buf) >= nb_frames_collecte:
                # Reconstruction de la séquence (forme + trajectoire), même logique qu'à l'extraction
                lh_shapes_arr = np.array(lh_shapes_buf)
                rh_shapes_arr = np.array(rh_shapes_buf)

                valid_lh = [x for x in lh_scales_buf if x > 1e-6]
                valid_rh = [x for x in rh_scales_buf if x > 1e-6]
                lh_scale_ref = lh_scales_buf[0] if lh_scales_buf[0] > 1e-6 else (np.mean(valid_lh) if valid_lh else 1e-6)
                rh_scale_ref = rh_scales_buf[0] if rh_scales_buf[0] > 1e-6 else (np.mean(valid_rh) if valid_rh else 1e-6)

                lh_traj = build_trajectory(lh_wrists_buf, lh_scale_ref)
                rh_traj = build_trajectory(rh_wrists_buf, rh_scale_ref)

                sequence = np.hstack([lh_shapes_arr, lh_traj, rh_shapes_arr, rh_traj])  # (T, 132)
                sequence = sequence[-nb_frames_gardees:]  # garde les 100 dernières frames

                mot, confiance = predict_class_with_confidence(
                    sequence, interpreter, input_details, output_details, classes, seuil=0.8
                )
                a_predit = True
                enregistrement = False

                lh_shapes_buf, rh_shapes_buf = [], []
                lh_wrists_buf, rh_wrists_buf = [], []
                lh_scales_buf, rh_scales_buf = [], []

                cv2.putText(frame, f" mot : {mot}; confiance ; {confiance} ", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        else:
            cv2.putText(frame, "Appuyer sur ESPACE pour démarrer", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if a_predit:
            if mot is not None:
                cv2.putText(frame, f"Dernier signe : {mot} ({confiance:.1%})", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 0), 2)
            else:
                cv2.putText(frame, f"Signe non reconnu (confiance trop faible: {confiance:.1%})", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("Lecteur Video OpenCV", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and not enregistrement:
            enregistrement = True
            lh_shapes_buf, rh_shapes_buf = [], []
            lh_wrists_buf, rh_wrists_buf = [], []
            lh_scales_buf, rh_scales_buf = [], []
        if key == ord('q'):
            break

    s.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    dossier = r"C:/Users/sanou/Documents/Documents_1/CODE_ALPHA/Video_detection/models"
    interpreter = Interpreter(model_path=os.path.join(dossier, "modele_1.tflite"))
    labels = {
        "THIN": 0,
        "GO": 1,
        "COMPUTER": 2,
        "HELP": 3,
        "COOL(HANDSOME)": 4
    }
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    webcam(interpreter, input_details, output_details, labels)
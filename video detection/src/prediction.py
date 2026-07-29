import numpy as np
from ai_edge_litert.interpreter import Interpreter



def predict_class_with_confidence(landmark_sequence, interpreter, input_details, output_details, classes, seuil=0.7):

    inv_classes = {v:k for k,v in classes.items()}
    sequence_batch = np.expand_dims(landmark_sequence, axis=0).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], sequence_batch)
    interpreter.invoke()
    probabilities = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(probabilities, axis=1)[0]
    confidence = probabilities[0][predicted_index]
    
    if confidence < seuil:
        return None, confidence
    return inv_classes[predicted_index], confidence

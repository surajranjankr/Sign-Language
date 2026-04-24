import tensorflow as tf
import numpy as np
import json
from src.utils import normalize_landmarks
from src.config import LABEL_MAP_PATH, SEQUENCE_LENGTH

class SignInterpreter:
    def __init__(self, model_path='models/sign_language_model.tflite'):
        # 1. Load the TFLite model
        # We don't use load_model here!
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        
        # 2. Allocate tensors (Essential step for TFLite)
        self.interpreter.allocate_tensors()
        
        # 3. Get input and output details for the "Select TF Ops"
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open(LABEL_MAP_PATH, 'r') as f:
            data = json.load(f)
            # Map index strings back to names
            self.label_map = {int(v): k for k, v in data.items()}
            
        self.buffer = []

    def predict(self, landmarks):
        self.buffer.append(landmarks)
        self.buffer = self.buffer[-SEQUENCE_LENGTH:] 

        if len(self.buffer) == SEQUENCE_LENGTH:
            # Prepare the data (32-bit float is required for TFLite)
            input_data = normalize_landmarks(np.array(self.buffer, dtype='float32'))
            input_data = np.expand_dims(input_data, axis=0) 

            # 4. TFLite Inference Process
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # 5. Extract results
            res = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            idx = np.argmax(res)
            return idx, res[idx]
        
        return None, 0
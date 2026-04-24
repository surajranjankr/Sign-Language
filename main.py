import cv2
import mediapipe as mp
import numpy as np
import time
from src.engine import SignInterpreter
from src.utils import extract_landmarks

# 1. Initialize AI and UI components
interpreter = SignInterpreter() # This now uses your TFLite model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 2. Sentence State Variables
sentence = []
predictions = []
THRESHOLD = 0.90     # Only accept high-confidence predictions
STABILITY_FRAMES = 8  # How many frames a sign must be held to be "real"

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(model_complexity=1, min_detection_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Image Processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        # --- LOGIC GATE: Only predict if hands are visible ---
        if results.left_hand_landmarks or results.right_hand_landmarks:
            landmarks = extract_landmarks(results)
            idx, confidence = interpreter.predict(landmarks)
            
            if idx is not None and confidence > THRESHOLD:
                word = interpreter.label_map[idx]
                predictions.append(word)
                
                # --- STABILITY CHECK: Consensus Rule ---
                # Only add if the last 8 frames were the SAME word
                recent_preds = predictions[-STABILITY_FRAMES:]
                if len(recent_preds) == STABILITY_FRAMES and len(set(recent_preds)) == 1:
                    if not sentence or word != sentence[-1]:
                        sentence.append(word)
        else:
            # If hands disappear, clear the temporary prediction buffer
            # This forces the model to "restart" when you sign again
            predictions = []

        # Keeping sentence bar at a reasonable length
        if len(sentence) > 5:
            sentence = sentence[-5:]

        # --- THE "PRESENTATION-READY" UI ---
        # Draw a clean black bar for the text
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (10, 10, 10), -1)
        
        # Overlay the sentence
        display_text = " ".join(sentence).upper()
        cv2.putText(frame, display_text, (20, 35), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the result
        cv2.imshow('Sign Language Interpreter v2.0', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
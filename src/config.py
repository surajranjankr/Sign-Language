# Model dimensions based on the 126-feature optimization
SEQUENCE_LENGTH = 30 
NUM_FEATURES = 126
THRESHOLD = 0.8  # Minimum confidence to accept a word

# Paths
MODEL_PATH = 'models/sign_language_model.tflite'
LABEL_MAP_PATH = 'models/label_map.json'
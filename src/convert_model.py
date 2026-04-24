import tensorflow as tf

# 1. Load your existing model
# We use compile=False because we only need the architecture for conversion
model = tf.keras.models.load_model('models/sign_language_crnn.h5', compile=False)

# 2. Setup the Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# A. Allow the model to use complex TensorFlow ops that don't have native TFLite versions
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # Standard TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS   # Use TF ops for the complex LSTM parts
]

# B. Disable the specific lowering pass that caused the 'tf.TensorListReserve' error
converter._experimental_lower_tensor_list_ops = False

# C. Optimization (Standard)
converter.optimizations = [tf.lite.Optimize.DEFAULT]



# 3. Convert and Save
try:
    tflite_model = converter.convert()
    with open('models/sign_language_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Optimization complete: models/sign_language_model.tflite created.")
except Exception as e:
    print(f"Conversion failed: {e}")
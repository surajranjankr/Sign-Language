# Sign-Language Interpreter Model
A real-time, landmark-based sign language recognition system that translates dynamic hand gestures into full sentences. This project leverages a Hybrid CRNN (1D-CNN + Bi-LSTM) architecture and MediaPipe Holistic to achieve high-accuracy interpretation.



# Key Features
- Landmark Extraction: Uses MediaPipe to process only $(x, y, z)$ coordinates, reducing data noise by 99% compared to raw video.

- Wrist-Centering Normalization: Mathematically centers gestures to the wrist origin, making the model "Person-Blind" and invariant to camera distance.

- Real-Time Sentence Builder: Implements a stability-based consensus rule to transition from isolated signs to readable sentences.

- Edge-Optimized: Model converted to TFLite with Flex delegates, achieving 30+ FPS on standard CPUs.

- Modular Architecture: Clean separation of concerns between data utility and the inference engine.



# Tech Stack
Language: Python 3.12+

Deep Learning: TensorFlow 2.x, Keras

Computer Vision: MediaPipe Holistic, OpenCV

Optimization: TFLite (TensorFlow Lite)

Data Management: NumPy, JSON



# The Architecture
The project utilizes a CRNN (Convolutional Recurrent Neural Network) to decouple spatial and temporal features:

1D-CNN Layers: Act as "eyes" to identify the static geometry and shape of the fingers.

Bi-LSTM Layers: Act as "memory" to interpret the motion flow over a 30-frame sliding window.



# The Team
Suraj Ranjan Kumar: Webcam integration & TFLite model optimization.

Rupesh Kumar: Google ASL model training & architecture research.

Ravi Ranjan Bharti: Google ASL data preprocessing & normalization logic.

Dharmendra Yadav: Initial Bi-LSTM experimentation & WLASL hyperparameter tuning.

Mohit: WLASL data preprocessing & augmentation strategies.

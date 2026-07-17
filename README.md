# Final SignLink

Final SignLink is a webcam-based American Sign Language (ASL) recognition app that predicts alphabet signs, builds a sentence on screen, and speaks the confirmed text using text-to-speech.

## Features
- Real-time webcam capture
- Hand-region extraction and preprocessing
- ASL alphabet prediction with confidence filtering
- Prediction stabilization to reduce flicker
- Sentence building with add/delete/clear/space controls
- Text-to-speech output for confirmed text
- Graceful handling for missing camera, missing model, and invalid inputs

## Project Structure
- main.py: application entry point
- camera.py: webcam helpers
- predictor.py: prediction logic and stabilization
- train_model.py: training pipeline and model export
- dataset.py: dataset preparation and loading
- text_to_speech.py: speech synthesis
- models/: trained Keras model and class names
- dataset/: prepared dataset images

## Installation
1. Create a virtual environment if desired.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup
The app will create a lightweight generated dataset automatically if no external dataset is available. You can also point the dataset pipeline to a custom URL by editing the training logic.

## Training
Run:
```bash
python train_model.py
```

## Running the Application
Run:
```bash
python main.py
```

## Controls
- Space: add the current stable prediction to the sentence
- Backspace: delete the last letter
- C: clear the sentence
- S: add a space between words
- Q: quit the app

## Troubleshooting
- Camera not detected: verify your webcam is connected and available to Python.
- Model missing: run the training script first.
- TTS unavailable: install pyttsx3 and ensure a system voice is available.

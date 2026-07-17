import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from camera import open_camera, read_frame, release_camera, safe_resize
from dataset import PREPARED_DATASET_DIR, get_dataset_summary
from predictor import ASLPredictor
from text_to_speech import TextToSpeech
from train_model import train_model_if_needed

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "asl_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"


def load_class_names():
    if CLASS_NAMES_PATH.exists():
        with CLASS_NAMES_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return []


def run_demo(predictor, speech):
    print("Starting demo mode with sample images...")
    demo_images = []
    if PREPARED_DATASET_DIR.exists():
        for class_dir in sorted(PREPARED_DATASET_DIR.iterdir()):
            if not class_dir.is_dir():
                continue
            image_paths = sorted(class_dir.glob("*.png"))
            if image_paths:
                demo_images.append((class_dir.name, image_paths[0]))

    if not demo_images:
        print("No demo images were found. Please ensure the dataset exists.")
        return

    for label, image_path in demo_images[:10]:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        prediction, confidence, _ = predictor.predict_frame(frame)
        if prediction:
            print(f"Demo -> {label}: {prediction} ({confidence:.2f})")
            speech.speak(prediction)
        time.sleep(0.3)


def run_app(camera_index=0, demo_mode=False):
    print("Starting SignLink camera application...")

    try:
        dataset_summary = get_dataset_summary()
        print("Dataset summary:", dataset_summary)
    except Exception as exc:
        print(f"Dataset check warning: {exc}")

    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        print("Model not found. Training a new one...")
        train_model_if_needed(force=True)

    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError("Training did not produce a usable model. Please check the logs.")

    predictor = ASLPredictor(model_path=str(MODEL_PATH), class_names_path=str(CLASS_NAMES_PATH))
    speech = TextToSpeech()

    if demo_mode:
        run_demo(predictor, speech)
        return

    capture = open_camera(camera_index)
    if capture is None:
        raise RuntimeError("Unable to access the webcam. Please verify it is connected.")

    word_letters = []
    sentence_words = []
    stable_prediction = ""
    stable_confidence = 0.0
    last_spoken_prediction = ""
    last_sentence = ""
    last_frame_time = time.time()
    fps = 0.0
    status_message = "Waiting for a hand..."

    try:
        while True:
            success, frame = read_frame(capture)
            if not success or frame is None:
                status_message = "Unable to read frames from the camera."
                break

            frame = cv2.flip(frame, 1)
            preview_frame = frame.copy()
            prediction, confidence, roi = predictor.predict_frame(frame)

            if prediction and confidence >= predictor.confidence_threshold:
                stable_prediction = prediction
                stable_confidence = confidence
                status_message = f"Stable prediction: {prediction}"
                if stable_prediction != last_spoken_prediction:
                    speech.speak(stable_prediction)
                    last_spoken_prediction = stable_prediction
            elif prediction:
                status_message = f"Low confidence: {prediction} ({confidence:.2f})"
            else:
                stable_prediction = ""
                stable_confidence = 0.0
                status_message = "No hand detected"

            if stable_prediction:
                cv2.putText(
                    preview_frame,
                    f"Prediction: {stable_prediction} ({stable_confidence:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            if roi is not None:
                roi_resized = safe_resize(roi, (180, 180))
                cv2.rectangle(preview_frame, (10, 60), (190, 240), (255, 255, 255), 2)
                preview_frame[70:250, 20:200] = roi_resized
                cv2.putText(
                    preview_frame,
                    "Hand ROI",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed > 0:
                fps = 1.0 / elapsed
            last_frame_time = current_time

            current_word = "".join(word_letters)
            current_sentence = " ".join(sentence_words + ([current_word] if current_word else []))
            cv2.putText(preview_frame, f"FPS: {fps:.1f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(preview_frame, f"Word: {current_word}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(preview_frame, f"Sentence: {current_sentence}", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(preview_frame, f"Status: {status_message}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(preview_frame, "Keys: space add letter | backspace delete | c clear | s add space | q quit", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("SignLink ASL Camera", preview_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and stable_prediction:
                word_letters.append(stable_prediction)
                current_sentence = " ".join(sentence_words + (["".join(word_letters)] if word_letters else []))
                if current_sentence != last_sentence:
                    speech.speak(current_sentence)
                    last_sentence = current_sentence
            elif key == 8 or key == 127:
                if word_letters:
                    word_letters.pop()
                elif sentence_words:
                    sentence_words.pop()
            elif key == ord("c"):
                word_letters.clear()
                sentence_words.clear()
            elif key == ord("s"):
                if word_letters:
                    sentence_words.append("".join(word_letters))
                    word_letters.clear()
                    speech.speak(" ".join(sentence_words))

        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("Keyboard interruption received. Exiting gracefully.")
    except Exception as exc:
        print(f"Runtime error: {exc}")
        cv2.destroyAllWindows()
    finally:
        release_camera(capture)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignLink ASL recognition")
    parser.add_argument("--demo", action="store_true", help="Run a demo using sample images")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index to use")
    args = parser.parse_args()

    try:
        run_app(camera_index=args.camera_index, demo_mode=args.demo)
    except SystemExit as exc:
        raise SystemExit(exc)
    except Exception as exc:
        print(f"Application failed: {exc}")
        sys.exit(1)

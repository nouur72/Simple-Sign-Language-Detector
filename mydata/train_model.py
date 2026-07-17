import argparse
import json
from pathlib import Path

from dataset import load_dataset

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "asl_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"


def train_model_if_needed(force=False, dataset_url=None):
    if MODEL_PATH.exists() and CLASS_NAMES_PATH.exists() and not force:
        print("Model already exists; skipping training.")
        return MODEL_PATH

    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError("TensorFlow is required for training. Install dependencies first.") from exc

    print("Preparing dataset...")
    dataset = load_dataset(dataset_url=dataset_url)
    train_images = dataset["train_images"]
    train_labels = dataset["train_labels"]
    val_images = dataset["val_images"]
    val_labels = dataset["val_labels"]
    class_names = dataset["class_names"]

    print("Dataset summary:")
    print(json.dumps(dataset["summary"], indent=2))

    input_shape = (96, 96, 3)
    num_classes = len(class_names)

    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
    ]

    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=6,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    with CLASS_NAMES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(class_names, handle)

    print("Training complete.")
    print("Accuracy:", history.history["accuracy"][-1])
    print("Validation Accuracy:", history.history["val_accuracy"][-1])
    print("Loss:", history.history["loss"][-1])
    print("Validation Loss:", history.history["val_loss"][-1])

    return MODEL_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SignLink ASL model")
    parser.add_argument("--dataset-url", type=str, default=None, help="GitHub or archive URL for the ASL dataset")
    args = parser.parse_args()
    train_model_if_needed(force=True, dataset_url=args.dataset_url)

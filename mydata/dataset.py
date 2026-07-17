import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from random import Random

import numpy as np
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DATASET_DIR = DATASET_DIR / "raw"
GENERATED_DATASET_DIR = DATASET_DIR / "generated"
PREPARED_DATASET_DIR = DATASET_DIR / "prepared"
IMAGE_SIZE = (96, 96)
CLASS_NAMES = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def ensure_dataset_ready(dataset_url=None):
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    PREPARED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if (DATASET_DIR / "ready").exists() and PREPARED_DATASET_DIR.exists():
        return PREPARED_DATASET_DIR

    if dataset_url:
        try:
            print(f"Downloading dataset from: {dataset_url}")
            download_dataset(dataset_url)
        except Exception as exc:
            print(f"Dataset download failed: {exc}. Falling back to generated data.")

    prepared = prepare_from_source(PREPARED_DATASET_DIR)
    if not prepared:
        print("No usable external dataset detected. Creating generated fallback data.")
        create_generated_dataset()
        prepare_from_source(PREPARED_DATASET_DIR, source_root=GENERATED_DATASET_DIR)

    missing_classes = [name for name in CLASS_NAMES if not (PREPARED_DATASET_DIR / name).exists()]
    if missing_classes:
        print(f"Missing classes after preparation: {', '.join(missing_classes)}")

    (DATASET_DIR / "ready").touch()
    return PREPARED_DATASET_DIR


def prepare_from_source(output_root, source_root=None):
    if source_root is None:
        source_root = find_dataset_source_root()
    if source_root is None or not source_root.exists():
        return False

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    available_classes = []
    for class_name in CLASS_NAMES:
        class_dir = find_class_directory(source_root, class_name)
        if class_dir is not None:
            available_classes.append(class_name)

    if len(available_classes) < 26:
        print(f"Only {len(available_classes)} classes were found in the provided dataset; falling back to generated data if necessary.")

    seen_hashes = set()
    for class_name in CLASS_NAMES:
        class_dir = find_class_directory(source_root, class_name)
        if class_dir is None:
            continue

        target_dir = output_root / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(image_path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image_bytes = image.tobytes()
                    digest = hashlib.sha256(image_bytes).hexdigest()
                    if digest in seen_hashes:
                        continue
                    seen_hashes.add(digest)
                    image = ImageOps.pad(image, IMAGE_SIZE)
                    image = np.array(image, dtype=np.float32) / 255.0
                    if not np.isfinite(image).all():
                        continue
                    output_path = target_dir / f"{class_name}_{count:03d}.png"
                    Image.fromarray((image * 255.0).astype(np.uint8)).save(output_path)
                    count += 1
            except Exception:
                continue

    return any((output_root / class_name).exists() and any((output_root / class_name).iterdir()) for class_name in CLASS_NAMES)


def find_dataset_source_root():
    if RAW_DATASET_DIR.exists():
        for path in sorted(RAW_DATASET_DIR.rglob("*")):
            if path.is_dir() and any(path.iterdir()):
                return path
    return None


def find_class_directory(root_dir, class_name):
    if root_dir is None or not root_dir.exists():
        return None
    for candidate in root_dir.rglob(class_name):
        if candidate.is_dir():
            return candidate
    return None


def download_dataset(dataset_url):
    if not dataset_url:
        raise ValueError("Dataset URL is required")

    archive_path = DATASET_DIR / "dataset.zip"
    resolved_url = dataset_url
    if "github.com" in resolved_url and "/archive/" not in resolved_url and "/releases/" not in resolved_url:
        resolved_url = resolved_url.rstrip("/") + "/archive/refs/heads/main.zip"
    urllib.request.urlretrieve(resolved_url, archive_path)
    shutil.unpack_archive(archive_path, RAW_DATASET_DIR)
    archive_path.unlink(missing_ok=True)


def create_generated_dataset(samples_per_class=60):
    generated_dir = GENERATED_DATASET_DIR
    if generated_dir.exists():
        shutil.rmtree(generated_dir)
    generated_dir.mkdir(parents=True, exist_ok=True)

    rng = Random(42)
    for class_name in CLASS_NAMES:
        class_dir = generated_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(samples_per_class):
            image = Image.new("RGB", IMAGE_SIZE, (245, 245, 245))
            draw_shape(image, class_name, index, rng)
            image = ImageOps.grayscale(ImageOps.colorize(image.convert("L"), black="black", white="white"))
            image = image.convert("RGB")
            image.save(class_dir / f"{class_name}_{index:02d}.png")

    for class_name in CLASS_NAMES:
        source_dir = generated_dir / class_name
        if not any(source_dir.iterdir()):
            raise RuntimeError(f"No generated images for {class_name}")


def draw_shape(image, class_name, index, rng):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    width, height = image.size
    center_x = width // 2
    center_y = height // 2
    stroke = 8

    draw.ellipse((center_x - 18, center_y - 18, center_x + 18, center_y + 18), fill=(210, 180, 120))

    offset = (index % 4) * 4
    for step in range(4):
        x1 = center_x - 24 + step * 8 + offset
        y1 = center_y - 20 + step * 4
        x2 = center_x + 8 + step * 4 + offset
        y2 = center_y + 20 - step * 6
        draw.line((x1, y1, x2, y2), fill=(60, 60, 60), width=stroke)

    letter_index = ord(class_name) - ord("A")
    if letter_index % 2 == 0:
        draw.rectangle((center_x - 24, center_y - 24, center_x + 24, center_y + 24), outline=(80, 80, 80), width=3)
    else:
        draw.ellipse((center_x - 24, center_y - 24, center_x + 24, center_y + 24), outline=(80, 80, 80), width=3)

    if class_name in {"A", "B", "C", "D", "E"}:
        draw.line((center_x - 20, center_y - 24, center_x - 20, center_y + 24), fill=(40, 40, 40), width=3)


def load_dataset(image_size=(96, 96), split=(0.7, 0.15, 0.15), dataset_url=None):
    ensure_dataset_ready(dataset_url=dataset_url)
    image_root = PREPARED_DATASET_DIR if PREPARED_DATASET_DIR.exists() else GENERATED_DATASET_DIR
    class_names = [path.name for path in sorted(image_root.iterdir()) if path.is_dir()]
    if not class_names:
        raise RuntimeError("No dataset folders were found")

    images = []
    labels = []
    per_class_counts = {}
    for class_idx, class_name in enumerate(class_names):
        class_dir = image_root / class_name
        per_class_counts[class_name] = 0
        for image_path in sorted(class_dir.glob("*.png")):
            try:
                with Image.open(image_path) as image:
                    image = ImageOps.pad(image.convert("RGB"), image_size)
                    image = np.array(image, dtype=np.float32) / 255.0
                    images.append(image)
                    labels.append(class_idx)
                    per_class_counts[class_name] += 1
            except Exception:
                continue

    if len(images) == 0:
        raise RuntimeError("No images were found in the dataset")

    rng = Random(42)
    combined = list(zip(images, labels))
    rng.shuffle(combined)
    images, labels = zip(*combined)

    total_images = len(images)
    train_count = int(total_images * split[0])
    val_count = int(total_images * split[1])
    test_count = total_images - train_count - val_count

    train_images = np.stack(images[:train_count])
    train_labels = np.array(labels[:train_count], dtype=np.int32)
    val_images = np.stack(images[train_count : train_count + val_count])
    val_labels = np.array(labels[train_count : train_count + val_count], dtype=np.int32)
    test_images = np.stack(images[train_count + val_count :])
    test_labels = np.array(labels[train_count + val_count :], dtype=np.int32)

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
        "test_images": test_images,
        "test_labels": test_labels,
        "class_names": class_names,
        "summary": {
            "classes": len(class_names),
            "per_class": per_class_counts,
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images),
            "total": total_images,
        },
    }


def get_dataset_summary():
    data = load_dataset()
    return data["summary"]

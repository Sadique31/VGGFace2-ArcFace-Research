import csv
import time
import numpy as np
from pathlib import Path
from deepface import DeepFace

# =========================
# Paths
# =========================
PROJECT = Path("/mnt/c/Users/ansar/Desktop/VGGFace2_ArcFace_Project")

IMAGE_DIR = PROJECT / "data/aligned/train"
OUTPUT_DIR = PROJECT / "results/arcface_embeddings"
BATCH_DIR = OUTPUT_DIR / "batches"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
PATHS_FILE = OUTPUT_DIR / "image_paths.csv"

# =========================
# Configuration
# =========================
BATCH_SIZE = 1000
EMBEDDING_DIM = 512

# =========================
# Collect images
# =========================
images = sorted(IMAGE_DIR.rglob("*.jpg"))

print("=" * 60)
print("DeepFace ArcFace Embedding Extraction")
print("=" * 60)
print(f"Total images: {len(images)}")
print(f"Batch size: {BATCH_SIZE}")
print()

# =========================
# Process batches
# =========================
total_start = time.time()

for start in range(0, len(images), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(images))
    batch_number = start // BATCH_SIZE

    batch_file = BATCH_DIR / f"batch_{batch_number:04d}.npy"

    # Skip completed batch
    if batch_file.exists():

        print(
            f"[SKIP] Batch {batch_number:04d} "
            f"({start}:{end}) already exists."
        )

        continue

    print(
        f"\n[START] Batch {batch_number:04d} "
        f"({start}:{end})"
    )

    batch_start = time.time()
    batch_embeddings = []

    for i in range(start, end):

        image_path = images[i]

        try:

            result = DeepFace.represent(
                img_path=str(image_path),
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False
            )

            embedding = np.asarray(
                result[0]["embedding"],
                dtype=np.float32
            )

            if embedding.shape != (EMBEDDING_DIM,):
                raise ValueError(
                    f"Unexpected embedding shape: {embedding.shape}"
                )

            batch_embeddings.append(embedding)

        except Exception as e:

            print(f"\n[ERROR] {image_path}")
            print(e)

            batch_embeddings.append(
                np.full(
                    EMBEDDING_DIM,
                    np.nan,
                    dtype=np.float32
                )
            )

        # Progress every 100 images
        if (i - start + 1) % 100 == 0:
            print(
                f"  Progress: {i - start + 1}/{end - start}",
                flush=True
            )

    batch_embeddings = np.asarray(
        batch_embeddings,
        dtype=np.float32
    )

    # Save only after the COMPLETE batch finishes
    np.save(batch_file, batch_embeddings)

    elapsed = time.time() - batch_start
    speed = (end - start) / elapsed

    print(
        f"[DONE] Batch {batch_number:04d} | "
        f"{end - start} images | "
        f"{elapsed:.1f}s | "
        f"{speed:.2f} img/s"
    )

# =========================
# Combine batches
# =========================
print("\n" + "=" * 60)
print("Combining batches")
print("=" * 60)

num_images = len(images)

# Create final .npy file without loading everything into RAM
final_embeddings = np.lib.format.open_memmap(
    EMBEDDINGS_FILE,
    mode="w+",
    dtype=np.float32,
    shape=(num_images, EMBEDDING_DIM)
)

position = 0

for start in range(0, num_images, BATCH_SIZE):

    end = min(start + BATCH_SIZE, num_images)
    batch_number = start // BATCH_SIZE

    batch_file = BATCH_DIR / f"batch_{batch_number:04d}.npy"

    batch = np.load(batch_file)

    final_embeddings[position:end] = batch

    position = end

    print(
        f"Combined: {position}/{num_images}"
    )

final_embeddings.flush()
del final_embeddings

# =========================
# Save image paths
# =========================
with open(PATHS_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow(["index", "image_path"])

    for i, path in enumerate(images):

        writer.writerow([
            i,
            str(path)
        ])

# =========================
# Final verification
# =========================
check = np.load(EMBEDDINGS_FILE, mmap_mode="r")

print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)

print(f"Images:      {num_images}")
print(f"Embeddings:  {check.shape}")
print(f"Data type:   {check.dtype}")
print(f"Output:      {EMBEDDINGS_FILE}")
print(f"Paths:       {PATHS_FILE}")

total_elapsed = time.time() - total_start

print(
    f"Total time:  {total_elapsed / 3600:.2f} hours"
)

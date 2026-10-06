import csv
import tarfile
from pathlib import Path

ARCHIVE = Path("data/raw/vggface2_train.tar.gz")
MANIFEST = Path("metadata/vggface2_selected_129465.csv")
OUTPUT = Path("data/selected")

OUTPUT.mkdir(parents=True, exist_ok=True)

# Read selected image paths
selected = set()

with MANIFEST.open("r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        selected.add("train/" + row["NAME_ID"] + ".jpg")

print(f"Selected images: {len(selected)}")
print("Opening archive...")

extracted = 0
skipped = 0

with tarfile.open(ARCHIVE, "r:gz") as tar:
    for member in tar:
        if not member.isfile():
            continue

        filename = Path(member.name).as_posix()

        if filename in selected:
            destination = OUTPUT / filename
            destination.parent.mkdir(parents=True, exist_ok=True)

            if destination.exists():
                skipped += 1
                continue

            tar.extract(member, OUTPUT)
            extracted += 1

            if extracted % 1000 == 0:
                print(f"Extracted: {extracted}")

print("\nExtraction complete.")
print(f"Newly extracted: {extracted}")
print(f"Already existed: {skipped}")
print(f"Total available: {extracted + skipped}")
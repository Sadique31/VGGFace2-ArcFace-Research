import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

PROJECT = "/mnt/c/Users/ansar/Desktop/VGGFace2_ArcFace_Project"

INPUT_GENUINE = os.path.join(
    PROJECT, "results/hard_pair_analysis/hard_genuine_pairs_top100.csv"
)

INPUT_IMPOSTOR = os.path.join(
    PROJECT, "results/hard_pair_analysis/hard_impostor_pairs_top100.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT, "results/hard_pair_analysis/contact_sheets"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_contact_sheet(csv_path, output_path, title):
    df = pd.read_csv(csv_path)

    # 5 pairs per row, 20 rows = 100 pairs
    cols = 5
    pair_width = 420
    pair_height = 230
    title_height = 70

    rows = math.ceil(len(df) / cols)

    sheet = Image.new(
        "RGB",
        (cols * pair_width, title_height + rows * pair_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text(
        (10, 10),
        title,
        fill="black",
        font=font
    )

    for i, row in df.iterrows():

        img1_path = row["image1"]
        img2_path = row["image2"]

        # Convert WSL paths if necessary
        if not os.path.exists(img1_path):
            img1_path = img1_path.replace(
                "/mnt/c/Users/ansar",
                "/mnt/c/Users/ansar"
            )

        if not os.path.exists(img2_path):
            img2_path = img2_path.replace(
                "/mnt/c/Users/ansar",
                "/mnt/c/Users/ansar"
            )

        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            # Resize each face image
            img1.thumbnail((195, 175))
            img2.thumbnail((195, 175))

            col = i % cols
            row_num = i // cols

            x = col * pair_width
            y = title_height + row_num * pair_height

            # Center images
            x1 = x + 5
            x2 = x + 210

            y_img = y + 5

            sheet.paste(img1, (x1, y_img))
            sheet.paste(img2, (x2, y_img))

            similarity = float(row["cosine_similarity"])

            label = f"#{i+1}  Similarity: {similarity:.4f}"

            draw.text(
                (x + 5, y + 185),
                label,
                fill="black",
                font=small_font
            )

        except Exception as e:
            print(f"Could not load pair {i+1}: {e}")

    sheet.save(output_path, quality=95)

    print(f"Saved: {output_path}")


print("Creating hard genuine contact sheet...")

create_contact_sheet(
    INPUT_GENUINE,
    os.path.join(OUTPUT_DIR, "hard_genuine_top100.jpg"),
    "Top 100 Hardest Genuine Pairs — Same Identity"
)

print("Creating hard impostor contact sheet...")

create_contact_sheet(
    INPUT_IMPOSTOR,
    os.path.join(OUTPUT_DIR, "hard_impostor_top100.jpg"),
    "Top 100 Hardest Impostor Pairs — Different Identities"
)

print("\nDone.")

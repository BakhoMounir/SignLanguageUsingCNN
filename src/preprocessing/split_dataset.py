import os
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIG
# ==========================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SOURCE_DIR = Path("data/raw/asl_alphabet_train")
OUTPUT_DIR = Path("data/processed")

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR   = OUTPUT_DIR / "val"
TEST_DIR  = OUTPUT_DIR / "test"

# ==========================================
# SAMPLING CONFIG (KEY FIX)
# ==========================================

SAMPLES_PER_CLASS = 200   # same idea as YOLO script

TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

# ==========================================
# CREATE OUTPUT DIRECTORIES
# ==========================================

for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    split_dir.mkdir(parents=True, exist_ok=True)

# ==========================================
# GET CLASSES
# ==========================================

classes = sorted([d.name for d in SOURCE_DIR.iterdir() if d.is_dir()])
print(f"Found {len(classes)} classes")

# ==========================================
# PROCESS EACH CLASS
# ==========================================

for class_name in classes:

    class_path = SOURCE_DIR / class_name

    images = sorted(list(class_path.glob("*")))

    if len(images) == 0:
        continue

    # ==========================================
    # STEP 1: BALANCED SAMPLING (IMPORTANT)
    # ==========================================

    if len(images) > SAMPLES_PER_CLASS:
        images = random.sample(images, SAMPLES_PER_CLASS)

    # ==========================================
    # STEP 2: SPLIT (TRAIN / TEMP)
    # ==========================================

    train_images, temp_images = train_test_split(
        images,
        test_size=(1 - TRAIN_SPLIT),
        random_state=RANDOM_SEED
    )

    # ==========================================
    # STEP 3: SPLIT TEMP → VAL / TEST
    # ==========================================

    val_images, test_images = train_test_split(
        temp_images,
        test_size=TEST_SPLIT / (TEST_SPLIT + VAL_SPLIT),
        random_state=RANDOM_SEED
    )

    # ==========================================
    # CREATE CLASS FOLDERS
    # ==========================================

    for split_name, split_dir in [
        ("train", TRAIN_DIR),
        ("val", VAL_DIR),
        ("test", TEST_DIR)
    ]:
        (split_dir / class_name).mkdir(parents=True, exist_ok=True)

    # ==========================================
    # COPY FUNCTION
    # ==========================================

    def copy_images(image_list, destination_folder):
        for img_path in image_list:
            shutil.copy(img_path, destination_folder / img_path.name)

    copy_images(train_images, TRAIN_DIR / class_name)
    copy_images(val_images, VAL_DIR / class_name)
    copy_images(test_images, TEST_DIR / class_name)

    # ==========================================
    # LOGGING
    # ==========================================

    print(
        f"{class_name}: "
        f"train={len(train_images)} | "
        f"val={len(val_images)} | "
        f"test={len(test_images)}"
    )

print("\nBalanced dataset splitting completed successfully.")
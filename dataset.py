import os
import shutil
import random

RAW_DIR = "/home/divya/Desktop/Dataset/raw_train"       
OUTPUT_DIR = "/home/divya/Desktop/Dataset"
TEST_SPLIT = 0.2                    
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_and_save(files, train_dir, test_dir, split_ratio):
    random.shuffle(files)
    test_count = int(len(files) * split_ratio)

    test_files = files[:test_count]
    train_files = files[test_count:]

    for f in train_files:
        shutil.copy(f, train_dir)
    for f in test_files:
        shutil.copy(f, test_dir)

    print(f"{len(train_files)} → train, {len(test_files)} → test saved into {os.path.basename(train_dir)}")

def main():
    print("\n----- Preparing Dataset -----\n")

    cats = []
    dogs = []

    for file in os.listdir(RAW_DIR):
        if not os.path.isfile(os.path.join(RAW_DIR, file)):
            continue

        file_path = os.path.join(RAW_DIR, file.lower())

        if "cat" in file.lower():
            cats.append(os.path.join(RAW_DIR, file))
        elif "dog" in file.lower():
            dogs.append(os.path.join(RAW_DIR, file))

    print(f"Found {len(cats)} cat images")
    print(f"Found {len(dogs)} dog images\n")

    train_cat_dir = os.path.join(OUTPUT_DIR, "train", "cats")
    train_dog_dir = os.path.join(OUTPUT_DIR, "train", "dogs")
    test_cat_dir  = os.path.join(OUTPUT_DIR, "test", "cats")
    test_dog_dir  = os.path.join(OUTPUT_DIR, "test", "dogs")

    for d in [train_cat_dir, train_dog_dir, test_cat_dir, test_dog_dir]:
        make_dir(d)

    
    split_and_save(cats, train_cat_dir, test_cat_dir, TEST_SPLIT)
    split_and_save(dogs, train_dog_dir, test_dog_dir, TEST_SPLIT)

    print("\n----- Dataset Ready! -----")
    print(f"Saved to: {OUTPUT_DIR}\n")

if __name__ == "__main__":
    main()

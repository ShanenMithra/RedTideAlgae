import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

# Set dataset path
dataset_path = "C:/Users/shanen/Desktop/Dataset"

# Image Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=15,  # Reduce rotation to avoid excessive tilting
    width_shift_range=0.1,  # Small shift to prevent cutting off too much
    height_shift_range=0.1,  # Same as width shift
    shear_range=0.1,  # Reduce shear to keep structure intact
    zoom_range=0.15,  # Slight zoom to add variety without major distortion
    horizontal_flip=True,  # Flip is safe for most datasets
    fill_mode="nearest"
)

# Loop through train/test/validation folders
for dataset_type in ["train", "test", "validation"]:
    dataset_folder = os.path.join(dataset_path, dataset_type)
    print(f"\nProcessing {dataset_type}...")  # Debugging Print

    for class_name in os.listdir(dataset_folder):
        class_path = os.path.join(dataset_folder, class_name)
        if not os.path.isdir(class_path):
            continue  # Skip if it's not a folder

        print(f" Processing Class: {class_name}")  # Debugging Print

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            # Check if the path is a file before trying to open
            if not os.path.isfile(img_path):
                continue

            print(f"Found image: {img_name}")  # Debugging Print

            try:
                img = load_img(img_path)  # Load image
                print(f" Loaded image: {img_name}")  # Debugging Print

                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                aug_iter = datagen.flow(img_array, batch_size=1)

                for i in range(3):  # Generate 3 augmented images per original image
                    aug_img = next(aug_iter)[0].astype("uint8")

                    aug_img_path = os.path.join(class_path, f"aug_{i+3}_{img_name}")
                    print(f" Saving: {aug_img_path}")  # Debugging Print

                    img_save = Image.fromarray(aug_img)
                    img_save.save(aug_img_path)

            except Exception as e:
                print(f"  Error processing {img_name}: {e}")  # Debugging Print

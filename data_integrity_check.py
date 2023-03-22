from PIL import Image
import os
from tqdm import tqdm

# Directory containing images
dir_path = "./data/images/test"

# Loop through all files in the directory
for filename in tqdm(os.listdir(dir_path)):
    try:
        # Attempt to open the image with Pillow
        img = Image.open(os.path.join(dir_path, filename))
        img.verify()
        img.close()
    except Exception as e:
        # If there is an exception, print the filename and the exception message
        print(f"Error opening file '{filename}': {e}")
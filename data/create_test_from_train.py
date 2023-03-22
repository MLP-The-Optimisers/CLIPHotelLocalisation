import os
import random
import shutil

# Define source and destination directories
src_dir = os.path.abspath("./images/train")
dst_dir = os.path.abspath("./images/test")

# Get a list of all files in the source directory
all_files = os.listdir(src_dir)

# Calculate the number of files to move (20% of the total)
num_files_to_move = int(len(all_files) * 0.2)

# Randomly select the files to move
files_to_move = random.sample(all_files, num_files_to_move)

# Move the selected files to the destination directory
for file_name in files_to_move:
    src_path = os.path.join(src_dir, file_name)
    dst_path = os.path.join(dst_dir, file_name)
    shutil.move(src_path, dst_path)

print(f"Moved {num_files_to_move} files from {src_dir} to {dst_dir}.")

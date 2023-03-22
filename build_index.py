from PIL import Image
import requests
import numpy as np
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import os
import glob
from rich.progress import track
import torch

from src.clip_index import CLIPIndex
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CLIPVisionModelWithProjection.from_pretrained("./clip_trained")
model.to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

index = CLIPIndex()
index.create_index("clip_index")

image_paths = glob.glob('data/images/train/*.jpg', recursive=True)

for image_path in track(image_paths, description="Processing..."):
    image_id = int(image_path.split('.')[0].split('/')[-1:][0])
    with Image.open(image_path) as image:
        try:
            inputs = processor(images=image, return_tensors="pt")
        except:
            print(str(image_id), " is corrupted\n")

        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
        image_embeds = outputs.image_embeds

        index.insert_cosine(image_embeds, np.array([image_id]))

index.save()

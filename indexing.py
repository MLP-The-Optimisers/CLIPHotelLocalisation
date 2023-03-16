from PIL import Image
import requests
import numpy as np
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import os
import glob
from rich.progress import track

from src.clip_index import CLIPIndex

model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

index = CLIPIndex()
index.create_index("clip_index")

image_paths = glob.glob('data/images/train/*.jpg', recursive=True)

for image_path in track(image_paths, description="Processing..."):
    image_id = int(image_path.split('.')[0].split('/')[-1:][0])
    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    image_embeds = outputs.image_embeds

    index.insert_cosine(image_embeds, np.array([image_id]))

index.save()
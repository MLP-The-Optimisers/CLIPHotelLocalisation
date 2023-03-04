from PIL import Image
import requests
import numpy as np
from transformers import AutoProcessor, CLIPVisionModelWithProjection

from src.clip_index import CLIPIndex

model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
image_embeds = outputs.image_embeds

index = CLIPIndex()
index.create_index("clip_index")
index.insert_cosine(image_embeds, np.array([232344]))
print(index.search_cosine(image_embeds, top_k=1))
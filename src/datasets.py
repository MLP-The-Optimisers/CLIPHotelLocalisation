from torch.utils.data import Dataset
from PIL import Image
import torch

class ImageTextDataset(Dataset):
    def __init__(self, image_paths, texts, processor, device):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor
        self.device = device
        self.label_to_id = {label: i for i, label in enumerate(sorted(set(texts)))}

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        text = self.texts[index]

        # Load the image and convert to a tensor
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values']
        pixel_values = pixel_values.squeeze(0)

        # Convert the text to a tensor
        max_length = self.processor.tokenizer.model_max_length
        text_tokens = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = text_tokens["input_ids"].squeeze(0)
        attention_mask = text_tokens["attention_mask"].squeeze(0)
        
        # Move to approproate device
        pixel_values = pixel_values.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        label = torch.tensor(self.label_to_id[text]).to(self.device)

        return (pixel_values, input_ids, attention_mask, label)

    def __len__(self):
        return len(self.image_paths)

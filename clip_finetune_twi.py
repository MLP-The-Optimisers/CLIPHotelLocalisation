import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from rich.progress import track
from rich import print
import pickle
import wandb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


run = wandb.init(
    # set the wandb project where this run will be logged
    project="MLP",
    
    # track hyperparameters and run metadata
    config={
        "learning_rate": 5e-5,
        "architecture": "CNN",
        "dataset": "LocalTest-600K-660K",
        "epochs": 10,
        "batch_size": 64
    }
)

def save_object(obj: any, repo: str, file: str):
    # Path constant to save the object
    PATH = f'{repo}/{file}.pkl'

    # Save as a pickle file
    with open(PATH, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object(repo: str, file: str):
    # Path constant to save the object
    PATH = f'{repo}/{file}.pkl'
    print("loading this pickle file: ", PATH)

    with open(PATH, 'rb') as f:
        print("opened pickle file will now load")
        return pickle.load(f)

# Define the dataset class for loading image-text pairs
class ImageTextDataset(Dataset):
    def __init__(self, image_paths, texts, processor):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor

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
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        return (pixel_values, input_ids, attention_mask)

    def __len__(self):
        return len(self.image_paths)

# Define the training function
def train_clip(epochs, batch_size, train_dataset, val_dataset, model, processor):
    # Create the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Define the loss function
    def compute_loss(logits, labels):
        return F.cross_entropy(logits, labels)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Train the model
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # Unpack the inputs and labels from the data loader
            pixel_values, input_ids, attention_mask = batch
            labels = torch.arange(len(images)).long().to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values, 
                return_loss=True)
            
            loss = outputs.loss
            wandb.log({"loss": loss, "it": i + epoch})
            loss.backward()
            optimizer.step()
        wandb.log({"epoch": epoch})

        # Evaluate the model on the validation set
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                # Unpack the inputs and labels from the data loader
                pixel_values, input_ids, attention_mask = batch

                # Forward pass
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    pixel_values=pixel_values)
                logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text
                logits = torch.cat([logits_per_image, logits_per_text])

                # Compute the accuracy
                predicted_labels = logits.argmax(dim=1)
                print(predicted_labels)
                total_correct += (predicted_labels == labels).sum().item()
                total_samples += len(labels)
                accuracy = total_correct / total_samples

                wandb.log({
                    "val": {
                        "acc": accuracy,
                        "epoch": epoch + 1/epochs
                    }
                })

                print(f"Epoch {epoch + 1}/{epochs}: validation accuracy = {accuracy}")

        torch.cuda.empty_cache()
        # Update the learning rate scheduler
        scheduler.step()

    return model


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model = model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load dataset DataFrames from Hotels50K
df_train = pd.read_csv("data/input/dataset/train_set.csv")
df_hotels = pd.read_csv("data/input/dataset/hotel_info.csv")
df_chains = pd.read_csv("data/input/dataset/chain_info.csv")

parse_img_id = lambda x: int(x.split('.')[0].split('/')[-1:][0])

# Build dataset split with chain labels
images = glob.glob("data/images/*/*.jpg")
print("read images: in format ", images[0])
labels = []
try:
    0/0
    labels = load_object('.', 'labels')
    images = load_object('.', 'images')
    print("this should have failed!")
except:
    for path in track(images, description="Preparing dataset..."):
        img_id = parse_img_id(path)

        hotel_id = df_train.loc[df_train['image_id'] == img_id]['hotel_id'].iloc[0]
        chain_id = df_hotels.loc[df_hotels['hotel_id'] == hotel_id]['chain_id'].iloc[0]
        chain_name = df_chains.loc[df_chains['chain_id'] == chain_id]['chain_name'].iloc[0]

        labels.append(chain_name)

    # Cache as Pickle files to be loaded for another run
    save_object(images, '.', 'images')
    save_object(labels, '.', 'labels')

# Split dataset
images_train, images_val, labels_train, labels_val = train_test_split(images, labels, test_size=0.33, random_state=42)
print(len(images_train))
print(images_train[0])
train_dataset = ImageTextDataset(images_train, labels_train, processor)
val_dataset = ImageTextDataset(images_val, labels_val, processor)

# Start training
epochs = 10
batch_size = 128

model = train_clip(epochs, batch_size, train_dataset, val_dataset, model, processor)
torch.save(model.state_dict(), f'saved_models_e{epochs}_bz{batch_size}_lr{5e-5}/model.pth')
artifact = wandb.Artifact('model', type='model')
artifact.add_file(f'saved_models_e{epochs}_bz{batch_size}_lr{5e-5}/model.pth')
run.log_artifact(artifact)

wandb.finish()
# External imports
import torch
import os
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from rich.progress import track
from rich import print
from rich.console import Console
import wandb

# Local imports
from src.utils.pickle_handler import save_object, load_object
from src.utils.train_argparser import build_arg_parser
from src.datasets import ImageTextDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Constants
PICKLE_REPO = './saved/pickles'
MODEL_REPO = './saved/models'
INDEX_REPO = './saved/index'

# Dataclass defining the experiment hparams
@dataclass
class ExperimentConfig:
    epochs: int
    batch_size: int
    lr: float


# Method for ensureing that folders for outputs are there
def build_expected_folders():
    if not os.path.isdir(os.path.abspath(PICKLE_REPO)):
        os.makedirs(os.path.abspath(PICKLE_REPO))
    
    if not os.path.isdir(os.path.abspath(MODEL_REPO)):
        os.makedirs(os.path.abspath(MODEL_REPO))

    if not os.path.isdir(os.path.abspath(INDEX_REPO)):
        os.makedirs(os.path.abspath(INDEX_REPO))


#===========================================================================
#                     TRAIN AND VALIDATION PROCEDURES
#===========================================================================

def fn_train(
        epoch: int,
        model: CLIPModel,
        optimizer: any,
        train_loader: DataLoader
):
    
    train_total, train_loss = 0, 0

    for batch in train_loader:
        optimizer.zero_grad()
        train_total += 1 
        # Unpack the inputs and labels from the data loader
        pixel_values, input_ids, attention_mask = batch

        # Forward pass
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values, 
            return_loss=True)
        
        loss = outputs.loss
        train_loss += loss

        loss.backward()
        optimizer.step()
    
    mean_batch_loss = train_loss / train_total
    wandb.log({"train": {"loss": mean_batch_loss}, "epoch": epoch})


def fn_val(
        model: CLIPModel,
        val_loader: DataLoader
):

    val_total, val_loss = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            val_total += 1
            # Unpack the inputs and labels from the data loader
            pixel_values, input_ids, attention_mask = batch

            # Forward pass
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values, 
                return_loss=True)
            
            loss = outputs.loss
            val_loss += loss
    
    mean_val_loss = val_loss / val_total
    wandb.log({"val": {"loss": mean_val_loss}})
    

# Define the training function
def train_clip(
        experiment_config: ExperimentConfig,
        train_dataset: DataLoader,
        val_dataset: DataLoader,
        model: CLIPModel
):
    
    # Create the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=experiment_config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=experiment_config.epochs)

    # Create the data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=experiment_config.batch_size, 
        shuffle=True)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=experiment_config.batch_size,
        shuffle=True)

    # Train and validate the model
    for epoch in range(experiment_config.epochs):

        # Train over each batch and report to WandB
        model.train() # Sets model into train mode, thus enable grads
        fn_train(epoch, model, optimizer, train_loader)

        # Evaluate the model on the validation set
        model.eval() # Sets model into val mode, thus no grads
        fn_val(model, val_loader)

        torch.cuda.empty_cache()

        # Update the learning rate scheduler
        scheduler.step()

    return model


#===========================================================================
#                            DATASET LOADING
#===========================================================================

def parse_img_id(path: str):
    return int(path.split('.')[0].split('/')[-1:][0])


def load_dataset_pairs():

    images_pickle_exists = os.path.exists(f'{PICKLE_REPO}/images.pkl')
    labels_pickle_exists = os.path.exists(f'{PICKLE_REPO}/labels.pkl')

    # If the pickles exist the load into memory and use those instead
    if images_pickle_exists and labels_pickle_exists:
        labels = load_object(PICKLE_REPO, 'labels')
        images = load_object(PICKLE_REPO, 'images')
    else:
        df_train = pd.read_csv("data/input/dataset/train_set.csv")
        df_hotels = pd.read_csv("data/input/dataset/hotel_info.csv")
        df_chains = pd.read_csv("data/input/dataset/chain_info.csv")

        images = glob.glob("data/images/*/*.jpg")
        labels = []

        for path in track(images, description="Preparing dataset..."):
            img_id = parse_img_id(path)

            hotel_id = df_train.loc[df_train['image_id'] == img_id]['hotel_id'].iloc[0]
            chain_id = df_hotels.loc[df_hotels['hotel_id'] == hotel_id]['chain_id'].iloc[0]
            chain_name = df_chains.loc[df_chains['chain_id'] == chain_id]['chain_name'].iloc[0]

            labels.append(chain_name)

        # Cache as Pickle files to be loaded for another run
        save_object(images, PICKLE_REPO, 'images')
        save_object(labels, PICKLE_REPO, 'labels')
    
    return (images, labels)


#===========================================================================
#                            MAIN CONTROL
#===========================================================================

if __name__ == "__main__":

    build_expected_folders()
    console = Console()

    with  console.status("Loading dataset...") as status:
        status.update(status=f'Parsing experiment setup...')

        # Parse arguments and build experiment configuration
        arg_parser = build_arg_parser()
        args = arg_parser.parse_args()

        experiment_config = ExperimentConfig(
            int(args.epochs), 
            int(args.batch_size), 
            float(args.lr)
        )

        console.log("Experiment parsed successfully!")
        console.log(experiment_config.__dict__)

        #========================= LOAD DATA & MODEL ============================
        status.update(status=f'Loading and splitting dataset...')

        # Setup WandB
        run = wandb.init( project="MLP", config=experiment_config.__dict__)

        # Load pretrained CLIP model for finetuning
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Load and split dataset
        images, labels = load_dataset_pairs()
        images_train, images_val, labels_train, labels_val = train_test_split(
            images, 
            labels, 
            test_size=0.20, 
            random_state=42
        )

        console.log("Dataset loaded and split successfully!")
        status.update(status=f'Creating dataset loading classes..')

        #======================== TRAIN AND VALIDATE ===========================
        # Build dataset loaders, such that not all images are loaded in mem at once
        train_dataset = ImageTextDataset(images_train, labels_train, processor, device)
        val_dataset = ImageTextDataset(images_val, labels_val, processor, device)

        console.log("Dataset loaders created successfully!")
        status.update(status=f'Training and validating CLIP..')

        # Train and validate model
        model = train_clip(experiment_config, train_dataset, val_dataset, model)

        console.log("Trained successfully!")

        #============================= FINALISE ================================
        model_name = f'clip_epochs{args.epochs}_bz{args.batch_size}_lr{args.batch_size}'
        status.update(status=f'Saving trained model into {MODEL_REPO}/{model_name}..')

        # Save model parameters
        model.save_pretrained(f'{MODEL_REPO}/{model_name}')

        wandb.finish()
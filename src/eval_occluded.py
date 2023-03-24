# External imports
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import glob
from rich import print
import pandas as pd
from typing import List
from torch import Tensor
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import json
from rich.progress import track

# Local imports
from clip_index import CLIPIndex
from utils.pickle_handler import save_object, load_object


class Eval():

    def __init__(self, index_name: str):
        self.index = CLIPIndex()
        self.index.load(index_name)

        self.model = CLIPVisionModelWithProjection.from_pretrained("./clip_epochs1_bz364_lr364")
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.df_train = pd.read_csv("data/input/dataset/train_set.csv")
        self.df_hotels = pd.read_csv("data/input/dataset/hotel_info.csv")
        self.df_test = pd.read_csv("data/input/dataset/test_set.csv")


    def _list_test_images(self, repo: str) -> List[str]:
        return glob.glob(repo, recursive=False)
    

    def _parse_img_id(self, image_path: str) -> int:
        return int(image_path.split('.')[0].split('/')[-1:][0])


    def _img_to_latent(self, image_path: str) -> Tensor:
        image = Image.open(image_path)

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

        return outputs.image_embeds

    def build_test_latents(self, test_repo: str):
        latents = []

        for image_path in track(self._list_test_images(test_repo), description=f"Processing latents..."):
            latents.append((self._parse_img_id(image_path), self._img_to_latent(image_path)))
        
        save_object(latents, 'test_latents_unoccluded', 'latents')

    def top_k_acc_instance(self, ks: List, test_repo: str) -> float:
        results = {f'top-{k}': 0 for k in ks}
        max_k = max(ks)
        test_latents = load_object('test_latents_unoccluded', 'latents')[0:1000]
        not_present = 0

        for gold_id, gold_latent in track(test_latents, description=f"processing up to top {max_k} instance"):
            _, ids = self.index.search_cosine(gold_latent, top_k=max_k)
            try:
                gold_hotel_id = self.df_test.loc[
                    self.df_test['image_id'] == gold_id]['hotel_id'].iloc[0]
            except:
                print("image {} not found in the fucking test set. FUCK YOU HOTELS 50K!!!!".format(gold_id))
                not_present += 1
                continue

            for idx, i in enumerate(ids[0]):
                hotel_id = self.df_train.loc[self.df_train['image_id'] == i]['hotel_id'].iloc[0]
                if hotel_id == gold_hotel_id:
                    for k in ks:
                        if idx < k:
                            results[f'top-{k}'] += 1
                    break

        total = len(test_latents) - not_present
        for k in ks:
            results[f'top-{k}'] /= total

        with open('top-k-instances.json', 'w') as fp:
            json.dump(results, fp, indent=4)

        return results



    def top_k_acc_chain(self, ks: List, test_repo: str) -> float:
        results = {f'top-{k}': 0 for k in ks}
        max_k = max(ks)
        test_latents = load_object('test_latents_occluded', 'latents')[0:10000]
        not_present = 0

        for gold_id, gold_latent in track(test_latents, description=f"processing up to top {max_k} chain"):
            _, ids = self.index.search_cosine(gold_latent, top_k=max_k)
            try:
                gold_hotel_id = self.df_test.loc[
                    self.df_test['image_id'] == gold_id]['hotel_id'].iloc[0]
            except:
                print("image {} not found in the fucking test set. FUCK YOU HOTELS 50K!!!!".format(gold_id))
                not_present += 1
                continue
            gold_chain_id = self.df_hotels.loc[
                self.df_hotels['hotel_id'] == gold_hotel_id]['chain_id'].iloc[0]

            for idx, i in enumerate(ids[0]):
                hotel_id = self.df_train.loc[
                    self.df_train['image_id'] == i]['hotel_id'].iloc[0]
                chain_id = self.df_hotels.loc[
                    self.df_hotels['hotel_id'] == hotel_id]['chain_id'].iloc[0]

                if chain_id == gold_chain_id:
                    for k in ks:
                        if idx < k:
                            results[f'top-{k}'] += 1
                    break

        total = len(test_latents) - not_present
        for k in ks:
            results[f'top-{k}'] /= total

        with open('top-k-chains.json', 'w') as fp:
            json.dump(results, fp, indent=4)

        return results



if __name__ == '__main__':
    evaluation = Eval('clip_index')
    # evaluation.build_test_latents('data/images/test/test/unoccluded/*/*/*/*.jpg')
    print("HOTEL CHAIN")
    # print(evaluation.top_k_acc_chain([1, 3, 5], 'data/images/test/test/unoccluded/*/*/*/*.jpg'))
    print()

    print("HOTEL INSTANCE")
    print(evaluation.top_k_acc_instance([1, 10, 100], 'data/images/test/test/unoccluded/*/*/*/*.jpg'))

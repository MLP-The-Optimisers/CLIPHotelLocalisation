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

# Local imports
from src.clip_index import CLIPIndex


class Eval():


    def __init__(self, index_name: str):
        self.index = CLIPIndex()
        self.index.load(index_name)

        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.df_train = pd.read_csv("data/input/dataset/train_set.csv")
        self.df_hotels = pd.read_csv("data/input/dataset/hotel_info.csv")


    def _list_test_images(self, repo: str) -> List[str]:
        return glob.glob(repo, recursive=True)
    

    def _parse_img_id(self, image_path: str) -> int:
        return int(image_path.split('.')[0].split('/')[-1:][0])


    def _img_to_latent(self, image_path: str) -> Tensor:
        image = Image.open(image_path)

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

        return outputs.image_embeds


    def top_k_acc_instance(self, ks: List, test_repo: str) -> float:
        results = {}

        for k in ks:
            accuracies = []

            for image_path in self._list_test_images(test_repo)[0:]:
                gold_id = self._parse_img_id(image_path)
                gold_latent = self._img_to_latent(image_path)

                _, ids = self.index.search_cosine(gold_latent, top_k=k)
                gold_hotel_id = self.df_train.loc[
                    self.df_train['image_id'] == gold_id]['hotel_id'].iloc[0]

                n_correct = 0
                for i in ids[0]:
                    hotel_id = self.df_train.loc[self.df_train['image_id'] == i]['hotel_id'].iloc[0]
                    if hotel_id == gold_hotel_id:
                        n_correct = 1
                        break

                accuracies.append(n_correct)

            results[f'top-{k}'] = sum(accuracies) / len(accuracies)

        return results


    def top_k_acc_chain(self, ks: List, test_repo: str) -> float:
        results = {}

        for k in ks:
            accuracies = []

            for image_path in self._list_test_images(test_repo)[0:]:
                gold_id = self._parse_img_id(image_path)
                gold_latent = self._img_to_latent(image_path)

                _, ids = self.index.search_cosine(gold_latent, top_k=k)
                gold_hotel_id = self.df_train.loc[
                    self.df_train['image_id'] == gold_id]['hotel_id'].iloc[0]
                gold_chain_id = self.df_hotels.loc[
                    self.df_hotels['hotel_id'] == gold_hotel_id]['chain_id'].iloc[0]

                n_correct = 0
                for i in ids[0]:
                    hotel_id = self.df_train.loc[
                        self.df_train['image_id'] == i]['hotel_id'].iloc[0]
                    chain_id = self.df_hotels.loc[
                        self.df_hotels['hotel_id'] == hotel_id]['chain_id'].iloc[0]

                    if chain_id == gold_chain_id:
                        n_correct = 1
                        break

                accuracies.append(n_correct)

            results[f'top-{k}'] = sum(accuracies) / len(accuracies)

        return results



if __name__ == '__main__':
    evaluation = Eval('clip_index')
    print("HOTEL CHAIN")
    print(evaluation.top_k_acc_chain([1, 3, 5], 'data/images/test/*.jpg'))
    print()

    print("HOTEL INSTANCE")
    print(evaluation.top_k_acc_instance([1, 10, 100], 'data/images/test/*.jpg'))

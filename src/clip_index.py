# External imports
from faiss import IndexFlatIP, IndexIDMap
import faiss
import torch
import os
import numpy as np

# Local imports
from utils.config import LATENT_DIMENSIONS


class CLIPIndex():

    def __init__(self,):
        current_path = os.path.dirname(os.path.realpath(__file__))
        self.index_repo = f'{current_path}/../index'


    def _val_index_present(self):
        if not self.index: raise ValueError('Index not loaded')


    def _latent_conversion(self, latents: torch.Tensor):
        latents = latents.to("cpu").detach().numpy()
        faiss.normalize_L2(latents)

        return latents


    def create_index(self, name: str):
        self.name = name
        self.index = IndexIDMap(IndexFlatIP(LATENT_DIMENSIONS))


    def load(self, name: str):
        path = f'{self.index_repo}/{name}'
        self.index = faiss.read_index(path)
    

    def save(self):
        path = f'{self.index_repo}/{self.name}'
        faiss.write_index(self.index, path)
    

    def insert_cosine(self, latents: torch.Tensor, ids: np.array):
        self._val_index_present()
        self.index.add_with_ids(self._latent_conversion(latents), ids)
    

    def search_cosine(self, latents: torch.Tensor, top_k: int = 1):
        self._val_index_present()
        return self.index.search(self._latent_conversion(latents), top_k)


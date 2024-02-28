import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import gdown
import zipfile
import os
links_dos_modelos = {'sentencebert':'https://drive.google.com/file/d/1hsejEi7BRO7p5c_GAbIEjCNCgzLUJY0d/view?usp=drive_link', 
                     'legalbert':'https://drive.google.com/file/d/1d_O3hhuTrAHuGTQ5PxkEfnoWdJQzs91Q/view?usp=drive_link'}

class Modelo:
    def __init__(self, corpus, codes):
        self.corpus_size = len(corpus)
        self.corpus = corpus
        self.codes = codes
        self.model = self.build_model()
        self.corpus_embeddings = self.model_embedding(corpus)

    def build_model(self):
        model = {}
        print("Baixando modelos...")
        for version in links_dos_modelos:
            gdown.download(url=links_dos_modelos[version], output=f"modelos/temp.zip", quiet=False, fuzzy=True)
            with zipfile.ZipFile(f"modelos/temp.zip", 'r') as zip_ref:
                zip_ref.extractall(f"modelos/")
            os.remove(f"modelos/temp.zip")
        print("Carregando modelos...")
        for version in links_dos_modelos:
            model[version] = SentenceTransformer(f"modelos/{version}/")
            model[version].max_seq_length=512
        return model

    def model_embedding(self, corpus):
        corpus_embeddings = {}
        for version in self.model:
            corpus_embeddings[version] = self.model[version].encode(corpus, convert_to_tensor=True)
        return corpus_embeddings


    def get_top(self, top_k, query, version):
        print(f'buscando com {version}')
        try:
            query_embedding = self.model[version].encode(query, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, self.corpus_embeddings[version])[0]
            top_results = torch.topk(cos_scores, k=top_k)
            return  [{"index": f'{e}', "cosine_similarity_score": cos_scores.tolist()[e],"COD": self.codes[e],
                       "txtEmenta": self.corpus[e], "Modelo":version} for e in top_results[1].numpy()
                       ]
        except Exception as e:
            print(f"ERRO ao realizar busca:{e}")
            return [{"ERRO":{e}}]
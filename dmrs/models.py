import os
import gc
import torch
import numpy as np
from evo2 import Evo2
import requests
from huggingface_hub import configure_http_backend
from tqdm import tqdm

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

class Evo2EmbedderSimple:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = Evo2(model_name)
        self.tokenizer = self.model.tokenizer


    def get_final_token_embedding(self, sequence, layer_name):
        input_ids = torch.tensor(
            self.tokenizer.tokenize(sequence),
            dtype=torch.int,
        ).unsqueeze(0).to('cuda:0') # [1, L]
        with torch.no_grad():
            _, embeddings = self.model(input_ids, return_embeddings=True, layer_names=[layer_name])
        #return embeddings[layer_name][0, -1, :].cpu().to(torch.float32).numpy() #without mean pooling
        embedding = embeddings[layer_name][0].mean(axis=0).cpu().to(torch.float32).numpy()  #with mean pooling

        # FIX: Normalize the embedding
        # from median of embedding = 91259465105408
        # embedding = embedding / (torch.norm(embedding) + 1e-8)
        
        return embedding #.cpu().to(torch.float32).numpy()

    def get_1dr_embeddings(self, fw, layer_name):
        emb_fwd = self.get_final_token_embedding(fw, layer_name)
        return emb_fwd #(3840, ): is a 1D vector

    def get_2dir_embeddings(self, fw, rv, layer_name):
        emb_fwd = self.get_final_token_embedding(fw, layer_name)
        emb_rev = self.get_final_token_embedding(rv, layer_name)
        emb_concat = np.concatenate((emb_fwd, emb_rev)) #use np.stack if need a 2D vectore (2,1920) for a special model that expects a 2D feature

        return emb_concat #(3840, ): is a 1D vector

    def embed_sequence(self, df, layer_name = 'blocks.24', clear_cache_every=500):
        
        embedding_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
            get_emb = self.get_2dir_embeddings(fw = row['sequence'], rv = row['rev_seq'], layer_name = layer_name) #add rv = row['rev_seq'] if get 2 directional sequences
            embedding_list.append(get_emb)
            
            # Clear cache periodically
            if (idx + 1) % clear_cache_every == 0:
                torch.cuda.empty_cache()
                gc.collect()

        # Export extracted embedding and store with layer name
        export_embeddings = np.array(embedding_list)

        embeddings_dir = "/home/localuser/evo2/embeddings"
        os.makedirs(embeddings_dir, exist_ok=True)

        output_path = os.path.join(embeddings_dir, f"{self.model_name}_{layer_name.replace('.', '_')}_dmrs_237k_2dr_meanpool_fixN.npy")
        print(f"Saving embeddings to: {output_path}")
        np.save(output_path, export_embeddings, allow_pickle=True)

        return embedding_list

    
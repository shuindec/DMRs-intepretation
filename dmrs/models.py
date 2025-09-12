import torch
from evo2 import Evo2
import requests
from huggingface_hub import configure_http_backend

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

class Evo2EmbedderSimple:
    def __init__(self, model_name):
        self.model = Evo2(model_name)
        self.tokenizer = self.model.tokenizer

    def embed_sequence(self, sequence, layer_name = 'blocks.26'):
        # Tokenize to IDs (not just tokens)
        input_ids = torch.tensor(
            self.tokenizer.tokenize(sequence),  # returns IDs for Evo2 tokenizer
            dtype=torch.int
        ).unsqueeze(0).to('cuda:0')  # [1, L]

        # Forward pass with embeddings from a specific layer
        with torch.no_grad():
            _, embeddings = self.model(input_ids, return_embeddings=True, layer_names=[layer_name])
        return embeddings[layer_name][0, -1, :].cpu().to(torch.float32).numpy()  # shape: (hidden_dim,)


    
import os
import torch
import numpy as np
from pathlib import Path


class ImageEncoderWrapper:
    def __init__(self, model_path=None, device='cuda'):
        from image_encoder import load_image_encoder as load_local_image_encoder
        
        self.device = device
        
        if model_path is None:
            current_dir = Path(__file__).resolve().parent
            model_path = str(current_dir)
        
        self.model = load_local_image_encoder(model_path, device)
    
    def encode_image(self, images, truncate_dim=None, **kwargs):
        try:
            with torch.no_grad():
                embeddings = self.model.encode_image(images, truncate_dim=truncate_dim, **kwargs)
                
                if isinstance(embeddings, np.ndarray):
                    embeddings = torch.from_numpy(embeddings).to(self.device)
                elif not isinstance(embeddings, torch.Tensor):
                    embeddings = torch.tensor(embeddings).to(self.device)
                elif embeddings.device != torch.device(self.device):
                    embeddings = embeddings.to(self.device)
                    
                return embeddings
        except Exception as e:
            raise e


def load_image_encoder(model_path=None, device='cuda'):
    if model_path is None:
        current_dir = Path(__file__).resolve().parent
        model_path = str(current_dir)
    return ImageEncoderWrapper(model_path, device)
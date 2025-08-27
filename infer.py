import os
import sys
import torch
import numpy as np
from pathlib import Path

current_dir = Path(__file__).resolve().parent

text_encode_path = current_dir / "text_encode"
sys.path.insert(0, str(text_encode_path))

image_path = current_dir / "image"
sys.path.insert(0, str(image_path))



from text_encode.text_encoder import load_text_encoder_with_tokenizer
from image.image_encoder_wrapper import load_image_encoder


def load_text_encoder(model_path=None, device='cuda', config=None):
    return load_text_encoder_with_tokenizer(model_path, device, config)


def load_image_encoder_wrapper(model_path=None, device='cuda'):
    return load_image_encoder(model_path, device)


def separate_tower_test():
   
    # 检查是否有可用的GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    text_encoder = load_text_encoder(device=device)
    image_encoder = load_image_encoder_wrapper(device=device)

    
    # 文本语料库
    sentences = [
        'غروب جميل على الشاطئ', # Arabic
        '海滩上美丽的日落', # Chinese
        'Un beau coucher de soleil sur la plage', # French
        'Ein wunderschöner Sonnenuntergang am Strand', # German
        'Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία', # Greek
        'समुद्र तट पर एक खूबसूरत सूर्यास्त', # Hindi
        'Un bellissimo tramonto sulla spiaggia', # Italian
        '浜辺に沈む美しい夕日', # Japanese
        '해변 위로 아름다운 일몰', # Korean
    ]
    
    # 图像URLs
    image_urls = [
        '/root/autodl-fs/jina/split_jina/images/0.jpg',
        '/root/autodl-fs/jina/split_jina/images/1.jpg'
    ]
    
    # 选择matryoshka维度，设置为None将获得完整的1024维向量
    truncate_dim = 512
    
    try:
        # 编码文本（会自动处理设备转移）
        text_embeddings = text_encoder.encode_text(sentences, truncate_dim=truncate_dim)
        # 编码图像
        image_embeddings = image_encoder.encode_image(
            image_urls, truncate_dim=truncate_dim
        )  # 支持PIL.Image、本地文件名、dataURI
        # 编码查询文本
        query = 'beautiful sunset over the beach' # English
        query_embeddings = text_encoder.encode_text(
            query, task='retrieval.query', truncate_dim=truncate_dim
        )
        
    except Exception as e:
        print(f"✗ Encoding failed: {e}")
        return
    
   
    # 确保所有嵌入都是numpy数组
    if torch.is_tensor(query_embeddings):
        query_embeddings = query_embeddings.cpu().numpy()
    if torch.is_tensor(text_embeddings):
        text_embeddings = text_embeddings.cpu().numpy()
    if torch.is_tensor(image_embeddings):
        image_embeddings = image_embeddings.cpu().numpy()
    
    print("\nSimilarity results:")
    # Text to Image
    print('En -> Img: ' + str(query_embeddings @ image_embeddings[0].T))
    # Image to Image
    print('Img -> Img: ' + str(image_embeddings[0] @ image_embeddings[1].T))
    # Text to Text
    print('En -> Ar: ' + str(query_embeddings @ text_embeddings[0].T))
    print('En -> Zh: ' + str(query_embeddings @ text_embeddings[1].T))
    print('En -> Fr: ' + str(query_embeddings @ text_embeddings[2].T))
    print('En -> De: ' + str(query_embeddings @ text_embeddings[3].T))
    print('En -> Gr: ' + str(query_embeddings @ text_embeddings[4].T))
    print('En -> Hi: ' + str(query_embeddings @ text_embeddings[5].T))
    print('En -> It: ' + str(query_embeddings @ text_embeddings[6].T))
    print('En -> Jp: ' + str(query_embeddings @ text_embeddings[7].T))
    print('En -> Ko: ' + str(query_embeddings @ text_embeddings[8].T))


if __name__ == '__main__':
    separate_tower_test()

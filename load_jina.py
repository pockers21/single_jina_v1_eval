import os
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# 统一使用与评测脚本相同的缓存目录
HF_CACHE_DIR = "/root/autodl-fs/jina/single_jina_eval/tmp_HF_cache"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ.setdefault("HF_HUB_CACHE", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)

# 配置项（硬编码，对齐 eval/run1.sh）
MODEL_PATH = "/root/autodl-fs/jina/single_jina_eval/jina_model"
TRUNCATE_DIM = 1024
QUERY_TEXT = "beautiful sunset over the beach"


def _resolve_model_dir(path: str) -> str:
    """兼容 .bin/.safetensors；若为文件则返回其父目录。"""
    if os.path.isfile(path) and (path.endswith(".bin") or path.endswith(".safetensors")):
        return os.path.dirname(path)
    return path


def compute_similarity(embedding1, embedding2):
    """计算两个嵌入向量之间的余弦相似度。
    假设底层模型已输出单位化向量，与评测链路一致，直接点乘即可。
    """
    if isinstance(embedding1, torch.Tensor):
        embedding1 = embedding1.cpu().numpy()
    if isinstance(embedding2, torch.Tensor):
        embedding2 = embedding2.cpu().numpy()

    if embedding1.ndim == 1:
        embedding1 = embedding1.reshape(1, -1)
    if embedding2.ndim == 1:
        embedding2 = embedding2.reshape(1, -1)

    return float(embedding1 @ embedding2.T)


def text_to_text_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    embeddings = model.encode([text1, text2])
    return compute_similarity(embeddings[0], embeddings[1])


def find_most_similar_text(query_text: str, candidate_texts: List[str], model: SentenceTransformer):
    similarities = []
    for candidate in candidate_texts:
        sim = text_to_text_similarity(query_text, candidate, model)
        similarities.append(sim)

    max_idx = int(np.argmax(similarities))
    return candidate_texts[max_idx], similarities[max_idx], similarities


def _set_max_seq_len(model: SentenceTransformer, max_length: int = 512) -> None:
    try:
        print("max len", model.max_seq_length)
        model.max_seq_length = max_length
    except Exception:
        pass


def main():
    model_dir = _resolve_model_dir(MODEL_PATH)

    # 与评测链路一致：使用 SentenceTransformer 文本编码，设置截断维度与最大长度
    # 加载时直接设置 device='cuda'，并在加载后切换为半精度
    assert torch.cuda.is_available(), "未检测到可用的 CUDA 设备，请在有 GPU 的环境运行。"
    model = SentenceTransformer(
        model_dir,
        trust_remote_code=True,
        truncate_dim=TRUNCATE_DIM,
        device='cuda',
    )
    _set_max_seq_len(model)  # 默认 512
    try:
        model.half()
    except Exception:
        pass
    model.eval()

    sentences: List[str] = [
        'غروب جميل على الشاطئ',  # Arabic
        '海滩上美丽的日落',           # Chinese
        'Un beau coucher de soleil sur la plage',  # French
        'Ein wunderschöner Sonnenuntergang am Strand',  # German
        'Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία',  # Greek
        'समुद्र तट पर एक खूबसूरत सूर्यास्त',              # Hindi
        'Un bellissimo tramonto sulla spiaggia',         # Italian
        '浜辺に沈む美しい夕日',                             # Japanese
        '해변 위로 아름다운 일몰',                        # Korean
    ]

    # 编码（文本）
    # 此处直接在相似度计算中按需编码，无需预先保存 embeddings

    print("=== 文本对相似度测试 ===")
    query = QUERY_TEXT
    for i, sentence in enumerate(sentences):
        sim = text_to_text_similarity(query, sentence, model)
        lang_names = ['Arabic', 'Chinese', 'French', 'German', 'Greek', 'Hindi', 'Italian', 'Japanese', 'Korean']
        print(f'En -> {lang_names[i]}: {sim:.4f}')

    print("\n=== 最相似匹配测试（文本） ===")
    best_text, best_sim, _ = find_most_similar_text(query, sentences, model)
    print(f'查询文本最相似的语言: {best_text} (相似度: {best_sim:.4f})')


if __name__ == "__main__":
    main()

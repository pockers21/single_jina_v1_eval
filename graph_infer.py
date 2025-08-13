import os

hf_cache = "/root/autodl-fs/jina/single_jina_v1_eval/hf_cache"
os.environ["HF_HOME"] = hf_cache
os.environ["HF_HUB_CACHE"] = hf_cache
os.environ["TRANSFORMERS_CACHE"] = hf_cache
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 暂时不设置离线模式，让它先下载文件
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
from typing import List
import numpy as np
import torch
from PIL import Image
from jina_clip_impl import JinaCLIPConfig, JinaCLIPModel


def load_local_config(model_dir: str) -> JinaCLIPConfig:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    # 直接用 transformers 的 auto_map 机制，JinaCLIPConfig.from_pretrained 也可；
    # 此处我们从字典构建，确保不依赖远端。
    cfg = JinaCLIPConfig(**raw)
    # 强制设置 truncate_dim 以便和上游评测一致
    if not getattr(cfg, "truncate_dim", None):
        cfg.truncate_dim = raw.get("projection_dim", 1024)
    return cfg


def build_model(model_dir: str, device: str = "cuda") -> JinaCLIPModel:
    cfg = load_local_config(model_dir)
    # 从本地权重加载
    model = JinaCLIPModel.from_pretrained(
        model_dir,
        config=cfg,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model

@torch.inference_mode()
def encode_texts(model: JinaCLIPModel, texts: List[str], device: str = "cuda") -> np.ndarray:
    return model.encode_text(
        texts,
        batch_size=min(32, max(1, len(texts))),
        convert_to_numpy=True,
        device=torch.device(device),
        normalize_embeddings=True,
        truncate_dim=model.config.truncate_dim or model.config.projection_dim,
        max_length=512,
        padding=True,
        truncation=True,
    )


@torch.inference_mode()
def encode_images(model: JinaCLIPModel, images: List[str], device: str = "cuda") -> np.ndarray:
    return model.encode_image(
        images,
        batch_size=min(16, max(1, len(images))),
        convert_to_numpy=True,
        device=torch.device(device),
        normalize_embeddings=True,
        truncate_dim=model.config.truncate_dim or model.config.projection_dim,
    )

def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "jina_model")

    device = "cuda" if torch.cuda.is_available() else "cpu"

   

    # 添加模块导入日志
    import sys
    original_import = __builtins__.__import__
    def logging_import(name, *args, **kwargs):
        result = original_import(name, *args, **kwargs)
        if ('modeling_lora' in name or 'modeling_xlm_roberta' in name or 'xlm_roberta' in name) and hasattr(result, '__file__') and result.__file__:
            print(f"[日志] 导入模块: {name} -> 文件路径: {result.__file__}")
        return result
    __builtins__.__import__ = logging_import

    print("[日志] 开始构建模型...")
    model = build_model(model_dir, device=device)
    print("[日志] 模型构建完成")

    # 检查已导入的关键模块
    print("[日志] === 检查已导入的关键模块 ===")
    for module_name, module in sys.modules.items():
        if ('modeling_lora' in module_name or 'xlm_roberta' in module_name) and hasattr(module, '__file__') and module.__file__:
            if 'transformers_modules' in module.__file__:
                print(f"[日志] 使用下载文件: {module_name} -> {module.__file__}")
    print("[日志] === 模块检查完成 ===")

    query_text = "beautiful sunset over the beach"
    sentences = [
        "ﻍﺭﻮﺑ ﺞﻤﻴﻟ ﻊﻟﻯ ﺎﻠﺷﺎﻄﺋ",
        "海滩上美丽的日落",
        "Un beau coucher de soleil sur la plage",
        "Ein wunderschöner Sonnenuntergang am Strand",
        "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία",
        "समुद्र तट पर एक खूबसूरत सूर्यास्त",
        "Un bellissimo tramonto sulla spiaggia",
        "浜辺に沈む美しい夕日",
        "해변 위로 아름다운 일몰",
    ]

    text_embs = encode_texts(model, [query_text] + sentences, device=device)
    query_emb, cand_embs = text_embs[0], text_embs[1:]

    sims = cand_embs @ query_emb.reshape(-1, 1)
    sims = sims.reshape(-1)
    best_idx = int(np.argmax(sims))
    print("Best:", sentences[best_idx])
    print("Similarity:", float(sims[best_idx]))


if __name__ == "__main__":
    main()
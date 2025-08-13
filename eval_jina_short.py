#!/usr/bin/env python3
"""
JinaCLIP-v2 模型召回能力评估（本地计算图实现）
- 任务与数据与原版保持一致：文搜图、图搜文、文搜文三类
- 切换为本地 `jina_clip_impl` 计算图与本地 `jina_model` 权重
"""

import os
import json
import numpy as np
import torch
import random
from tqdm import tqdm
import time
from typing import Dict, List, Tuple
from PIL import Image
import traceback

# 使用与原工程相同的缓存目录，且开启离线模式
HF_CACHE_DIR = "/root/autodl-tmp/bge_vl/datasets/single_jina_eval/tmp_HF_cache"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ.setdefault("HF_HUB_CACHE", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from jina_clip_impl import JinaCLIPModel


# 本地模型路径与配置
MODEL_PATH = "/root/autodl-tmp/bge_vl/datasets/single_jina3_eval/jina_model"
TRUNCATE_DIM = 1024


def print_progress(message: str, model_name: str = "JinaCLIP-v2"):
    timestamp = time.strftime("%H:%M:%S")
    if model_name:
        print(f"[{timestamp}] [{model_name}] {message}")
    else:
        print(f"[{timestamp}] {message}")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[{timestamp}] GPU Memory: {allocated:.1f}GB / {total:.1f}GB")
    print("-" * 80)


class JinaCLIPLocalWrapper:
    """本地JinaCLIP计算图封装，暴露与旧包装器一致的方法签名。"""

    def __init__(self):
        self.model_name = "JinaCLIP-v2(LocalGraph)"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.truncate_dim = TRUNCATE_DIM
        self._load_model()

    def _load_model(self):
        print_progress("开始加载JinaCLIP-v2（本地计算图）", self.model_name)
        assert os.path.isdir(MODEL_PATH), f"模型目录不存在: {MODEL_PATH}"

        self.model: JinaCLIPModel = JinaCLIPModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self.model.to(self.device)
        self.model.eval()
        print_progress("模型加载完成", self.model_name)

    def encode_text_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        print_progress(f"开始编码 {len(texts)} 个文本，批次大小: {batch_size}", self.model_name)
        with torch.inference_mode():
            embs = self.model.encode_text(
                texts,
                batch_size=min(batch_size, max(1, len(texts))),
                convert_to_numpy=True,
                device=self.device,
                normalize_embeddings=True,
                truncate_dim=self.truncate_dim,
                padding=True,
                truncation=True,
                max_length=512,
            )
            if isinstance(embs, torch.Tensor):
                embs = embs.cpu().numpy()
        print_progress(f"文本编码完成，维度：{np.asarray(embs).shape}", self.model_name)
        return np.asarray(embs)

    def encode_image_batch(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        print_progress(f"开始编码 {len(image_paths)} 个图像，批次大小: {batch_size}", self.model_name)
        with torch.inference_mode():
            imgs = []
            for p in image_paths:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    imgs.append(Image.new("RGB", (224, 224), color="white"))
            embs = self.model.encode_image(
                imgs,
                batch_size=min(batch_size, max(1, len(imgs))),
                convert_to_numpy=True,
                device=self.device,
                normalize_embeddings=True,
                truncate_dim=self.truncate_dim,
            )
            if isinstance(embs, torch.Tensor):
                embs = embs.cpu().numpy()
        print_progress(f"图像编码完成，维度：{np.asarray(embs).shape}", self.model_name)
        return np.asarray(embs)


class DatasetLoader:
    @staticmethod
    def load_image_text_dataset(data_path: str, images_path: str = None) -> Tuple[List[str], List[str], List[str]]:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts, image_paths, indices = [], [], []
        for item in data:
            texts.append(item['text'])
            if images_path and 'image_path' in item:
                image_paths.append(os.path.join(images_path, item['image_path']))
            indices.append(str(item.get('index', len(indices))))
        return texts, image_paths, indices

    @staticmethod
    def load_text_text_dataset(data_path: str) -> Tuple[List[str], List[str], List[str]]:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        s1, s2, indices = [], [], []
        for item in data:
            s1.append(item['sentence1'])
            s2.append(item['sentence2'])
            indices.append(str(item.get('index', len(indices))))
        return s1, s2, indices


class RecallEvaluator:
    def __init__(self, model: JinaCLIPLocalWrapper):
        self.model = model

    def calculate_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norms = np.linalg.norm(a, axis=1, keepdims=True)
        b_norms = np.linalg.norm(b, axis=1, keepdims=True)
        a_norms = np.where(a_norms == 0, 1e-8, a_norms)
        b_norms = np.where(b_norms == 0, 1e-8, b_norms)
        a = a / a_norms
        b = b / b_norms
        return np.dot(a, b.T)

    def calculate_recall_at_k(self, sims: np.ndarray, k_values: List[int] = [1, 3, 5, 10]) -> Dict[int, float]:
        n = sims.shape[0]
        recalls = {}
        for k in k_values:
            correct = 0
            for i in range(n):
                top_k = np.argsort(sims[i])[-k:]
                if i in top_k:
                    correct += 1
            recalls[k] = correct / n
        return recalls

    @staticmethod
    def build_resource_pool(all_data: List, pool_size: int, seed: int = 42) -> Tuple[List, List[int]]:
        if len(all_data) <= pool_size:
            return all_data, list(range(len(all_data)))
        local_random = random.Random(seed)
        indices = local_random.sample(range(len(all_data)), pool_size)
        indices.sort()
        pool_data = [all_data[i] for i in indices]
        return pool_data, indices

    def get_pool_sizes(self, dataset_size: int) -> List[int]:
        base_sizes = [1000, 2000, 3000]
        valid = [s for s in base_sizes if s <= dataset_size]
        if dataset_size not in base_sizes and dataset_size > (max(valid) if valid else 0):
            valid.append(dataset_size)
        elif not valid:
            valid.append(dataset_size)
        return valid

    def evaluate_text_to_image(self, texts: List[str], image_paths: List[str], batch_size: int = 32) -> Dict:
        print_progress("开始文搜图评测", self.model.model_name)
        results = {}
        for idx, pool_size in enumerate(self.get_pool_sizes(len(texts)), 1):
            print_progress(f"文搜图 [{idx}/{len(self.get_pool_sizes(len(texts)))}] 资源池: {pool_size}", self.model.model_name)
            pool_texts, pool_indices = RecallEvaluator.build_resource_pool(texts, pool_size)
            pool_images = [image_paths[i] for i in pool_indices]
            text_emb = self.model.encode_text_batch(pool_texts, batch_size)
            img_emb = self.model.encode_image_batch(pool_images, batch_size)
            sims = self.calculate_similarity(text_emb, img_emb)
            results[pool_size] = self.calculate_recall_at_k(sims)
            print_progress(f"资源池{pool_size}结果: {results[pool_size]}", self.model.model_name)
        print_progress("文搜图评测完成", self.model.model_name)
        return results

    def evaluate_image_to_text(self, texts: List[str], image_paths: List[str], batch_size: int = 32) -> Dict:
        print_progress("开始图搜文评测", self.model.model_name)
        results = {}
        for idx, pool_size in enumerate(self.get_pool_sizes(len(texts)), 1):
            print_progress(f"图搜文 [{idx}/{len(self.get_pool_sizes(len(texts)))}] 资源池: {pool_size}", self.model.model_name)
            pool_texts, pool_indices = RecallEvaluator.build_resource_pool(texts, pool_size)
            pool_images = [image_paths[i] for i in pool_indices]
            img_emb = self.model.encode_image_batch(pool_images, batch_size)
            text_emb = self.model.encode_text_batch(pool_texts, batch_size)
            sims = self.calculate_similarity(img_emb, text_emb)
            results[pool_size] = self.calculate_recall_at_k(sims)
            print_progress(f"资源池{pool_size}结果: {results[pool_size]}", self.model.model_name)
        print_progress("图搜文评测完成", self.model.model_name)
        return results

    def evaluate_text_to_text(self, s1: List[str], s2: List[str], batch_size: int = 32) -> Dict:
        print_progress("开始文搜文评测", self.model.model_name)
        results = {}
        for idx, pool_size in enumerate(self.get_pool_sizes(len(s1)), 1):
            print_progress(f"文搜文 [{idx}/{len(self.get_pool_sizes(len(s1)))}] 资源池: {pool_size}", self.model.model_name)
            pool_s1, pool_indices = RecallEvaluator.build_resource_pool(s1, pool_size)
            pool_s2 = [s2[i] for i in pool_indices]
            e1 = self.model.encode_text_batch(pool_s1, batch_size)
            e2 = self.model.encode_text_batch(pool_s2, batch_size)
            sims = self.calculate_similarity(e1, e2)
            results[pool_size] = self.calculate_recall_at_k(sims)
            print_progress(f"资源池{pool_size}结果: {results[pool_size]}", self.model.model_name)
        print_progress("文搜文评测完成", self.model.model_name)
        return results


def evaluate_jinaclip_model(datasets_config: Dict, batch_size: int, results_dir: str) -> Dict:
    print_progress("开始评估 JinaCLIP-v2 (本地计算图)", "")
    try:
        model = JinaCLIPLocalWrapper()
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        evaluator = RecallEvaluator(model)
        effective_bs = min(batch_size // 4, 64)
        print_progress(f"使用批次大小: {effective_bs}", model.model_name)

        model_results = {}
        for dataset_idx, (name, cfg) in enumerate(datasets_config.items(), 1):
            print_progress(f"[{dataset_idx}/{len(datasets_config)}] 开始评测数据集：{name} (类型: {cfg['type']})", model.model_name)
            if cfg['type'] == 'image_text':
                texts, images, _ = DatasetLoader.load_image_text_dataset(cfg['data_path'], cfg.get('images_path'))
                print_progress(f"已加载 {len(texts)} 个 {name} 样本", model.model_name)
                t2i = evaluator.evaluate_text_to_image(texts, images, effective_bs)
                i2t = evaluator.evaluate_image_to_text(texts, images, effective_bs)
                model_results[name] = { 'text_to_image': t2i, 'image_to_text': i2t }
                print_progress(f"数据集 {name} 评测完成", model.model_name)
            elif cfg['type'] == 'text_text':
                s1, s2, _ = DatasetLoader.load_text_text_dataset(cfg['data_path'])
                print_progress(f"已加载 {len(s1)} 个 {name} 样本", model.model_name)
                t2t = evaluator.evaluate_text_to_text(s1, s2, effective_bs)
                model_results[name] = { 'text_to_text': t2t }
                print_progress(f"数据集 {name} 评测完成", model.model_name)

        os.makedirs(results_dir, exist_ok=True)
        model_result_file = os.path.join(results_dir, 'JinaCLIP-v2_results.json')
        with open(model_result_file, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        print_progress(f"结果已保存到 {model_result_file}", model.model_name)
        return {"JinaCLIP-v2": model_results}

    except Exception as e:
        print_progress(f"JinaCLIP-v2 模型评测失败: {str(e)}", "")
        print(traceback.format_exc())
        return {"JinaCLIP-v2": {'error': str(e)}}


def main():
    print_progress("开始JinaCLIP-v2模型召回能力评估 (本地计算图)", "")

    datasets_dir = "/root/autodl-tmp/bge_vl/eval/datasets"
    results_dir = "/root/autodl-tmp/bge_vl/datasets/single_jina3_eval/short_results"
    os.makedirs(results_dir, exist_ok=True)

    batch_size = 512
    datasets_config = {
        'wukong': {
            'type': 'image_text',
            'data_path': os.path.join(datasets_dir, 'wukong/data.json'),
            'images_path': os.path.join(datasets_dir, 'wukong/images')
        },
        'coco_captions': {
            'type': 'image_text',
            'data_path': os.path.join(datasets_dir, 'coco_captions/data.json'),
            'images_path': os.path.join(datasets_dir, 'coco_captions/images')
        },
        'Emoji_dataset-Openmoji': {
            'type': 'image_text',
            'data_path': os.path.join(datasets_dir, 'Emoji_dataset-Openmoji/data.json'),
            'images_path': os.path.join(datasets_dir, 'Emoji_dataset-Openmoji/images')
        },
        'chinese-tst': {
            'type': 'text_text',
            'data_path': os.path.join(datasets_dir, 'chinese-tst/data.json')
        },
        'stsbenchmark-sts': {
            'type': 'text_text',
            'data_path': os.path.join(datasets_dir, 'stsbenchmark-sts/data.json')
        }
    }

    print_progress(f"配置 {len(datasets_config)} 个数据集", "")
    results = evaluate_jinaclip_model(datasets_config, batch_size, results_dir)

    all_results_file = os.path.join(results_dir, 'all_results.json')
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print_progress("JinaCLIP-v2评测已完成!", "")
    print_progress(f"结果保存目录: {results_dir}", "")
    print_progress(f"汇总结果文件: {all_results_file}", "")


if __name__ == "__main__":
    # 网络优化命令（可忽略失败）
    print_progress("执行网络优化命令", "")
    os.system("bash -c 'source /etc/network_turbo 2>/dev/null || true'")
    main()



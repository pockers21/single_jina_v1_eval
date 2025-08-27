#!/usr/bin/env python3
"""
Split JinaCLIP长文本评估脚本
基于jinaclip_long_eval.py，对长评估数据集进行文搜图、图搜文、文搜文三种测试
支持资源池大小为1000, 2000, 3000的可复现评估
使用分离的文本编码器和图像编码器加载方案
"""

import os
import sys
import json
import numpy as np
import torch
import random
from tqdm import tqdm
import time
from typing import Dict, List, Tuple
from PIL import Image
import traceback
from pathlib import Path

# 添加模型路径
current_dir = Path(__file__).resolve().parent.parent
text_encode_path = current_dir / "text_encode"
sys.path.insert(0, str(text_encode_path))
image_path = current_dir / "image"
sys.path.insert(0, str(image_path))

# 添加父目录到路径
sys.path.insert(0, str(current_dir))

# 导入分离的编码器
from text_encode.text_encoder import load_text_encoder_with_tokenizer
from image.image_encoder_wrapper import load_image_encoder

def print_progress(message: str, model_name: str = "Split-JinaCLIP"):
    """打印带时间戳的进度信息"""
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

class SplitJinaCLIPWrapper:
    """Split JinaCLIP模型包装器"""
    
    def __init__(self):
        self.model_name = "Split-JinaCLIP"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.truncate_dim = 1024  # 使用全维度
        self._load_models()
        
    def _load_models(self):
        """加载分离的文本和图像编码器"""
        print_progress("开始加载Split JinaCLIP分离编码器", self.model_name)
        
        try:
            # 加载文本编码器
            print_progress("加载文本编码器", self.model_name)
            self.text_encoder = load_text_encoder_with_tokenizer(
                model_path=None,  # 使用默认路径
                device=self.device
            )
            
            # 加载图像编码器
            print_progress("加载图像编码器", self.model_name)
            self.image_encoder = load_image_encoder(
                model_path=None,  # 使用默认路径
                device=self.device
            )
            
            print_progress("Split JinaCLIP分离编码器加载成功", self.model_name)
            
        except Exception as e:
            print_progress(f"Split JinaCLIP分离编码器加载失败: {str(e)}", self.model_name)
            raise e
    
    def encode_text_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码文本"""
        print_progress(f"开始编码 {len(texts)} 个文本，批次大小: {batch_size}", self.model_name)
        
        all_embeddings = []
        failed_count = 0
        
        for i in tqdm(range(0, len(texts), batch_size), desc=f"[{self.model_name}] 编码文本批次"):
            batch_texts = texts[i:i+batch_size]
            
            try:
                with torch.no_grad():
                    # 使用分离的文本编码器
                    text_features = self.text_encoder.encode_text(
                        batch_texts, 
                        truncate_dim=self.truncate_dim
                    )
                    
                    # 转换为numpy数组并确保在CPU上
                    if isinstance(text_features, torch.Tensor):
                        text_features = text_features.cpu().numpy()
                    
                    # 添加到结果列表
                    if text_features.ndim == 1:
                        all_embeddings.append(text_features)
                    else:
                        for feat in text_features:
                            all_embeddings.append(feat)
                        
            except Exception as e:
                print(f"  警告: 编码文本批次失败: {e}")
                # 使用随机向量作为fallback
                emb_dim = self.truncate_dim
                fallback_embs = [np.random.normal(0, 0.01, (emb_dim,)) for _ in batch_texts]
                all_embeddings.extend(fallback_embs)
                failed_count += len(batch_texts)
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        result = np.array(all_embeddings)
        print_progress(f"文本编码完成，维度：{result.shape}，失败数量：{failed_count}", self.model_name)
        return result
    
    def encode_image_batch(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码图像"""
        print_progress(f"开始编码 {len(image_paths)} 个图像，批次大小: {batch_size}", self.model_name)
        
        all_embeddings = []
        failed_count = 0
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"[{self.model_name}] 编码图像批次"):
            batch_paths = image_paths[i:i+batch_size]
            
            try:
                with torch.no_grad():
                    # 使用分离的图像编码器
                    image_features = self.image_encoder.encode_image(
                        batch_paths, 
                        truncate_dim=self.truncate_dim
                    )
                    
                    # 转换为numpy数组并确保在CPU上
                    if isinstance(image_features, torch.Tensor):
                        image_features = image_features.cpu().numpy()
                    
                    # 添加到结果列表
                    if image_features.ndim == 1:
                        all_embeddings.append(image_features)
                    else:
                        for feat in image_features:
                            all_embeddings.append(feat)
                            
            except Exception as e:
                print(f"  警告: 编码图像批次失败: {e}")
                # 使用随机向量作为fallback
                emb_dim = self.truncate_dim
                fallback_embs = [np.random.normal(0, 0.01, (emb_dim,)) for _ in batch_paths]
                all_embeddings.extend(fallback_embs)
                failed_count += len(batch_paths)
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        result = np.array(all_embeddings)
        print_progress(f"图像编码完成，维度：{result.shape}，失败数量：{failed_count}", self.model_name)
        return result

class DatasetLoader:
    """数据集加载器"""
    
    @staticmethod
    def load_long_t2i_en_dataset(data_path: str, images_dir: str) -> Tuple[List[str], List[str], List[str]]:
        """加载英文长文本到图像数据集"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        image_paths = []
        indices = []
        
        for item in data:
            texts.append(item['text'])
            # 移除多余的 images/ 前缀
            image_filename = item['image_path']
            if image_filename.startswith('images/'):
                image_filename = image_filename[7:]  # 移除 'images/' 前缀
            image_path = os.path.join(images_dir, image_filename)
            image_paths.append(image_path)
            indices.append(str(item['index']))
            
        return texts, image_paths, indices
    
    @staticmethod
    def load_long_t2i_zh_dataset(data_path: str, images_dir: str) -> Tuple[List[str], List[str], List[str]]:
        """加载中文长文本到图像数据集"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        image_paths = []
        indices = []
        
        for item in data:
            texts.append(item['text'])
            image_path = os.path.join(images_dir, item['image_path'])
            image_paths.append(image_path)
            indices.append(str(item['index']))
            
        return texts, image_paths, indices
    
    @staticmethod
    def load_long_t_zh_dataset(data_path: str) -> Tuple[List[str], List[str], List[str]]:
        """加载中文长文本对文本数据集"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text1s = []
        text2s = []
        indices = []
        
        for i, item in enumerate(data):
            text1s.append(item['text1'])
            text2s.append(item['text2'])
            indices.append(str(i))
            
        return text1s, text2s, indices

class RecallEvaluator:
    """召回率评估器"""
    
    def __init__(self, model: SplitJinaCLIPWrapper):
        self.model = model
    
    def calculate_similarity(self, query_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
        """计算相似度矩阵"""
        # L2归一化
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        
        # 避免除零
        query_norms = np.where(query_norms == 0, 1e-8, query_norms)
        candidate_norms = np.where(candidate_norms == 0, 1e-8, candidate_norms)
        
        query_embeddings = query_embeddings / query_norms
        candidate_embeddings = candidate_embeddings / candidate_norms
        
        # 计算余弦相似度
        similarities = np.dot(query_embeddings, candidate_embeddings.T)
        return similarities
    
    def calculate_recall_at_k(self, similarities: np.ndarray, k_values: List[int] = [1, 3, 5, 10]) -> Dict[int, float]:
        """计算Recall@K"""
        n_queries = similarities.shape[0]
        recalls = {}
        
        for k in k_values:
            correct = 0
            for i in range(n_queries):
                # 获取top-k最相似的候选
                top_k_indices = np.argsort(similarities[i])[-k:]
                # 检查正确答案是否在top-k中
                if i in top_k_indices:
                    correct += 1
            recalls[k] = correct / n_queries
            
        return recalls
    
    @staticmethod
    def build_resource_pool(all_data: List, pool_size: int, seed: int = 42) -> Tuple[List, List[int]]:
        """构建资源池"""
        if len(all_data) <= pool_size:
            return all_data, list(range(len(all_data)))
        
        # 使用固定种子确保所有模型获得相同的资源池
        local_random = random.Random(seed)
        indices = local_random.sample(range(len(all_data)), pool_size)
        indices.sort()
        pool_data = [all_data[i] for i in indices]
        return pool_data, indices
    
    def get_pool_sizes(self, dataset_size: int) -> List[int]:
        """根据数据集大小确定资源池大小"""
        base_sizes = [1000, 2000, 3000]
        valid_sizes = []
        
        for size in base_sizes:
            if size <= dataset_size:
                valid_sizes.append(size)
        
        # 如果数据集太小，使用数据集本身的大小
        if not valid_sizes and dataset_size > 0:
            valid_sizes.append(dataset_size)
            
        return valid_sizes
    
    def evaluate_text_to_image(self, texts: List[str], image_paths: List[str], batch_size: int = 32) -> Dict:
        """评估文搜图任务"""
        print_progress("开始文搜图评测", self.model.model_name)
        
        # 如果数据集太大，先随机采样到3000个
        if len(texts) > 3000:
            print_progress(f"数据集样本数({len(texts)})超过3000，随机采样3000个样本", self.model.model_name)
            sampled_texts, sampled_indices = RecallEvaluator.build_resource_pool(texts, 3000)
            sampled_images = [image_paths[i] for i in sampled_indices]
            texts, image_paths = sampled_texts, sampled_images
        
        pool_sizes = self.get_pool_sizes(len(texts))
        results = {}
        
        for pool_idx, pool_size in enumerate(pool_sizes, 1):
            total_pools = len(pool_sizes)
            print_progress(f"文搜图 [{pool_idx}/{total_pools}] 开始评测资源池大小：{pool_size}", self.model.model_name)
            
            # 构建资源池
            pool_texts, pool_indices = RecallEvaluator.build_resource_pool(texts, pool_size)
            pool_images = [image_paths[i] for i in pool_indices]
            
            # 编码
            print_progress(f"文搜图 [{pool_idx}/{total_pools}] 资源池{pool_size} - 开始编码文本", self.model.model_name)
            text_embeddings = self.model.encode_text_batch(pool_texts, batch_size)
            
            print_progress(f"文搜图 [{pool_idx}/{total_pools}] 资源池{pool_size} - 开始编码图像", self.model.model_name)
            image_embeddings = self.model.encode_image_batch(pool_images, batch_size)
            
            # 计算相似度和召回率
            print_progress(f"文搜图 [{pool_idx}/{total_pools}] 资源池{pool_size} - 计算相似度和召回率", self.model.model_name)
            similarities = self.calculate_similarity(text_embeddings, image_embeddings)
            recalls = self.calculate_recall_at_k(similarities)
            
            results[pool_size] = recalls
            print_progress(f"文搜图 [{pool_idx}/{total_pools}] 资源池{pool_size}结果：{recalls}", self.model.model_name)
            
        print_progress("文搜图评测完成", self.model.model_name)
        return results
    
    def evaluate_image_to_text(self, texts: List[str], image_paths: List[str], batch_size: int = 32) -> Dict:
        """评估图搜文任务"""
        print_progress("开始图搜文评测", self.model.model_name)
        
        # 如果数据集太大，先随机采样到3000个
        if len(texts) > 3000:
            print_progress(f"数据集样本数({len(texts)})超过3000，随机采样3000个样本", self.model.model_name)
            sampled_texts, sampled_indices = RecallEvaluator.build_resource_pool(texts, 3000)
            sampled_images = [image_paths[i] for i in sampled_indices]
            texts, image_paths = sampled_texts, sampled_images
        
        pool_sizes = self.get_pool_sizes(len(texts))
        results = {}
        
        for pool_idx, pool_size in enumerate(pool_sizes, 1):
            total_pools = len(pool_sizes)
            print_progress(f"图搜文 [{pool_idx}/{total_pools}] 开始评测资源池大小：{pool_size}", self.model.model_name)
            
            # 构建资源池
            pool_texts, pool_indices = RecallEvaluator.build_resource_pool(texts, pool_size)
            pool_images = [image_paths[i] for i in pool_indices]
            
            # 编码
            print_progress(f"图搜文 [{pool_idx}/{total_pools}] 资源池{pool_size} - 开始编码图像", self.model.model_name)
            image_embeddings = self.model.encode_image_batch(pool_images, batch_size)
            
            print_progress(f"图搜文 [{pool_idx}/{total_pools}] 资源池{pool_size} - 开始编码文本", self.model.model_name)
            text_embeddings = self.model.encode_text_batch(pool_texts, batch_size)
            
            # 计算相似度和召回率
            print_progress(f"图搜文 [{pool_idx}/{total_pools}] 资源池{pool_size} - 计算相似度和召回率", self.model.model_name)
            similarities = self.calculate_similarity(image_embeddings, text_embeddings)
            recalls = self.calculate_recall_at_k(similarities)
            
            results[pool_size] = recalls
            print_progress(f"图搜文 [{pool_idx}/{total_pools}] 资源池{pool_size}结果：{recalls}", self.model.model_name)
            
        print_progress("图搜文评测完成", self.model.model_name)
        return results
    
    def evaluate_text_to_text(self, text1s: List[str], text2s: List[str], batch_size: int = 32) -> Dict:
        """评估文搜文任务"""
        print_progress("开始文搜文评测", self.model.model_name)
        
        # 如果数据集太大，先随机采样到3000个
        if len(text1s) > 3000:
            print_progress(f"数据集样本数({len(text1s)})超过3000，随机采样3000个样本", self.model.model_name)
            sampled_text1, sampled_indices = RecallEvaluator.build_resource_pool(text1s, 3000)
            sampled_text2 = [text2s[i] for i in sampled_indices]
            text1s, text2s = sampled_text1, sampled_text2
        
        pool_sizes = self.get_pool_sizes(len(text1s))
        results = {}
        
        for pool_idx, pool_size in enumerate(pool_sizes, 1):
            total_pools = len(pool_sizes)
            print_progress(f"文搜文 [{pool_idx}/{total_pools}] 开始评测资源池大小：{pool_size}", self.model.model_name)
            
            # 构建资源池
            pool_text1, pool_indices = RecallEvaluator.build_resource_pool(text1s, pool_size)
            pool_text2 = [text2s[i] for i in pool_indices]
            
            # 编码
            print_progress(f"文搜文 [{pool_idx}/{total_pools}] 资源池{pool_size} - 开始编码第一组文本", self.model.model_name)
            text1_embeddings = self.model.encode_text_batch(pool_text1, batch_size)
            
            print_progress(f"文搜文 [{pool_idx}/{total_pools}] 资源池{pool_size} - 开始编码第二组文本", self.model.model_name)
            text2_embeddings = self.model.encode_text_batch(pool_text2, batch_size)
            
            # 计算相似度和召回率
            print_progress(f"文搜文 [{pool_idx}/{total_pools}] 资源池{pool_size} - 计算相似度和召回率", self.model.model_name)
            similarities = self.calculate_similarity(text1_embeddings, text2_embeddings)
            recalls = self.calculate_recall_at_k(similarities)
            
            results[pool_size] = recalls
            print_progress(f"文搜文 [{pool_idx}/{total_pools}] 资源池{pool_size}结果：{recalls}", self.model.model_name)
            
        print_progress("文搜文评测完成", self.model.model_name)
        return results

def evaluate_split_jinaclip_long_model(datasets_config: Dict, batch_size: int, results_dir: str) -> Dict:
    """评估Split JinaCLIP长文本模型"""
    print_progress("开始评估 Split JinaCLIP 长文本模型", "")
    
    try:
        # 加载模型
        model = SplitJinaCLIPWrapper()
        
        # 设置随机种子（确保资源池选择一致）
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            
        evaluator = RecallEvaluator(model)
        
        # 使用较小的批次大小适配大模型
        effective_batch_size = min(batch_size // 4, 512)
        print_progress(f"使用批次大小: {effective_batch_size}", model.model_name)
            
        model_results = {}
        
        for dataset_idx, (dataset_name, config) in enumerate(datasets_config.items(), 1):
            total_datasets = len(datasets_config)
            print_progress(f"[{dataset_idx}/{total_datasets}] 开始评测数据集：{dataset_name} (类型: {config['type']})", model.model_name)
            
            if config['type'] == 'text_to_image':
                # 文本到图像数据集 (long_t2i_en, long_t2i_zh)
                if dataset_name == 'long_t2i_en':
                    texts, image_paths, _ = DatasetLoader.load_long_t2i_en_dataset(
                        config['data_path'], config['images_dir']
                    )
                else:  # long_t2i_zh
                    texts, image_paths, _ = DatasetLoader.load_long_t2i_zh_dataset(
                        config['data_path'], config['images_dir']
                    )
                
                print_progress(f"[{dataset_idx}/{total_datasets}] 已加载 {len(texts)} 个 {dataset_name} 样本", model.model_name)
                
                dataset_results = {}
                
                # 对于有实际图像的数据集，进行文搜图和图搜文评测
                if dataset_name == 'long_t2i_en':
                    # 文搜图任务
                    print_progress(f"[{dataset_idx}/{total_datasets}] 开始 {dataset_name} 文搜图任务", model.model_name)
                    text_to_image_results = evaluator.evaluate_text_to_image(texts, image_paths, effective_batch_size)
                    dataset_results['text_to_image'] = text_to_image_results
                    print_progress(f"[{dataset_idx}/{total_datasets}] {dataset_name} 文搜图任务完成", model.model_name)
                    
                    # 图搜文任务
                    print_progress(f"[{dataset_idx}/{total_datasets}] 开始 {dataset_name} 图搜文任务", model.model_name)
                    image_to_text_results = evaluator.evaluate_image_to_text(texts, image_paths, effective_batch_size)
                    dataset_results['image_to_text'] = image_to_text_results
                    print_progress(f"[{dataset_idx}/{total_datasets}] {dataset_name} 图搜文任务完成", model.model_name)
                
                else:  # long_t2i_zh
                    # 对于long_t2i_zh，也进行文搜图和图搜文任务
                    # 文搜图任务
                    print_progress(f"[{dataset_idx}/{total_datasets}] 开始 {dataset_name} 文搜图任务", model.model_name)
                    text_to_image_results = evaluator.evaluate_text_to_image(texts, image_paths, effective_batch_size)
                    dataset_results['text_to_image'] = text_to_image_results
                    print_progress(f"[{dataset_idx}/{total_datasets}] {dataset_name} 文搜图任务完成", model.model_name)
                    
                    # 图搜文任务
                    print_progress(f"[{dataset_idx}/{total_datasets}] 开始 {dataset_name} 图搜文任务", model.model_name)
                    image_to_text_results = evaluator.evaluate_image_to_text(texts, image_paths, effective_batch_size)
                    dataset_results['image_to_text'] = image_to_text_results
                    print_progress(f"[{dataset_idx}/{total_datasets}] {dataset_name} 图搜文任务完成", model.model_name)
                
                model_results[dataset_name] = dataset_results
                
                # 立即保存单个数据集的结果
                dataset_result_file = os.path.join(results_dir, f'Split-JinaCLIP_{dataset_name}_results.json')
                with open(dataset_result_file, 'w', encoding='utf-8') as f:
                    json.dump({dataset_name: model_results[dataset_name]}, f, indent=2, ensure_ascii=False)
                print_progress(f"数据集 {dataset_name} 结果已保存到 {dataset_result_file}", model.model_name)
                
                print_progress(f"[{dataset_idx}/{total_datasets}] 数据集 {dataset_name} 评测完成", model.model_name)
                
            elif config['type'] == 'text_to_text':
                # 文本到文本数据集 (long_t_zh)
                text1s, text2s, _ = DatasetLoader.load_long_t_zh_dataset(config['data_path'])
                
                print_progress(f"[{dataset_idx}/{total_datasets}] 已加载 {len(text1s)} 个 {dataset_name} 样本", model.model_name)
                
                # 文搜文任务
                print_progress(f"[{dataset_idx}/{total_datasets}] 开始 {dataset_name} 文搜文任务", model.model_name)
                text_to_text_results = evaluator.evaluate_text_to_text(text1s, text2s, effective_batch_size)
                print_progress(f"[{dataset_idx}/{total_datasets}] {dataset_name} 文搜文任务完成", model.model_name)
                
                model_results[dataset_name] = {
                    'text_to_text': text_to_text_results
                }
                
                # 立即保存单个数据集的结果
                dataset_result_file = os.path.join(results_dir, f'Split-JinaCLIP_{dataset_name}_results.json')
                with open(dataset_result_file, 'w', encoding='utf-8') as f:
                    json.dump({dataset_name: model_results[dataset_name]}, f, indent=2, ensure_ascii=False)
                print_progress(f"数据集 {dataset_name} 结果已保存到 {dataset_result_file}", model.model_name)
                
                print_progress(f"[{dataset_idx}/{total_datasets}] 数据集 {dataset_name} 评测完成", model.model_name)
        
        # 保存结果
        model_result_file = os.path.join(results_dir, 'Split-JinaCLIP_long_eval_results.json')
        with open(model_result_file, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        
        print_progress(f"结果已保存到 {model_result_file}", model.model_name)
        return {"Split-JinaCLIP": model_results}
        
    except Exception as e:
        print_progress(f"Split JinaCLIP 长文本模型评测失败: {str(e)}", "")
        print(traceback.format_exc())
        return {"Split-JinaCLIP": {'error': str(e)}}

def main():
    print_progress("开始Split JinaCLIP长文本模型召回能力评估", "")
    
    # 配置路径
    datasets_dir = "/root/autodl-tmp/bge_vl/datasets/long_eval/datasets"
    results_dir = "/root/autodl-tmp/bge_vl/merge_jina1/split_jina/eval/long_eval_results"
    
    os.makedirs(results_dir, exist_ok=True)
    
    batch_size = 512  # 修改批次大小为512
    
    # 数据集配置 - 重新排序，先进行文本到文本任务
    datasets_config = {
        'long_t_zh': {
            'type': 'text_to_text',
            'data_path': os.path.join(datasets_dir, 'long_t_zh/data.json')
        },
        'long_t2i_zh': {
            'type': 'text_to_image',
            'data_path': os.path.join(datasets_dir, 'long_t2i_zh/data/data.json'),
            'images_dir': os.path.join(datasets_dir, 'long_t2i_zh/data/images')
        },
        'long_t2i_en': {
            'type': 'text_to_image',
            'data_path': os.path.join(datasets_dir, 'long_t2i_en/data/data.json'),
            'images_dir': os.path.join(datasets_dir, 'long_t2i_en/data/images')
        }
    }
    
    print_progress(f"配置 {len(datasets_config)} 个数据集", "")
    
    # 评估Split JinaCLIP模型
    results = evaluate_split_jinaclip_long_model(datasets_config, batch_size, results_dir)
    
    # 保存汇总结果
    all_results_file = os.path.join(results_dir, 'all_long_eval_results.json')
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print_progress("Split JinaCLIP长文本评测已完成!", "")
    print_progress(f"结果保存目录: {results_dir}", "")
    print_progress(f"汇总结果文件: {all_results_file}", "")

if __name__ == "__main__":
    # 执行网络优化命令
    print_progress("执行网络优化命令", "")
    os.system("bash -c 'source /etc/network_turbo 2>/dev/null || true'")
    
    main()
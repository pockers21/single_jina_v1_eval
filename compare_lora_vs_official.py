#!/usr/bin/env python3
"""
对比LoRA合并实现 vs 官方JinaCLIP实现
Python vs Python的余弦相似度对比
"""
import sys
import os
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import json

def load_lora_merged_implementation():
    """加载LoRA合并后的实现（infer.py的逻辑）"""
    print("🔄 加载LoRA合并实现...")
    
    # 添加路径
    current_dir = Path('/autodl-fs/data/jina/split_jina').resolve()
    sys.path.insert(0, str(current_dir))
    sys.path.insert(0, str(current_dir / 'text_encode'))
    sys.path.insert(0, str(current_dir / 'image'))
    
    try:
        from text_encode.text_encoder import load_text_encoder_with_tokenizer
        from image.image_encoder_wrapper import load_image_encoder
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   使用设备: {device}")
        
        # 加载模型（会自动使用LoRA合并权重）
        text_encoder = load_text_encoder_with_tokenizer(device=device)
        image_encoder = load_image_encoder(device=device)
        
        print("✅ LoRA合并实现加载成功")
        return text_encoder, image_encoder, device
        
    except Exception as e:
        print(f"❌ 加载LoRA合并实现失败: {e}")
        return None, None, None

def load_official_implementation():
    """加载官方JinaCLIP实现"""
    print("🔄 加载官方JinaCLIP实现...")
    
    try:
        # 按照官方脚本方式加载
        from transformers import AutoModel
        
        model_name = "jinaai/jina-clip-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 指定下载路径到 /root/autodl-fs/jina/split_jina/offical/
        cache_dir = "/root/autodl-fs/jina/split_jina/offical"
        
        # 创建目录如果不存在
        import os
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"   从HuggingFace加载: {model_name}")
        print(f"   下载路径: {cache_dir}")
        
        # 加载官方模型到指定路径
        model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=False  # 允许从网络下载
        )
        model = model.to(device)
        model.eval()
        
        print("✅ 官方JinaCLIP实现加载成功")
        return model, device
        
    except Exception as e:
        print(f"❌ 加载官方实现失败: {e}")
        print(f"   可能需要安装或配置transformers库")
        return None, None

def encode_with_lora_implementation(text_encoder, image_encoder, text_input, image_path, device):
    """使用LoRA合并实现进行编码"""
    print("🔧 使用LoRA合并实现编码...")
    
    try:
        with torch.no_grad():
            # 编码文本
            text_embedding = text_encoder.encode_text([text_input], truncate_dim=None)
            if torch.is_tensor(text_embedding):
                text_embedding = text_embedding.cpu().numpy()
            if text_embedding.ndim > 1:
                text_embedding = text_embedding[0]  # 取第一个样本
            
            # 编码图像
            image_embedding = image_encoder.encode_image([image_path], truncate_dim=None)
            if torch.is_tensor(image_embedding):
                image_embedding = image_embedding.cpu().numpy()
            if image_embedding.ndim > 1:
                image_embedding = image_embedding[0]  # 取第一个样本
                
            print(f"   文本嵌入形状: {text_embedding.shape}, L2范数: {np.linalg.norm(text_embedding):.6f}")
            print(f"   图像嵌入形状: {image_embedding.shape}, L2范数: {np.linalg.norm(image_embedding):.6f}")
            
            return text_embedding, image_embedding
            
    except Exception as e:
        print(f"❌ LoRA实现编码失败: {e}")
        return None, None

def encode_with_official_implementation(model, text_input, image_path, device):
    """使用官方实现进行编码"""
    print("🔧 使用官方实现编码...")
    
    try:
        with torch.no_grad():
            # 按照官方脚本方式编码文本
            text_embeddings = model.encode_text([text_input], truncate_dim=None)
            if torch.is_tensor(text_embeddings):
                text_embeddings = text_embeddings.cpu().numpy()
            text_embedding = text_embeddings[0]  # 取第一个样本
            
            # 按照官方脚本方式编码图像
            image_embeddings = model.encode_image([image_path], truncate_dim=None)
            if torch.is_tensor(image_embeddings):
                image_embeddings = image_embeddings.cpu().numpy()  
            image_embedding = image_embeddings[0]  # 取第一个样本
            
            print(f"   文本嵌入形状: {text_embedding.shape}, L2范数: {np.linalg.norm(text_embedding):.6f}")
            print(f"   图像嵌入形状: {image_embedding.shape}, L2范数: {np.linalg.norm(image_embedding):.6f}")
            
            return text_embedding, image_embedding
            
    except Exception as e:
        print(f"❌ 官方实现编码失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def calculate_similarity_metrics(lora_text, lora_image, official_text, official_image):
    """计算相似度指标"""
    print("📊 计算相似度指标...")
    
    metrics = {}
    
    # 文本嵌入对比
    if lora_text is not None and official_text is not None:
        text_cosine = cosine_similarity([lora_text], [official_text])[0, 0]
        text_mse = np.mean((lora_text - official_text) ** 2)
        text_max_diff = np.max(np.abs(lora_text - official_text))
        
        metrics['text_comparison'] = {
            'cosine_similarity': text_cosine,
            'mse': text_mse,
            'max_abs_diff': text_max_diff,
            'lora_norm': np.linalg.norm(lora_text),
            'official_norm': np.linalg.norm(official_text)
        }
        
        print(f"   文本嵌入余弦相似度: {text_cosine:.8f}")
    
    # 图像嵌入对比
    if lora_image is not None and official_image is not None:
        image_cosine = cosine_similarity([lora_image], [official_image])[0, 0]
        image_mse = np.mean((lora_image - official_image) ** 2)
        image_max_diff = np.max(np.abs(lora_image - official_image))
        
        metrics['image_comparison'] = {
            'cosine_similarity': image_cosine,
            'mse': image_mse,
            'max_abs_diff': image_max_diff,
            'lora_norm': np.linalg.norm(lora_image),
            'official_norm': np.linalg.norm(official_image)
        }
        
        print(f"   图像嵌入余弦相似度: {image_cosine:.8f}")
    
    # 跨模态相似度对比
    if all(x is not None for x in [lora_text, lora_image, official_text, official_image]):
        lora_cross_modal = np.dot(lora_text, lora_image)
        official_cross_modal = np.dot(official_text, official_image)
        
        metrics['cross_modal_comparison'] = {
            'lora_text_image_similarity': lora_cross_modal,
            'official_text_image_similarity': official_cross_modal,
            'cross_modal_diff': abs(lora_cross_modal - official_cross_modal)
        }
        
        print(f"   LoRA跨模态相似度: {lora_cross_modal:.6f}")
        print(f"   官方跨模态相似度: {official_cross_modal:.6f}")
        print(f"   跨模态差异: {abs(lora_cross_modal - official_cross_modal):.6f}")
    
    return metrics

def main():
    print("🚀 LoRA合并实现 vs 官方JinaCLIP实现对比")
    print("Python vs Python 余弦相似度分析")
    print("=" * 80)
    
    # 测试输入
    test_text = "A beautiful sunset over the beach"
    test_image = "/autodl-fs/data/jina/split_jina/images/0.jpg"
    
    print(f"测试文本: {test_text}")
    print(f"测试图像: {test_image}")
    print()
    
    # 加载LoRA合并实现
    lora_text_encoder, lora_image_encoder, lora_device = load_lora_merged_implementation()
    
    # 加载官方实现
    official_model, official_device = load_official_implementation()
    
    if lora_text_encoder is None or official_model is None:
        print("❌ 无法加载所需的模型，测试终止")
        return False
    
    print(f"\n{'='*80}")
    print("🔧 执行编码对比")
    print('='*80)
    
    # LoRA实现编码
    lora_text_emb, lora_image_emb = encode_with_lora_implementation(
        lora_text_encoder, lora_image_encoder, test_text, test_image, lora_device
    )
    
    # 官方实现编码
    official_text_emb, official_image_emb = encode_with_official_implementation(
        official_model, test_text, test_image, official_device
    )
    
    if any(x is None for x in [lora_text_emb, lora_image_emb, official_text_emb, official_image_emb]):
        print("❌ 编码过程中出现错误，无法进行对比")
        return False
    
    print(f"\n{'='*80}")
    print("📊 相似度分析结果")
    print('='*80)
    
    # 计算相似度指标
    metrics = calculate_similarity_metrics(lora_text_emb, lora_image_emb, official_text_emb, official_image_emb)
    
    # 分析结果
    print(f"\n🎯 关键发现:")
    
    if 'text_comparison' in metrics:
        text_sim = metrics['text_comparison']['cosine_similarity']
        if text_sim >= 0.99:
            print(f"   ✅ 文本编码高度一致 ({text_sim:.6f})")
        elif text_sim >= 0.90:
            print(f"   🟡 文本编码基本一致 ({text_sim:.6f})")
        else:
            print(f"   ❌ 文本编码显著差异 ({text_sim:.6f})")
    
    if 'image_comparison' in metrics:
        image_sim = metrics['image_comparison']['cosine_similarity']
        if image_sim >= 0.99:
            print(f"   ✅ 图像编码高度一致 ({image_sim:.6f})")
        elif image_sim >= 0.90:
            print(f"   🟡 图像编码基本一致 ({image_sim:.6f})")
        else:
            print(f"   ❌ 图像编码显著差异 ({image_sim:.6f})")
    
    # 保存结果
    result_data = {
        'timestamp': '2025-08-27',
        'comparison_type': 'lora_vs_official_python',
        'test_inputs': {
            'text': test_text,
            'image': test_image
        },
        'metrics': {k: {mk: float(mv) if isinstance(mv, (np.float32, np.float64)) else mv 
                       for mk, mv in v.items()} for k, v in metrics.items()},
        'embeddings_sample': {
            'lora_text': lora_text_emb[:10].tolist() if lora_text_emb is not None else None,
            'lora_image': lora_image_emb[:10].tolist() if lora_image_emb is not None else None,
            'official_text': official_text_emb[:10].tolist() if official_text_emb is not None else None,
            'official_image': official_image_emb[:10].tolist() if official_image_emb is not None else None
        }
    }
    
    output_file = 'lora_vs_official_comparison.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细结果已保存到: {output_file}")
    
    # 结论
    print(f"\n🏆 结论:")
    if 'text_comparison' in metrics and 'image_comparison' in metrics:
        avg_sim = (metrics['text_comparison']['cosine_similarity'] + 
                  metrics['image_comparison']['cosine_similarity']) / 2
        if avg_sim >= 0.95:
            print("LoRA合并实现与官方实现高度一致，差异主要在C++实现端")
        elif avg_sim >= 0.80:
            print("LoRA合并实现与官方实现基本一致，存在一定差异")  
        else:
            print("LoRA合并实现与官方实现存在显著差异，需要检查LoRA合并逻辑")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
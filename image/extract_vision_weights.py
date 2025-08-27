#!/usr/bin/env python3
"""
从完整的JinaCLIP模型中提取图像编码器权重
"""

import torch
import os

def extract_vision_weights():
    """
    从jina-clip-v2的pytorch_model.bin中提取vision相关的权重
    """
    # 加载完整模型权重
    full_model_path = '/root/autodl-tmp/bge_vl/jina-clip-v2/pytorch_model.bin'
    print(f"Loading full model weights from: {full_model_path}")
    
    if not os.path.exists(full_model_path):
        print(f"Error: Model file not found at {full_model_path}")
        return False
    
    try:
        full_weights = torch.load(full_model_path, map_location='cpu', weights_only=True)
        print(f"Loaded {len(full_weights)} parameters from full model")
        
        # 提取vision相关的权重
        vision_weights = {}
        vision_projection_weights = {}
        
        for key, value in full_weights.items():
            if key.startswith('vision_model.'):
                # 移除'vision_model.'前缀
                new_key = key[len('vision_model.'):]
                vision_weights[new_key] = value
                print(f"Extracted vision weight: {key} -> {new_key}")
            elif key.startswith('visual_projection.'):
                # 保留visual_projection权重
                vision_projection_weights[key] = value
                print(f"Extracted projection weight: {key}")
        
        print(f"\nExtracted {len(vision_weights)} vision model parameters")
        print(f"Extracted {len(vision_projection_weights)} projection parameters")
        
        # 保存vision模型权重
        vision_weights_path = '/root/autodl-tmp/bge_vl/merge_jina1/text_jina/image/vision_model_weights.bin'
        torch.save(vision_weights, vision_weights_path)
        print(f"Saved vision model weights to: {vision_weights_path}")
        
        # 保存projection权重
        if vision_projection_weights:
            projection_weights_path = '/root/autodl-tmp/bge_vl/merge_jina1/text_jina/image/visual_projection_weights.bin'
            torch.save(vision_projection_weights, projection_weights_path)
            print(f"Saved projection weights to: {projection_weights_path}")
        
        # 打印一些权重信息用于验证
        print("\nSample vision weights:")
        for i, (key, value) in enumerate(vision_weights.items()):
            if i < 5:  # 只显示前5个
                print(f"  {key}: {value.shape}")
            else:
                break
        
        return True
        
    except Exception as e:
        print(f"Error extracting weights: {e}")
        return False

if __name__ == '__main__':
    success = extract_vision_weights()
    if success:
        print("\n✓ Vision weights extraction completed successfully!")
    else:
        print("\n❌ Vision weights extraction failed!")
from transformers import Qwen2ForCausalLM, AutoConfig
import torch
import os
import json
from safetensors import safe_open

def print_model_architecture(model_path):
    """打印模型架构信息"""
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        print("\n--- Model Architecture Information ---")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Model Type: {config.get('model_type', 'Unknown')}")
        print(f"Architecture: {config.get('architectures', 'Unknown')}")
        print(f"Hidden Size: {config.get('hidden_size', 'Unknown')}")
        print(f"Intermediate Size: {config.get('intermediate_size', 'Unknown')}")
        print(f"Number of Layers: {config.get('num_hidden_layers', 'Unknown')}")
        print(f"Number of Attention Heads: {config.get('num_attention_heads', 'Unknown')}")
        print(f"Vocabulary Size: {config.get('vocab_size', 'Unknown')}")
        print(f"Max Position Embeddings: {config.get('max_position_embeddings', 'Unknown')}")
        print("-" * 50)
        return config
    else:
        print(f"Warning: Config file not found at {config_path}")
        return None

def load_safetensors_model(model_path):
    """加载safetensors格式的模型权重"""
    print("\n--- Attempting to load safetensors files ---")
    
    # 检查是否有单个模型文件
    single_file = os.path.join(model_path, 'model.safetensors')
    if os.path.exists(single_file):
        print(f"Found single safetensors file: {single_file}")
        with safe_open(single_file, framework="pt", device="cpu") as f:
            state_dict = {key: f.get_tensor(key) for key in f.keys()}
        return state_dict
    
    # 检查是否有分片文件
    index_file = os.path.join(model_path, 'model.safetensors.index.json')
    if os.path.exists(index_file):
        print(f"Found safetensors index file: {index_file}")
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        # 获取所有分片文件
        shard_files = set(index_data['weight_map'].values())
        state_dict = {}
        
        for shard_file in shard_files:
            shard_path = os.path.join(model_path, shard_file)
            if os.path.exists(shard_path):
                print(f"Loading shard: {shard_path}")
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            else:
                print(f"Warning: Shard file not found: {shard_path}")
        
        return state_dict
    
    print("No safetensors files found")
    return None

def load_bin_model(model_path):
    """加载bin格式的模型权重"""
    print("\n--- Attempting to load bin files ---")
    
    # 检查是否有单个模型文件
    single_file = os.path.join(model_path, 'pytorch_model.bin')
    if os.path.exists(single_file):
        print(f"Found single bin file: {single_file}")
        try:
            state_dict = torch.load(single_file, map_location="cpu", weights_only=True)
            return state_dict
        except Exception as e:
            print(f"Error loading {single_file}: {e}")
            return None
    
    # 检查是否有分片文件
    index_file = os.path.join(model_path, 'pytorch_model.bin.index.json')
    if os.path.exists(index_file):
        print(f"Found bin index file: {index_file}")
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        # 获取所有分片文件
        shard_files = set(index_data['weight_map'].values())
        state_dict = {}
        
        for shard_file in shard_files:
            shard_path = os.path.join(model_path, shard_file)
            if os.path.exists(shard_path):
                print(f"Loading shard: {shard_path}")
                try:
                    shard_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
                    state_dict.update(shard_dict)
                except Exception as e:
                    print(f"Error loading {shard_path}: {e}")
            else:
                print(f"Warning: Shard file not found: {shard_path}")
        
        return state_dict
    
    print("No bin files found")
    return None

def main():
    cu_path = "/root/autodl-tmp/bge_vl/merge_jina1/split_jina/text_encode"
    
    print(f"Inspecting model at: {cu_path}")
    if not os.path.isdir(cu_path):
        print(f"Error: Directory not found at {cu_path}")
        return
    
    # 打印模型架构信息
    config = print_model_architecture(cu_path)
    
    # --- 使用transformers加载模型 ---
    print("\n--- Loading model using transformers ---")
    try:
        model = Qwen2ForCausalLM.from_pretrained(cu_path)
        print("Model loaded successfully using Qwen2ForCausalLM.")

        print("\n--- All Model Parameters (Name and Shape) ---")
        total_params = 0
        for name, param in model.named_parameters():
            print(f"Layer: {name}, Shape: {param.shape}")
            # Print first 10 elements
            first_10 = param.flatten()[:10]
            print(f"  First 10 elements: {first_10.tolist()}")
            total_params += param.numel()
        print("-" * 50)
        print(f"Total parameters in loaded model: {total_params:,}")
        print("-" * 50)

    except Exception as e:
        print(f"Error loading model with transformers: {e}")
    
    # --- 尝试直接加载权重文件 ---
    print("\n--- Attempting to load model weights directly ---")
    
    # 优先尝试safetensors格式
    state_dict = load_safetensors_model(cu_path)
    
    # 如果没有safetensors，尝试bin格式
    if state_dict is None:
        state_dict = load_bin_model(cu_path)
    
    if state_dict is not None:
        print("\n--- All State Dict Keys and Shapes ---")
        total_params_dict = 0
        for key, tensor in state_dict.items():
            print(f"{key}: {tensor.shape}")
            # Print first 10 elements
            first_10 = tensor.flatten()[:10]
            print(f"  First 10 elements: {first_10.tolist()}")
            total_params_dict += tensor.numel()
        print("-" * 50)
        print(f"Total parameters in state dict: {total_params_dict:,}")
        print("-" * 50)
    else:
        print("Failed to load model weights directly")

if __name__ == "__main__":
    main() 

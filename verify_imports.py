#!/usr/bin/env python3
"""验证 load_lora.py 实际导入的模块"""

import sys
from pathlib import Path

# 复制 load_lora.py 的路径设置
current_dir = Path('/root/autodl-fs/jina/HF_cache_raw').resolve()

# 添加路径（与load_lora.py相同的顺序）
hub_model_path = current_dir / "hub" / "models--jinaai--xlm-roberta-flash-implementation" / "snapshots" / "2b6bc3f30750b3a9648fe9b63448c09920efe9be"
sys.path.insert(0, str(hub_model_path))

modules_path = current_dir / "modules" / "transformers_modules" / "jinaai" / "xlm-roberta-flash-implementation"
sys.path.insert(0, str(modules_path))

print("=== Python搜索路径（前5个）===")
for i, path in enumerate(sys.path[:5]):
    print(f"{i}: {path}")

print("\n=== 尝试导入模块 ===")
try:
    from configuration_xlm_roberta import XLMRobertaFlashConfig
    print(f"✅ XLMRobertaFlashConfig 导入成功")
    print(f"   模块文件: {XLMRobertaFlashConfig.__module__}")
    
    # 检查模块文件路径
    import configuration_xlm_roberta
    if hasattr(configuration_xlm_roberta, '__file__'):
        print(f"   文件路径: {configuration_xlm_roberta.__file__}")
        
except ImportError as e:
    print(f"❌ 导入失败: {e}")

try:
    from modeling_xlm_roberta import XLMRobertaModel
    print(f"✅ XLMRobertaModel 导入成功")
    
    import modeling_xlm_roberta
    if hasattr(modeling_xlm_roberta, '__file__'):
        print(f"   文件路径: {modeling_xlm_roberta.__file__}")
        
    # 检查是否是CLIP模型
    print(f"   XLMRobertaModel类: {XLMRobertaModel}")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")

# 检查是否有jina-clip相关的模块被导入
print(f"\n=== 检查已导入的模块 ===")
clip_modules = [name for name in sys.modules if 'clip' in name.lower()]
if clip_modules:
    print("包含'clip'的模块:")
    for mod in clip_modules:
        print(f"  {mod}")
else:
    print("没有发现包含'clip'的模块")

print(f"\n=== 验证是否经过jina-clip-implementation ===")
jina_clip_path = str(current_dir / "modules" / "transformers_modules" / "jinaai" / "jina-clip-implementation")
clip_in_path = any(jina_clip_path in p for p in sys.path)
print(f"jina-clip-implementation是否在搜索路径中: {clip_in_path}")

if clip_in_path:
    print("路径中包含jina-clip-implementation!")
else:
    print("路径中不包含jina-clip-implementation")
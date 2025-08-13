# Jina CLIP v1 Evaluation

这是一个 Jina CLIP 模型的评估和调试项目，包含了模型推理、配置分析和 LoRA 机制研究。

## 📁 项目结构

```
single_jina_v1_eval/
├── jina_clip_impl/          # Jina CLIP 核心实现
│   ├── __init__.py
│   ├── configuration_clip.py
│   ├── eva_model.py
│   ├── hf_model.py          # HuggingFace 模型封装
│   ├── modeling_clip.py     # 主模型实现
│   ├── processing_clip.py
│   ├── rope_embeddings.py
│   └── transform.py
├── jina_model/              # 模型权重和配置
│   ├── config.json          # 主配置文件
│   ├── pytorch_model.bin    # 模型权重 (1.7GB, 未上传)
│   ├── tokenizer.json       # 分词器
│   └── ...
├── jina_text_config/        # 文本模型配置
│   ├── config.json          # 包含 auto_map 配置
│   └── ...
├── hf_cache/               # HuggingFace 缓存文件
├── graph_infer.py          # 图推理脚本
└── eval_jina_short.py      # 评估脚本
```

## 🚀 快速开始

### 环境准备
```bash
pip install torch transformers pillow numpy
```

### 运行推理
```bash
export HF_ENDPOINT=https://hf-mirror.com  # 使用镜像源
python graph_infer.py
```

## 🔍 关键特性

### 1. LoRA 适配器支持
- 模型包含 196 个 LoRA 权重参数
- 支持多任务切换：`retrieval.query`, `retrieval.passage` 等
- 通过 `XLMRobertaLoRA` 包装器实现

### 2. 自动配置下载
- 通过 `auto_map` 配置自动下载远程实现
- 支持 Flash Attention 和 xFormers 优化
- 缓存管理和离线模式支持

### 3. 多语言文本相似度
内置多语言测试样本，支持：
- 阿拉伯语、中文、法语、德语
- 希腊语、印地语、意大利语
- 日语、韩语等

## 📊 模型配置

### 文本模型
- 基础模型：XLM-RoBERTa
- 维度：1024
- LoRA 秩：4
- LoRA α：4
- 支持任务：检索查询/文档

### 视觉模型  
- 基础模型：EVA ViT
- 图像尺寸：512×512
- 补丁大小：14×14
- 层数：24 层

## 🛠️ 调试工具

### 1. 配置追踪
```bash
python jina_model/trace_config.py
```

### 2. LoRA 权重检查
```bash
python jina_model/check_lora.py
```

### 3. 调用链分析
```bash
python jina_model/analyze_call_chain.py
```

## ⚠️ 注意事项

1. **模型文件未上传**：`pytorch_model.bin` (1.7GB) 需要单独下载
2. **网络配置**：首次运行需要联网下载远程实现文件
3. **缓存目录**：自动创建在 `hf_cache/` 下
4. **依赖要求**：需要 PyTorch 1.9+ 和 Transformers 4.20+

## 📈 性能表现

在多语言相似度任务上表现优异：
- 英语查询 "beautiful sunset over the beach"
- 德语匹配 "Ein wunderschöner Sonnenuntergang am Strand"  
- 相似度：0.924

## 🔗 相关资源

- [Jina AI](https://jina.ai/)
- [Transformers 文档](https://huggingface.co/docs/transformers/)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)

## 📝 开发日志

记录了完整的开发过程，包括：
- 配置解析机制分析
- LoRA 封装过程研究  
- PyTorch parametrize 机制理解
- HuggingFace auto_map 工作流程
#!/usr/bin/env python3
"""
精确追踪 AutoModel.from_config(config) 中 config 的来源
"""

def trace_config_source():
    print("=== 精确追踪 config 来源 ===\n")
    
    print("代码执行流程:")
    print("1. hf_model.py 第146行:")
    print("   self.config = AutoConfig.from_pretrained(")
    print("       model_name_or_path,  # 这是 '/root/autodl-fs/jina/single_jina_v1_eval/jina_text_config'")
    print("       trust_remote_code=trust_remote_code,")
    print("   )")
    print()
    
    print("2. hf_model.py 第151行:")
    print("   self.config.update(model_config_kwargs)")
    print("   # model_config_kwargs 来自主配置文件的 hf_model_config_kwargs")
    print()
    
    print("3. hf_model.py 第154行:")
    print("   self.transformer = AutoModel.from_config(")
    print("       self.config,  # ← 这就是我们要追踪的 config")
    print("   )")
    print()

def show_config_composition():
    print("=== config 对象的组成 ===\n")
    
    print("这个 config 对象是:")
    print("1. 基础来源: /jina_text_config/config.json")
    print("   - 通过 AutoConfig.from_pretrained('/jina_text_config') 读取")
    print("   - 返回 XLMRobertaFlashConfig 实例")
    print()
    
    print("2. 额外更新: model_config_kwargs")
    print("   - 来自 /jina_model/config.json 的 text_config.hf_model_config_kwargs")
    print("   - 通过 self.config.update(model_config_kwargs) 合并")
    print()
    
    print("3. 最终的 config 包含:")
    print("   - /jina_text_config/config.json 的所有内容")
    print("   - 加上 /jina_model/config.json 中 hf_model_config_kwargs 的覆盖")
    print()

def show_specific_config_content():
    print("=== 具体配置内容 ===\n")
    
    import json
    
    # 读取基础配置
    print("基础配置 (/jina_text_config/config.json):")
    with open("/root/autodl-fs/jina/single_jina_v1_eval/jina_text_config/config.json", "r") as f:
        text_config = json.load(f)
    
    print("  关键字段:")
    print(f"  - auto_map: {text_config.get('auto_map', {})}")
    print(f"  - lora_adaptations: {text_config.get('lora_adaptations', [])}")
    print(f"  - load_trained_adapters: {text_config.get('load_trained_adapters', False)}")
    print()
    
    # 读取覆盖配置
    print("覆盖配置 (/jina_model/config.json 的 hf_model_config_kwargs):")
    with open("/root/autodl-fs/jina/single_jina_v1_eval/jina_model/config.json", "r") as f:
        jina_config = json.load(f)
    
    hf_config_kwargs = jina_config["text_config"]["hf_model_config_kwargs"]
    print("  覆盖字段:")
    for key, value in hf_config_kwargs.items():
        print(f"  - {key}: {value}")
    print()
    
    print("最终 config 的关键特点:")
    print("  - 类型: XLMRobertaFlashConfig")
    print("  - auto_map 指向: modeling_lora.XLMRobertaLoRA")
    print("  - 包含完整的 LoRA 配置参数")
    print("  - load_trained_adapters 决定是否加载已训练的 LoRA")

def show_automodel_config_usage():
    print("\n=== AutoModel.from_config() 如何使用这个 config ===\n")
    
    print("AutoModel.from_config(config) 执行时:")
    print("1. 检查 config.auto_map['AutoModel']")
    print("   → 'jinaai/xlm-roberta-flash-implementation--modeling_lora.XLMRobertaLoRA'")
    print()
    print("2. 解析并下载 modeling_lora.py")
    print()
    print("3. 导入 XLMRobertaLoRA 类")
    print()
    print("4. 调用 XLMRobertaLoRA(config, add_pooling_layer=False)")
    print("   这里传入的 config 就是合并后的完整配置")
    print()
    print("5. XLMRobertaLoRA.__init__() 读取 config 中的:")
    print("   - config.lora_adaptations")
    print("   - config.lora_rank") 
    print("   - config.lora_alpha")
    print("   - config.load_trained_adapters")
    print("   - 等等...")

if __name__ == "__main__":
    trace_config_source()
    show_config_composition()
    show_specific_config_content()
    show_automodel_config_usage()
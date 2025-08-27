## 项目结构

```
split_jina/
├── image/                  # 图像编码器模型文件
├── text_encode/           # 文本编码器模型文件
├── eval/                  # 评估相关文件
├── infer.py              # 主推理脚本
├── infer_fixed.py        # 修复版推理脚本
├── test_import.py        # 导入测试脚本
├── requirements.txt      # 依赖包列表
└── README.md            # 本文档
```

## 安装配置


```bash
pip install -U -r requirements.txt
```


## 使用方法

```
python infer.py
```

## 输出示例

```
Similarity results:
En -> Img: 0.002523379
Img -> Img: 0.3029244
En -> Ar: 0.5410333
En -> Zh: 0.68158805
En -> Fr: 0.6232703
En -> De: 0.6322594
En -> Gr: 0.5915469
En -> Hi: 0.64402163
En -> It: 0.6029575
En -> Jp: 0.57573664
En -> Ko: 0.6145118
```


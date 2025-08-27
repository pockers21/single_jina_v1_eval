#!/usr/bin/env python3
"""
独立的图像编码器模块
从jina-clip-v2中提取的图像编码器实现
"""

import base64
import json
import os
import warnings
from functools import partial
from io import BytesIO
from typing import List, Optional, Union

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor

# 导入本地模块
from eva_model import EVAVisionTransformer
from configuration_clip import JinaCLIPVisionConfig
from transform import image_transform


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class VisionEncoder(nn.Module):
    """
    独立的图像编码器类
    """
    
    def __init__(self, config_path=None, weights_path=None, device='cuda'):
        super().__init__()
        self.device = device
        
        # 加载配置
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'vision_config.json')
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        self.config = JinaCLIPVisionConfig(**config_dict)
        
        # 构建vision模型
        self.vision_model = self._build_vision_tower(self.config)
        
        # 加载权重
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(__file__), 'vision_model_weights.bin')
        
        if os.path.exists(weights_path):
            print(f"Loading vision weights from: {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            self.vision_model.load_state_dict(state_dict)
        else:
            print(f"Warning: Vision weights not found at {weights_path}")
        
        # 移动到设备
        self.to(device)
        self.eval()
        
        # 初始化预处理器
        self.preprocess = None
    
    def _build_vision_tower(self, config: JinaCLIPVisionConfig) -> EVAVisionTransformer:
        """构建vision tower"""
        norm_layer = partial(LayerNorm, eps=1e-6)
        
        if config.fused_layer_norm:
            try:
                from apex.normalization import FusedLayerNorm
                norm_layer = partial(FusedLayerNorm, eps=1e-6)
            except (ModuleNotFoundError, ImportError):
                warnings.warn('Please install apex to use fused layer norm, ignoring')
        
        return EVAVisionTransformer(
            img_size=config.image_size,
            patch_size=config.patch_size,
            num_classes=config.embed_dim,
            use_mean_pooling=False,
            init_values=config.ls_init_value,
            patch_dropout=config.patch_dropout,
            embed_dim=config.width,
            depth=config.layers,
            num_heads=config.width // config.head_width,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_path_rate=0.0,
            norm_layer=norm_layer,
            xattn=config.x_attention,
            rope=config.rope_embeddings,
            postnorm=config.post_norm,
            pt_hw_seq_len=config.pt_hw_seq_len,
            intp_freq=config.intp_freq,
            naiveswiglu=config.naive_swiglu,
            subln=config.subln
        )
    
    def get_preprocess(self):
        """获取图像预处理器"""
        if self.preprocess is None:
            try:
                # 尝试从本地加载预处理器配置
                preprocessor_config_path = os.path.join(os.path.dirname(__file__), 'preprocessor_config.json')
                if os.path.exists(preprocessor_config_path):
                    self.preprocess = AutoImageProcessor.from_pretrained(
                        os.path.dirname(__file__), 
                        trust_remote_code=True
                    )
                else:
                    # 使用默认的图像变换
                    self.preprocess = image_transform(
                        image_size=self.config.image_size,
                        is_train=False
                    )
            except Exception as e:
                print(f"Warning: Could not load preprocessor, using default transform: {e}")
                self.preprocess = image_transform(
                    image_size=self.config.image_size,
                    is_train=False
                )
        return self.preprocess
    
    def _decode_image_data(self, data_url: str) -> Image.Image:
        """解码base64图像数据"""
        header, data = data_url.split(',', 1)
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data))
    
    def _preprocess_images(self, images: List[Union[str, Image.Image]]) -> torch.Tensor:
        """预处理图像"""
        processed_images = []
        
        for img in images:
            if isinstance(img, str):
                if img.startswith('http'):
                    response = requests.get(img)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                elif img.startswith('data:image/'):
                    image = self._decode_image_data(img).convert('RGB')
                else:
                    image = Image.open(img).convert('RGB')
            elif isinstance(img, Image.Image):
                image = img.convert('RGB')
            else:
                raise ValueError('Unsupported image format')
            
            processed_images.append(image)
        
        # 使用预处理器
        preprocess = self.get_preprocess()
        
        try:
            # 尝试作为HuggingFace预处理器使用
            if hasattr(preprocess, 'preprocess') or hasattr(preprocess, '__call__'):
                result = preprocess(processed_images, return_tensors='pt')
                if isinstance(result, dict) and 'pixel_values' in result:
                    pixel_values = result['pixel_values']
                else:
                    # 如果是函数式预处理器
                    pixel_values = torch.stack([preprocess(img) for img in processed_images])
            else:
                # 如果是函数式预处理器
                pixel_values = torch.stack([preprocess(img) for img in processed_images])
        except Exception as e:
            print(f"预处理器失败，使用默认变换: {e}")
            # 使用默认的图像变换
            from transform import image_transform
            default_transform = image_transform(
                image_size=self.config.image_size,
                is_train=False
            )
            pixel_values = torch.stack([default_transform(img) for img in processed_images])
        
        return pixel_values
    
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """获取图像特征"""
        return self.vision_model(x=pixel_values)
    
    def _truncate_embeddings(self, embeddings: torch.Tensor, truncate_dim: int) -> torch.Tensor:
        """截断嵌入维度"""
        return embeddings[:, :truncate_dim]
    
    @torch.inference_mode()
    def encode_image(
        self,
        images: Union[str, List[Union[str, Image.Image]]],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True,
        truncate_dim: Optional[int] = None,
        **kwargs
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        编码图像为嵌入向量
        
        Args:
            images: 图像路径、URL、PIL图像或base64字符串
            batch_size: 批处理大小
            convert_to_numpy: 是否转换为numpy数组
            convert_to_tensor: 是否返回单个tensor
            normalize_embeddings: 是否归一化嵌入
            truncate_dim: 截断维度
            **kwargs: 其他参数
        
        Returns:
            图像嵌入向量
        """
        _is_training = self.training
        self.eval()
        
        if convert_to_tensor:
            convert_to_numpy = False
        
        _input_was_single_img = False
        if isinstance(images, str) or not hasattr(images, '__len__'):
            images = [images]
            _input_was_single_img = True
        
        all_embeddings = []
        
        # 按长度排序以优化批处理
        _permutation = np.argsort([-len(str(i)) for i in images])
        _inverse_permutation = np.argsort(_permutation)
        images = [images[idx] for idx in _permutation]
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # 预处理图像
            pixel_values = self._preprocess_images(batch_images)
            pixel_values = pixel_values.to(self.device)
            
            # 获取图像特征
            embeddings = self.get_image_features(pixel_values)
            
            # 截断维度
            if truncate_dim:
                embeddings = self._truncate_embeddings(embeddings, truncate_dim)
            
            # 归一化
            if normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # 转换为CPU
            if convert_to_numpy:
                embeddings = embeddings.cpu()
            
            all_embeddings.extend(embeddings)
        
        # 恢复原始顺序
        all_embeddings = [all_embeddings[idx] for idx in _inverse_permutation]
        
        # 转换输出格式
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray(
                [emb.to(torch.float32).numpy() for emb in all_embeddings]
            )
        
        if _input_was_single_img:
            all_embeddings = all_embeddings[0]
        
        self.train(_is_training)
        return all_embeddings


class ImageEncoderWrapper:
    """
    图像编码器包装类，兼容原有接口
    """
    
    def __init__(self, model_path=None, device='cuda'):
        """
        初始化图像编码器
        
        Args:
            model_path: 模型路径（现在使用本地image目录）
            device: 设备
        """
        self.device = device
        
        # 使用本地image目录
        if model_path is None:
            model_path = os.path.dirname(__file__)
        
        self.model = VisionEncoder(
            config_path=os.path.join(model_path, 'vision_config.json'),
            weights_path=os.path.join(model_path, 'vision_model_weights.bin'),
            device=device
        )
    
    def encode_image(self, images, truncate_dim=None, **kwargs):
        """
        编码图像
        
        Args:
            images: 图像输入
            truncate_dim: 截断维度
            **kwargs: 其他参数
        
        Returns:
            torch.Tensor: 图像嵌入
        """
        try:
            with torch.no_grad():
                embeddings = self.model.encode_image(
                    images, 
                    truncate_dim=truncate_dim, 
                    convert_to_tensor=True,
                    convert_to_numpy=False,
                    **kwargs
                )
                
                # 确保返回torch.Tensor并在正确的设备上
                if isinstance(embeddings, np.ndarray):
                    embeddings = torch.from_numpy(embeddings).to(self.device)
                elif not isinstance(embeddings, torch.Tensor):
                    embeddings = torch.tensor(embeddings).to(self.device)
                elif embeddings.device != torch.device(self.device):
                    embeddings = embeddings.to(self.device)
                
                return embeddings
        except Exception as e:
            print(f"❌ 图像编码失败: {e}")
            raise e


def load_image_encoder(model_path=None, device='cuda'):
    """
    加载图像编码器
    
    Args:
        model_path: 模型路径（默认使用当前目录）
        device: 设备
    
    Returns:
        ImageEncoderWrapper: 图像编码器包装器
    """
    if model_path is None:
        model_path = os.path.dirname(__file__)
    
    return ImageEncoderWrapper(model_path, device)
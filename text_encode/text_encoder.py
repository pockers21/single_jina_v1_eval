import json
import math
import os
import re
from functools import partial
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
import torch.nn.utils.parametrize as parametrize
from torch.nn import Parameter
from transformers import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)

from configuration_xlm_roberta import XLMRobertaFlashConfig
from modeling_xlm_roberta import (
    XLMRobertaFlashConfig,
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)


_POOLERS = {}


def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


def register_pooler(cls):
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    @staticmethod
    def forward(x: BaseModelOutput, attention_mask: torch.Tensor):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class MaxPooler(nn.Module):
    @staticmethod
    def forward(x: BaseModelOutput, attention_mask: torch.Tensor):
        masked_output = x.last_hidden_state.masked_fill(
            attention_mask.unsqueeze(-1), -torch.inf
        )
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    def __init__(self, use_pooler_output: bool = True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, _: torch.Tensor):
        if (
            self.use_pooler_output
            and isinstance(
                x,
                (
                    BaseModelOutputWithPooling,
                    BaseModelOutputWithPoolingAndCrossAttentions,
                ),
            )
            and (x.pooler_output is not None)
        ):
            return x.pooler_output
        return x.last_hidden_state[:, self.cls_token_position, :]


def initialized_weights(
    shape: Tuple[int], num_adaptations: int, init: str = "kaiming"
) -> torch.Tensor:
    weight_data = []
    for _ in range(num_adaptations):
        new_adaption = torch.zeros(shape)
        if init == "kaiming":
            nn.init.kaiming_uniform_(new_adaption, a=math.sqrt(5))
        elif init == "normal":
            nn.init.normal_(new_adaption)
        else:
            raise NotImplementedError
        weight_data.append(new_adaption)
    return torch.stack(weight_data, dim=0)


class LoRAParametrization(nn.Module):
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        layer_type: str = "linear",
        num_adaptations: int = 1,
        rank: int = 4,
        dropout_p: float = 0.0,
        alpha: float = 1,
    ):
        super().__init__()
        fan_in_fan_out = layer_type == "embedding"
        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)

        if layer_type == "linear":
            self.lora_A = nn.Parameter(
                initialized_weights((rank, fan_in), num_adaptations, init="kaiming")
            )
            self.lora_B = nn.Parameter(torch.zeros((num_adaptations, fan_out, rank)))
        elif layer_type == "embedding":
            self.lora_A = nn.Parameter(torch.zeros((num_adaptations, fan_in, rank)))
            self.lora_B = nn.Parameter(
                initialized_weights(
                    (rank, fan_out), num_adaptations=num_adaptations, init="normal"
                )
            )
        else:
            raise NotImplementedError

        self.lora_alpha, self.rank = alpha, rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if dropout_p > 0 else lambda x: x
        self.register_buffer(
            "lora_dropout_mask",
            torch.ones(self.swap((1, fan_in)), dtype=self.lora_A.dtype),
            persistent=False,
        )

    def _dropout(self, A):
        return A * self.lora_dropout(self.lora_dropout_mask)

    def lora_forward(self, X, current_task):
        return (
            X
            + torch.matmul(
                *self.swap(
                    (
                        self.lora_B[current_task],
                        self.dropout_fn(self.lora_A[current_task]),
                    )
                )
            ).view(X.shape)
            * self.scaling
        )

    def forward(self, X):
        return X

    @classmethod
    def from_linear(
        cls,
        layer: nn.Module,
        num_adaptations: int,
        rank: int,
        dropout_p: float,
        alpha: float,
    ):
        assert isinstance(layer, nn.Linear)
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in,
            fan_out,
            num_adaptations=num_adaptations,
            layer_type="linear",
            rank=rank,
            dropout_p=dropout_p,
            alpha=alpha,
        )

    @classmethod
    def from_embedding(
        cls,
        layer: nn.Module,
        num_adaptations: int,
        rank: int,
        dropout_p: float,
        alpha: float,
    ):
        assert isinstance(layer, nn.Embedding)
        fan_in, fan_out = layer.weight.shape
        return cls(
            fan_in,
            fan_out,
            num_adaptations=num_adaptations,
            layer_type="embedding",
            rank=rank,
            dropout_p=dropout_p,
            alpha=alpha,
        )

    @classmethod
    def add_to_layer(
        cls,
        layer: nn.Module,
        num_adaptations: int,
        rank: int,
        dropout_p: float,
        alpha: float,
    ):
        if isinstance(layer, nn.Linear):
            parametrize.register_parametrization(
                layer,
                "weight",
                cls.from_linear(
                    layer,
                    num_adaptations=num_adaptations,
                    rank=rank,
                    dropout_p=dropout_p,
                    alpha=alpha,
                ),
            )

            def new_forward(self, input, task_id=None, residual=False):
                if task_id is not None:
                    weights = self.parametrizations.weight[0].lora_forward(
                        self.weight, current_task=task_id
                    )
                else:
                    weights = self.weight

                out = F.linear(input, weights, self.bias)

                if residual:
                    return out, input
                return out

            layer.forward = new_forward.__get__(layer, layer.__class__)

        elif isinstance(layer, nn.Embedding):
            parametrize.register_parametrization(
                layer,
                "weight",
                cls.from_embedding(
                    layer,
                    num_adaptations=num_adaptations,
                    rank=rank,
                    dropout_p=dropout_p,
                    alpha=alpha,
                ),
            )

            def new_forward(self, input, task_id=None):
                if task_id is not None:
                    weights = self.parametrizations.weight[0].lora_forward(
                        self.weight, current_task=task_id
                    )
                else:
                    weights = self.weight

                out = F.embedding(
                    input,
                    weights,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )

                return out

            layer.forward = new_forward.__get__(layer, layer.__class__)


class XLMRobertaLoRA(XLMRobertaPreTrainedModel):
    def __init__(
        self,
        config: XLMRobertaFlashConfig,
        roberta: Optional[XLMRobertaModel] = None,
        add_pooling_layer: bool = True,
    ):
        super().__init__(config)
        if roberta is None:
            self.roberta = XLMRobertaModel(config, add_pooling_layer=add_pooling_layer)
        else:
            self.roberta = roberta
        self.main_params_trainable = False

    @property
    def rotary_emb_base(self):
        return self.roberta.rotary_emb_base

    @rotary_emb_base.setter
    def rotary_emb_base(self, base):
        self.roberta.rotary_emb_base = base

    @property
    def main_params_trainable(self):
        return self._main_params_trainable

    @main_params_trainable.setter
    def main_params_trainable(self, val: bool):
        self._main_params_trainable = val
        for name, param in super().named_parameters():
            if "lora" not in name:
                param.requires_grad_(val)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        for key in list(kwargs.keys()):
            if key in config.to_dict():
                config.update({key: kwargs.pop(key)})
        if config.load_trained_adapters:
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        else:
            roberta = XLMRobertaModel.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                use_flash_attn=config.use_flash_attn,
                **kwargs,
            )
            return cls(config, roberta=roberta)

    def _register_lora(self, num_adaptations, rank, dropout_p, alpha):
        self.apply(
            partial(
                LoRAParametrization.add_to_layer,
                num_adaptations=num_adaptations,
                rank=rank,
                dropout_p=dropout_p,
                alpha=alpha,
            )
        )

    def forward(self, *args, **kwargs):
        return self.roberta(*args, **kwargs)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        for name, param in super().named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        ):
            if "lora" in name or self.main_params_trainable:
                yield name, param

    @torch.inference_mode()
    def encode(
        self,
        sentences: Union[str, List[str]],
        *args,
        task: Optional[str] = None,
        **kwargs,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        if task and task not in self._lora_adaptations:
            raise ValueError(
                f"Unsupported task '{task}'. "
                f"Supported tasks are: {', '.join(self.config.lora_adaptations)}."
                f"Alternatively, don't pass the `task` argument to disable LoRA."
            )
        adapter_mask = None
        if task:
            task_id = self._adaptation_map[task]
            num_examples = 1 if isinstance(sentences, str) else len(sentences)
            adapter_mask = torch.full(
                (num_examples,), task_id, dtype=torch.int32, device=self.device
            )
            if isinstance(sentences, str):
                sentences = self._task_instructions[task] + sentences
            else:
                sentences = [
                    self._task_instructions[task] + sentence for sentence in sentences
                ]
        return self.roberta.encode(
            sentences, *args, adapter_mask=adapter_mask, **kwargs
        )


class TextEncoderWrapper:
    def __init__(self, model, tokenizer, device='cuda', config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
        if config is None:
            config = {
                "model_type": "jina_clip_text",
                "pooler_type": "mean_pooler",
                "proj_bias": False,
                "proj_type": None,
                "output_tokens": False
            }
        self.config = config
        
        pooler_type = config.get("pooler_type", "mean_pooler")
        self.pooler = _POOLERS[pooler_type]()
        
        d_model = getattr(model.config, 'hidden_size', 768)
        output_dim = d_model
        proj_type = config.get("proj_type")
        proj_bias = config.get("proj_bias", False)
        
        if (d_model == output_dim) and (proj_type is None):
            self.proj = nn.Identity()
        elif (d_model != output_dim) or proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=proj_bias)
        elif proj_type == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=proj_bias),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=proj_bias),
            )
        else:
            self.proj = nn.Identity()
            
        self.output_tokens = config.get("output_tokens", False)
        self.proj.to(device)
        
    def encode_text(self, sentences, task='retrieval.passage', truncate_dim=None, normalize_embeddings=True, convert_to_numpy=True, convert_to_tensor=False, **kwargs):
        _is_training = self.model.training
        self.model.eval()
        
        if isinstance(sentences, str):
            sentences = [sentences]
            single_input = True
        else:
            single_input = False
            
        if convert_to_tensor:
            convert_to_numpy = False
            
        if task == 'retrieval.query':
            instruction = "Represent the query for retrieving evidence documents: "
            sentences = [instruction + sentence for sentence in sentences]
            
        import numpy as np
        _permutation = np.argsort([-len(i) for i in sentences])
        _inverse_permutation = np.argsort(_permutation)
        sentences = [sentences[idx] for idx in _permutation]
            
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = self.pooler(outputs, attention_mask)
            embeddings = self.proj(pooled_output)
            
            if truncate_dim is not None and truncate_dim < embeddings.size(1):
                embeddings = embeddings[:, :truncate_dim]
                
            if normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
        if _is_training:
            self.model.train()
            
        if not single_input and len(sentences) > 1:
            embeddings_list = [embeddings[i] for i in range(embeddings.size(0))]
            embeddings_list = [embeddings_list[idx] for idx in _inverse_permutation]
            embeddings = torch.stack(embeddings_list)
            
        if convert_to_numpy:
            embeddings = embeddings.cpu()
            if single_input:
                embeddings = embeddings.squeeze(0).to(torch.float32).numpy()
            else:
                embeddings = np.asarray([emb.to(torch.float32).numpy() for emb in embeddings])
        elif convert_to_tensor:
            if single_input:
                embeddings = embeddings.squeeze(0)
        else:
            if single_input:
                embeddings = embeddings.squeeze(0)
                
        return embeddings


def load_text_encoder_with_tokenizer(model_path=None, device='cuda', config=None):
    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = current_dir
    
    print(f'model_path: {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    cfg = XLMRobertaFlashConfig()
    cfg.name_or_path = model_path
    
    model = XLMRobertaLoRA._from_config(cfg, add_pooling_layer=False)
    model.eval()
    
    weights_path = os.path.join(model_path, 'pytorch_model.bin')
    if os.path.exists(weights_path):
        state = torch.load(weights_path, weights_only=True, map_location='cpu')
        model.load_state_dict(state)
    else:
        weights_path = os.path.join(model_path, 'text_encoder_weights.bin')
        if os.path.exists(weights_path):
            state = torch.load(weights_path, weights_only=True, map_location='cpu')
            model.load_state_dict(state)
        else:
            print(f"Warning: Could not find weights file at {weights_path}")
    
    text_encoder = TextEncoderWrapper(model, tokenizer, device, config)
    
    return text_encoder
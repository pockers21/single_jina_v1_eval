import json
import math
import os
from functools import partial
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from transformers import PretrainedConfig
from pathlib import Path
import os
#os.environ['HF_HOME'] = "/Users/yr/Desktop/notebook1/code/jinaclip_arch/modelClss1_加载模型/HF_cache_raw"
import sys
# 获取当前文件的目录
current_dir = Path(__file__).resolve().parent

# 将当前文件的目录添加到sys.path
sys.path.insert(0, str(current_dir))

# 添加hub目录下的模型实现路径
hub_model_path = current_dir / "hub" / "models--jinaai--xlm-roberta-flash-implementation" / "snapshots" / "2b6bc3f30750b3a9648fe9b63448c09920efe9be"
sys.path.insert(0, str(hub_model_path))

# 添加modules目录下的路径
modules_path = current_dir / "modules" / "transformers_modules" / "jinaai" / "xlm-roberta-flash-implementation"
sys.path.insert(0, str(modules_path))

from configuration_xlm_roberta import XLMRobertaFlashConfig
from modeling_xlm_roberta import (
    XLMRobertaFlashConfig,
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)


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
    """
    This LoRA implementation was inspired by  https://github.com/cccntu/minLoRA
    The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """

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
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
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
        self.scaling = alpha / rank #1
        self.lora_dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if dropout_p > 0 else lambda x: x# p=0
        self.register_buffer(
            "lora_dropout_mask",
            torch.ones(self.swap((1, fan_in)), dtype=self.lora_A.dtype),
            persistent=False,
        )

    def _dropout(self, A):
        # to mimic the original implementation: A @ dropout(x), we do (A * dropout(ones)) @ x
        return A * self.lora_dropout(self.lora_dropout_mask)

    def lora_forward(self, X, current_task): ### w+AB  或 w+BA
        print(100,current_task) ## 什么时候调佣weight?
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
        """
        Registering LoRA adapters to all embedding and linear layers.
        Additionally, we implement a custom forward function for LoRA parametrization.
        This function modifies the layer's forward pass to optionally use task-specific
        parameters. When a `task_id` is provided, it employs a LoRA parametrization
        to modify the original weights according to the specific task. This allows
        the layer to adapt dynamically to different tasks at runtime. If no `task_id`
        is specified, the layer uses its original weights.
        """
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

            def new_forward(self, input, task_id=None, residual=False):#[1  8 1024], taskid=0, true

                if task_id is not None:
                    #
                    weights = self.parametrizations.weight[0].lora_forward(
                        self.weight, current_task=task_id
                    )
                else:
                    weights = self.weight

                out = F.linear(input, weights, self.bias)# inp w[3072 1024]

                if residual:# here
                    return out, input # [1 8 3072] [1 8 1024]
                return out

            layer.forward = new_forward.__get__(layer, layer.__class__)
            tmp=[new_forward,layer,layer.__class__]

        elif isinstance(layer, nn.Embedding):
            parametrize.register_parametrization(
                layer,
                "weight",
                cls.from_embedding(
                    layer,
                    num_adaptations=num_adaptations,
                    rank=rank,#4
                    dropout_p=dropout_p,#0
                    alpha=alpha,#4
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
                    input, # [1 8] t
                    weights, # word_emb[25000 1024]  token_type_id[1 1024]
                    self.padding_idx, # 1   none
                    self.max_norm, # none none
                    self.norm_type,# 2  2
                    self.scale_grad_by_freq, # false
                    self.sparse, # false
                )

                return out # [1  8 1024]

            layer.forward = new_forward.__get__(layer, layer.__class__)


class XLMRobertaLoRA(XLMRobertaPreTrainedModel):
    """
    A wrapper class around the Jina XLM-RoBERTa model that integrates LoRA (Low-Rank Adaptation) adapters.
    """

    def __init__(
        self,
        config: XLMRobertaFlashConfig,
        roberta: Optional[XLMRobertaModel] = None, ## none
        add_pooling_layer: bool = True, # false
    ):
        #config.num_hidden_layers=1 ####yr
        super().__init__(config)
        if roberta is None:
            print(config)
            self.roberta = XLMRobertaModel(config, add_pooling_layer=add_pooling_layer)
        else:# not here
            self.roberta = roberta
        if False:
            self._lora_adaptations = config.lora_adaptations
            if (
                not isinstance(self._lora_adaptations, list)
                or len(self._lora_adaptations) < 1
            ):
                raise ValueError(
                    f"`lora_adaptations` must be a list and contain at least one element"
                )
            self._task_instructions = config.task_instructions
            if (
                not isinstance(self._task_instructions, dict)
                or len(self._task_instructions) != len(self._lora_adaptations)
                or not all(
                    [v in self._lora_adaptations for v in self._task_instructions.keys()]
                )
            ):
                raise ValueError(
                    f"`task_instructions` must be a dict and contain the same number of elements "
                    f"as `lora_adaptations` with all keys in `task_instructions` present in `lora_adaptations`."
                )
            self._adaptation_map = {
                name: idx for idx, name in enumerate(self._lora_adaptations)
            }
            self._rank = config.lora_rank
            self._dropout_p = config.lora_dropout_p
            self._alpha = config.lora_alpha
            self._register_lora(
                num_adaptations=len(self._lora_adaptations),
                rank=self._rank,
                dropout_p=self._dropout_p,
                alpha=self._alpha,
            )
            self.main_params_trainable = config.lora_main_params_trainable

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
        """Whether the main parameters (i.e. those that are not LoRA) should be trainable.
        This method sets the `requires_grad_` attribute of the main weights
        and controls which parameters are returned in `self.parameters()`.
        :param val: Whether or not to make the parameters trainable.
        :return: None
        """
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
        if config.load_trained_adapters:  # checkpoint already contains LoRA adapters
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
        else:  # initializing new adapters
            roberta = XLMRobertaModel.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                use_flash_attn=config.use_flash_attn,
                **kwargs,
            )
            return cls(config, roberta=roberta)

    def _register_lora(self, num_adaptations, rank, dropout_p, alpha):
        self.apply( ### each module 应用这个方法
            partial(
                LoRAParametrization.add_to_layer,
                num_adaptations=num_adaptations,
                rank=rank,
                dropout_p=dropout_p,
                alpha=alpha,
            )
        )

    def forward(self, *args, **kwargs):
        print()#

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
        """
        Computes sentence embeddings.
        sentences(`str` or `List[str]`):
            Sentence or sentences to be encoded
        task(`str`, *optional*, defaults to `None`):
            Specifies the task for which the encoding is intended. If `task` is not provided,
            all LoRA adapters are disabled, and the model reverts to its original,
            general-purpose weights.
        """
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


def dumInp():
    args = ()
    k = {}
    k['input_ids'] = torch.tensor( [[0,      6, 229845,    575, 125426,    635,   6656,      2]],dtype=torch.int32)
    k['attention_mask'] = torch.ones(size=[1, 8])
    k['adapter_mask'] = torch.tensor([0], dtype=torch.int32)  # shape[1,]
    # for _,v in k.items():
    #     print (v.shape)
    return args,k


def see_stateDict(model):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def load_merged_lora_weight(model):
    PATH='/root/autodl-tmp/jina/pytorch_model_mergeLoraPartcorrect.bin'
    state=torch.load(PATH, weights_only=True)
    
    print('load merge lora weight...')
    state1={}
    for k in state:
        state1[k.replace('text_model.transformer.','')]=state[k]
        print(k.replace('text_model.transformer.',''))
    model.load_state_dict(state1)

    return model.eval()


def simple_test():
    torch.manual_seed(123)

    # cfg=XLMRobertaFlashConfig.from_pretrained('/Users/yr/Desktop/notebook1/download_model/jina_emb_v3')#不行,有所有task
    cfg = XLMRobertaFlashConfig()
    #f1 = open('/Users/yr/Desktop/notebook1/code/jinaclip_arch/robertLora_diff.json')
    #cfg.update(json.load(f1))

    #cfg.num_hidden_layers = 1  ###用一层 验证计算结果
    cfg.name_or_path = '/autodl-fs/data/jina/HF_cache_raw/jina-embeddings-v3'
    modelclass = XLMRobertaLoRA
    # model=XLMRobertaLoRA(config=cfg,add_pooling_layer=False)#丢了目录 autoToken
    kwargs = {'add_pooling_layer': False}
    model = modelclass._from_config(cfg, **kwargs);
    model.eval()##影响dropout

    print(f'model: {model}')
    
    if False:
        print('model class has key...',model)
        see_stateDict(model)
    if 1:###
        model = load_merged_lora_weight(model)
        print('modelling_lora.py',model)
    args, k = dumInp()
    o = model(*args, **k)
    import pandas as pd

    lhs = o.last_hidden_state
    print(lhs[0, 0, :5])## 1.4098, -2.0369, -2.0049, -0.0043, -0.4095
    #pd.to_pickle(lhs.detach().numpy(), '/Users/yr/Desktop/notebook1/code/jinaclip_arch/varifyTXT.pkl')




def separate_tower_test():
    from transformers import AutoModel
    
    print("Loading custom text model...")
    torch.manual_seed(123)
    cfg = XLMRobertaFlashConfig()
    cfg.name_or_path = '/autodl-fs/data/jina/HF_cache_raw/jina-embeddings-v3'
    text_model = XLMRobertaLoRA._from_config(cfg, add_pooling_layer=False)
    text_model.eval()
    
    PATH = '/root/autodl-tmp/jina/pytorch_model_mergeLoraPartcorrect.bin'
    merged_text_weights = torch.load(PATH, map_location='cpu')
    
    state_dict_mapped = {}
    for k, v in merged_text_weights.items():
        new_key = k.replace('text_model.transformer.', 'roberta.')
        state_dict_mapped[new_key] = v
    
    text_model.load_state_dict(state_dict_mapped, strict=False)
    print("Custom text model loaded successfully!")
    
    print("Loading full model for image tower...")
    full_model = AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)
    full_model.eval()
    print("Image tower loaded successfully!")
    
    sentences = [
        'غروب جميل على الشاطئ',
        '海滩上美丽的日落',
        'Un beau coucher de soleil sur la plage',
        'Ein wunderschöner Sonnenuntergang am Strand',
        'Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία',
        'समुद्र तट पर एक खूबसूरत सूर्यास्त',
        'Un bellissimo tramonto sulla spiaggia',
        '浜辺に沈む美しい夕日',
        '해변 위로 아름다운 일몰',
    ]
    
    image_urls = [
        '/root/llama.cpp-clip/tools/mtmd/test-1.jpeg',
        '/root/llama.cpp-clip/media/llama1-logo.png'
    ]
    
    truncate_dim = 512
    
    print("Encoding text with custom model...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3')
        
        inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = text_model(**inputs)
            print(f"Model output type: {type(outputs)}")
            print(f"Model output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'No keys'}")
            
            if hasattr(outputs, 'last_hidden_state'):
                text_embeddings = outputs.last_hidden_state.mean(dim=1)
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                text_embeddings = outputs.hidden_states[-1].mean(dim=1)
            else:
                text_embeddings = outputs.mean(dim=1) if isinstance(outputs, torch.Tensor) else outputs
                
            if truncate_dim and hasattr(text_embeddings, 'shape') and len(text_embeddings.shape) > 1:
                text_embeddings = text_embeddings[:, :truncate_dim]
        
        print(f"Text encoding successful! Shape: {text_embeddings.shape}")
    except Exception as e:
        print(f"Text encoding failed: {e}")
        return
    
    print("Encoding images with original model...")
    try:
        image_embeddings = full_model.encode_image(image_urls, truncate_dim=truncate_dim)
        print(f"Image encoding successful! Shape: {image_embeddings.shape}")
    except Exception as e:
        print(f"Image encoding failed: {e}")
        return
    
    print("Encoding query with custom model...")
    query = 'beautiful sunset over the beach'
    try:
        query_inputs = tokenizer([query], return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            query_outputs = text_model(**query_inputs)
            if hasattr(query_outputs, 'last_hidden_state'):
                query_embeddings = query_outputs.last_hidden_state.mean(dim=1)
            elif hasattr(query_outputs, 'hidden_states') and query_outputs.hidden_states is not None:
                query_embeddings = query_outputs.hidden_states[-1].mean(dim=1)
            else:
                query_embeddings = query_outputs.mean(dim=1) if isinstance(query_outputs, torch.Tensor) else query_outputs
                
            if truncate_dim and hasattr(query_embeddings, 'shape') and len(query_embeddings.shape) > 1:
                query_embeddings = query_embeddings[:, :truncate_dim]
        
        print(f"Query encoding successful! Shape: {query_embeddings.shape}")
    except Exception as e:
        print(f"Query encoding failed: {e}")
        return
    
    print("\nSimilarity results:")
    print('En -> Img: ' + str(query_embeddings @ image_embeddings[0].T))
    print('Img -> Img: ' + str(image_embeddings[0] @ image_embeddings[1].T))
    print('En -> Ar: ' + str(query_embeddings @ text_embeddings[0].T))
    print('En -> Zh: ' + str(query_embeddings @ text_embeddings[1].T))
    print('En -> Fr: ' + str(query_embeddings @ text_embeddings[2].T))
    print('En -> De: ' + str(query_embeddings @ text_embeddings[3].T))
    print('En -> Gr: ' + str(query_embeddings @ text_embeddings[4].T))
    print('En -> Hi: ' + str(query_embeddings @ text_embeddings[5].T))
    print('En -> It: ' + str(query_embeddings @ text_embeddings[6].T))
    print('En -> Jp: ' + str(query_embeddings @ text_embeddings[7].T))
    print('En -> Ko: ' + str(query_embeddings @ text_embeddings[8].T))
    
    print("\nSeparate tower test completed!")
    return text_model, full_model


if __name__=='__main__':
    separate_tower_test()


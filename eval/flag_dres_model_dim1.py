from typing import cast, List, Dict, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, is_torch_npu_available
dim1=512//2
dim1=1024
dim1=32
dim1=512
from sentence_transformers import SentenceTransformer

class FlagDRESModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            batch_size: int = 256,
        dim1:int=1024,
        task:str=None,
        maxlen1:int=512
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        #self.model = AutoModel.from_pretrained(model_name_or_path,trust_remote_code=True)
        self.model =SentenceTransformer(model_name_or_path, trust_remote_code=True,truncate_dim=dim1) 
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method
        self.batch_size = batch_size
        self.task_str=task # for lora
        print('max len',self.model.max_seq_length)  ## 8194 jina 
        self.model.max_seq_length=maxlen1
        
        
        
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.batch_size = self.batch_size * num_gpus


    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if self.query_instruction_for_retrieval is not None:
            input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(input_texts)


    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        return self.encode(input_texts)


    @torch.no_grad()
    def encode(self, sentences: List[str], **kwargs ) -> np.ndarray:
        self.model.eval()
        

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches", disable=len(sentences)<256):
            sentences_batch = sentences[start_index:start_index + self.batch_size]
            # inputs = self.tokenizer(
            #     sentences_batch,
            #     padding=True,
            #     truncation=True,
            #     return_tensors='pt',
            #     max_length=512,
            # ).to(self.device)
            # last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            # embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
            # if self.normalize_embeddings:
            #     embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            # embeddings = cast(torch.Tensor, embeddings)
            # #print(83,embeddings.dtype)
            # if embeddings.dtype == torch.bfloat16:
            #     embeddings = embeddings.float()
            #     #embeddings = embeddings.numpy()
            embeddings = self.model.encode(#['Sample text'],
                                        sentences_batch,
                                      #truncate_dim=dim1,
                                        task=self.task_str
            )
            #all_embeddings.append(embeddings.cpu().numpy())
            #print('98 emb shape',embeddings.shape) # [len dim512]
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)
        #return all_embeddings

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor=None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d



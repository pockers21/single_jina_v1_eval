import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate(model, dataloader, tokenizer,  device, amp=True, recall_k_list=[5]):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    recall_k_list: list of int
        recall@k k's to use
    
    Returns
    -------
    
    dict of retrieval metrics
    """
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)
    autocast = torch.cuda.amp.autocast if amp else suppress
    for batch_images, batch_texts, inds in tqdm(dataloader):
        batch_images = batch_images.to(device)
        # tokenize all texts in the batch
        batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        # store the index of image for each text
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)
            batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok), dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)
        
    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    # get the score for each text and image pair
    scores  = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()

    return metrics

def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end

def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)
import json
from torch.utils.data import Dataset




class jsonDataset_flck8(Dataset):  ### return [txt,imgpath]
    def __init__(self,
                 # transforms, img_key, caption_key, sep="\t", tokenizer=None
                 ):
        f=open(datapath1)
        img_cap=json.load(f)
        ### img-txt
        ll=[]
        for k,v in img_cap.items():
            k=imgdir+k
            ll.append([k,v])
        print('total data',len(ll))
        logging.debug(f'Done loading data.{len(ll)}')

        ###

        self.bsz=8
        # self.tokenize = tokenizer
        self.data = ll
        self.indx=list(range(len(self.data))[::self.bsz])[:maxsee]




    def __len__(self):
        return len(self.indx)

    def __getitem__(self, idx):
        # images = self.transforms(Image.open(str(self.images[idx])))
        # texts = self.tokenize([str(self.captions[idx])])[0]
        # return images, texts
        idx1=self.indx[idx]
        batch=self.data[idx1:idx1+self.bsz]
        return [e[0] for e in batch],[e[1] for e in batch] ### img | txt

    
    
import argparse  
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_name_or_path', default=None, type=str)

    parser.add_argument('-d','--dim', default=None, type=str) #### 
    parser.add_argument('-o','--outname', default=None, type=str) ####
    
    return parser.parse_args()
if __name__=='__main__':
    args = get_args()
    maxsee=800
    #truncate_dim = 1024
    truncate_dim=int(args.dim); 
    imgdir='/root/autodl-tmp/download_data_flick8k/down_data_flick8k/Images/'
    datapath1='/root/autodl-tmp/download_data_flick8k/img_cap_flick8_zh.json'
    ## load data,model | emb | score retrieval
    import os

    # 设置 HF_HOME 环境变量
    os.environ['HF_HOME'] = "/root/autodl-tmp/tmp_HF_cache"

    from transformers import AutoModel
    import torch
    ### 模型路径
    #mpath = '/root/autodl-tmp/downloaded/jinaclipV2'
    mpath=args.model_name_or_path
    # Initialize the model
    device = 'cuda'
    model = AutoModel.from_pretrained(mpath,
                                      trust_remote_code=True
                                      )
    model = model.to(device)
    # Choose a matryoshka dimension, set to None to get the full 1024-dim vectors
     
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    ds=jsonDataset_flck8()
    for images_path, sentences in ds:
        with torch.no_grad():
            # Encode text and images
            text_embeddings = model.encode_text(sentences, truncate_dim=truncate_dim,
                                                convert_to_tensor=True
                                                )
            image_embeddings = model.encode_image(
                images_path, truncate_dim=truncate_dim, convert_to_tensor=True
            )  # also accepts PIL.Image.Image, local filenames, dataURI

            print('txt emb', text_embeddings.shape, text_embeddings.requires_grad)
            print('img emb', image_embeddings.shape, image_embeddings.requires_grad)

            batch_images_emb = F.normalize(image_embeddings, dim=-1)
            batch_texts_emb = F.normalize(text_embeddings, dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())


    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    # get the score for each text and image pair
    scores = texts_emb @ images_emb.t()

    positive_pairs = torch.zeros_like(scores, dtype=bool)
    #positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    positive_pairs[torch.arange(len(scores)), torch.arange(len(scores))] = True
    metrics = {}
    recall_k_list=[1,5]
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (
                    batchify(recall_at_k, scores, positive_pairs, batch_size, device,
                             k=recall_k) > 0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (
                    batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device,
                             k=recall_k) > 0).float().mean().item()

    print(metrics)
    name='jinaclip2_flick8k_zh_'+args.outname
    wr=open(f'./zh_result_jinaclip/{name}_dim{truncate_dim}_sample{maxsee}rst.json','w')
    import json
    json.dump(metrics,wr)
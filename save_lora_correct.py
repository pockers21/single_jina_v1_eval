import torch

ckpt='/root/autodl-tmp/jina/pytorch_model.bin'
W=torch.load(ckpt)
for k in W:
    W[k]=W[k].to(torch.float32) ## 原本的float16不支持乘法
    #continue
print()


totalMergeLoraPart_dict={}


def emb_w_multipl(key):## emb:w+AB
    out=W[key[0]]+ torch.matmul(W[key[1]][0], W[key[2]][0])
    return out
def linear_w_mutipl(key):## linear w+BA
    out = W[key[0]] + torch.matmul(W[key[2]][0], W[key[1]][0])
    return out
word_emb_key=['text_model.transformer.roberta.embeddings.word_embeddings.parametrizations.weight.original',
              'text_model.transformer.roberta.embeddings.word_embeddings.parametrizations.weight.0.lora_A',
              'text_model.transformer.roberta.embeddings.word_embeddings.parametrizations.weight.0.lora_B']
word_emb_w=emb_w_multipl(word_emb_key)
totalMergeLoraPart_dict['text_model.transformer.roberta.embeddings.word_embeddings.weight']=word_emb_w
#torch.save(totalMerge_dict,outputDir)
#tmp=torch.load(outputDir)
#
tokenType_emb_key=['text_model.transformer.roberta.embeddings.token_type_embeddings.parametrizations.weight.original',
                   'text_model.transformer.roberta.embeddings.token_type_embeddings.parametrizations.weight.0.lora_A',
                   'text_model.transformer.roberta.embeddings.token_type_embeddings.parametrizations.weight.0.lora_B']
tokenType_emb_w=emb_w_multipl(tokenType_emb_key)
totalMergeLoraPart_dict['text_model.transformer.roberta.embeddings.token_type_embeddings.weight']=tokenType_emb_w
#
# emb_ln_key=['text_model.transformer.roberta.emb_ln.weight','text_model.transformer.roberta.emb_ln.bias']
nlayer=24
for layer in range(nlayer):
    Wqkv_l0_key=[f'text_model.transformer.roberta.encoder.layers.{layer}.mixer.Wqkv.parametrizations.weight.original',
                  f'text_model.transformer.roberta.encoder.layers.{layer}.mixer.Wqkv.parametrizations.weight.0.lora_A',
                  f'text_model.transformer.roberta.encoder.layers.{layer}.mixer.Wqkv.parametrizations.weight.0.lora_B',
                  f'text_model.transformer.roberta.encoder.layers.{layer}.mixer.Wqkv.bias']
    Wqkv_l0_w,Wqkv_l0_bias=linear_w_mutipl(Wqkv_l0_key),W[Wqkv_l0_key[-1]]
    totalMergeLoraPart_dict[f'text_model.transformer.roberta.encoder.layers.{layer}.mixer.Wqkv.weight']=Wqkv_l0_w

#
    outproj_l0_key=[
    f'text_model.transformer.roberta.encoder.layers.{layer}.mixer.out_proj.parametrizations.weight.original',
    f'text_model.transformer.roberta.encoder.layers.{layer}.mixer.out_proj.parametrizations.weight.0.lora_A',
    f'text_model.transformer.roberta.encoder.layers.{layer}.mixer.out_proj.parametrizations.weight.0.lora_B',
    f'text_model.transformer.roberta.encoder.layers.{layer}.mixer.out_proj.bias']
    outproj_l0_w,outproj_l0_bias=linear_w_mutipl(outproj_l0_key),W[outproj_l0_key[-1]]
    totalMergeLoraPart_dict[f'text_model.transformer.roberta.encoder.layers.{layer}.mixer.out_proj.weight']=outproj_l0_w
#
# norm1_l0_key=['text_model.transformer.roberta.encoder.layers.0.norm1.weight','text_model.transformer.roberta.encoder.layers.0.norm1.bias']
#
    fc1_key=[f'text_model.transformer.roberta.encoder.layers.{layer}.mlp.fc1.parametrizations.weight.original',
         f'text_model.transformer.roberta.encoder.layers.{layer}.mlp.fc1.parametrizations.weight.0.lora_A',
         f'text_model.transformer.roberta.encoder.layers.{layer}.mlp.fc1.parametrizations.weight.0.lora_B',
        f'text_model.transformer.roberta.encoder.layers.{layer}.mlp.fc1.bias']
    fc1_w,fc1_bias=linear_w_mutipl(fc1_key),W[fc1_key[-1]]
    totalMergeLoraPart_dict[f'text_model.transformer.roberta.encoder.layers.{layer}.mlp.fc1.weight']=fc1_w
#
    fc2_key=[f'text_model.transformer.roberta.encoder.layers.{layer}.mlp.fc2.parametrizations.weight.original',
         f'text_model.transformer.roberta.encoder.layers.{layer}.mlp.fc2.parametrizations.weight.0.lora_A',
         f'text_model.transformer.roberta.encoder.layers.{layer}.mlp.fc2.parametrizations.weight.0.lora_B',
    f'text_model.transformer.roberta.encoder.layers.{layer}.mlp.fc2.bias']
    fc2_w,fc2_bias=linear_w_mutipl(fc2_key),W[fc2_key[-1]]
    totalMergeLoraPart_dict[f'text_model.transformer.roberta.encoder.layers.{layer}.mlp.fc2.weight']=fc2_w
def turn_float16():
    #########
    # 转成float16
    for k in totalMergeLoraPart_dict:
        totalMergeLoraPart_dict[k]=totalMergeLoraPart_dict[k].to(torch.float16)
########


# 把非lora的参数也存进来
for k,v in W.items():
    if 'text_model' not in k: continue
    if 'original' in k or 'lora' in k: continue 
    if 'vision_model' in k: continue
    totalMergeLoraPart_dict[k] = v
torch.save(totalMergeLoraPart_dict,'/root/autodl-tmp/jina/pytorch_model_mergeLoraPartcorrect.bin')
#
# norm2_l0_key=['text_model.transformer.roberta.encoder.layers.0.norm2.weight','text_model.transformer.roberta.encoder.layers.0.norm2.bias']




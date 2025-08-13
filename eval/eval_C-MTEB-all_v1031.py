import argparse
import torch
from C_MTEB.tasks import *
from flag_dres_model_dim1 import FlagDRESModel
from mteb import MTEB

query_instruction_for_retrieval_dict = {
    "BAAI/bge-large-zh": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-large-zh-noinstruct": None,
    "BAAI/bge-base-zh": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-small-zh": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-large-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-base-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-small-zh-v.15": "为这个句子生成表示以用于检索相关文章：",
}

#modpath='/root/autodl-tmp/jinaclip-v3'


#maindir='/root/autodl-tmp/MRL1/'

#modpath='/root/autodl-tmp/bge-large-zh-v1.5'
#modelname='BGE_data1029-3epoch-hardNegData' ### output
#modelname='2024127'

if False:
    #modpath='/root/autodl-tmp/MRL1/output/matryoshka_nli_bge-large-zh-v1.5-2024-10-14_11-16-48/checkpoint-387'
    #modpath='/root/autodl-tmp/MRL1/' + 'output/matryoshka_nli_bge-large-zh-v1.5-2024-10-14_11-46-20/checkpoint-258'
    #modpath='/root/autodl-tmp/MRL1/' + 'output/matryoshka_nli_bge-large-zh-v1.5-2024-10-14_17-52-00/checkpoint-2857'

    #modpath=maindir+'output/matryoshka_nli_bge-large-zh-v1.5-2024-10-14_19-52-27/checkpoint-11427' # 1epoch  nli+med
    #modpath=maindir+'output/matryoshka_nli_bge-large-zh-v1.5-2024-10-15_10-26-13/checkpoint-8302' # 1epoch  nli
    modpath=maindir+'output/matryoshka_nli_bge-large-zh-v1.5-2024-10-15_11-37-27/checkpoint-22854' # 2epoch  nli+med
    modpath=maindir+'output/matryoshka_nli_bge-large-zh-v1.5-2024-10-15_14-47-19/checkpoint-24140' # 2epoch nli2 med
    modpath=maindir+'output/matryoshka_nli_bge-large-zh-v1.5-2024-10-15_19-42-08/checkpoint-17142' # dim 64 768 nli med
    modpath=maindir+'output/matryoshka_nli_bge-large-zh-v1.5-2024-10-16_11-46-45/checkpoint-5714' ### dim 1024 64
    modpath=maindir+'output/matryoshka_nli_bge-large-zh-v1.5-2024-10-16_16-40-47/checkpoint-5714' ### softdim nli med
    modpath=maindir+'output/matryoshka_nli_bge-large-zh-v1.5-2024-10-16_18-23-08/checkpoint-11397' ## softdim nli 均衡样本
    modpath='/root/autodl-tmp/angle1/save_ckpt/checkpoint-1600/checkpoint-5697'  # 64dim ANGLELOSS
    modpath='/root/autodl-tmp/MRL1/output/matryoshka_nli_bge-large-zh-v1.5-2024-10-28_10-35-22/checkpoint-11397'  ### loss-dimW [1 1 1 1 5] dim64:5   32dim
    modpath='/root/autodl-tmp/MRL1/output/matryoshka_nli_bge-large-zh-v1.5-2024-10-29_21-52-01/checkpoint-11065'  ###[1 ... 10,10] 64 32 dim weight10
    modpath=maindir+'output/matryoshka_nli_bge-large-zh-v1.5-2024-10-30_14-37-53/checkpoint-22130' ### data1029 3epoch [0 0 0 0 5 5] dim weight
    modpath=maindir+'output/matryoshka_nli_bge-large-zh-v1.5-2024-10-30_18-44-11/checkpoint-6564' ### hard negative data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_name_or_path', default=None, type=str)
    parser.add_argument('--task_type', default=None, type=str)
    parser.add_argument('--add_instruction', action='store_true', help="whether to add instruction for query")
    parser.add_argument('-p','--pooling_method', default='cls', type=str)
    parser.add_argument('-d','--truncate_dim', default=256, type=int)
    parser.add_argument('-o','--outname', default=None, type=str) #### output dir name
    
    return parser.parse_args()

 


choose_task0='CMedQAv2 T2Retrieval CovidRetrieval'.split() ####rerank retrieval
choose_task1= ['Cmnli', 'Ocnli']


if __name__ == '__main__':
    args = get_args()
    
    
    bsz1=256
    #dim_trunc=256 #1024//4
    #dim_trunc=32#64
    dim_trunc=args.truncate_dim
    task_lora=None#'separation'
    
    maxlen1=2048
    maxlen1=512
    
    print(args.model_name_or_path)

    model = FlagDRESModel(model_name_or_path=args.model_name_or_path,
                          query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                          pooling_method=args.pooling_method,
                             dim1=dim_trunc,
                          batch_size=bsz1,
                          task=task_lora,
                          maxlen1=maxlen1)
    
     

    task_names = [t.description["name"] for t in MTEB(task_types=args.task_type,
                                                      task_langs=['zh', 'zh-CN']).tasks]
    
    print(task_names)
    task_names=['AmazonReviewsClassification', 'MassiveIntentClassification', 'MassiveScenarioClassification', 'TNews', 'IFlyTek', 'MultilingualSentiment', 'JDReview', 'OnlineShopping', 'Waimai', 'TNews', 'IFlyTek', 'MultilingualSentiment', 'JDReview', 'OnlineShopping', 'Waimai', 'CLSClusteringS2S', 'CLSClusteringP2P', 'ThuNewsClusteringS2S', 'ThuNewsClusteringP2P', 'CLSClusteringS2S', 'CLSClusteringP2P', 'ThuNewsClusteringS2S', 'ThuNewsClusteringP2P', 'Ocnli', 'Cmnli', 'Ocnli', 'Cmnli', 'T2Reranking', 'MMarcoReranking', 'CMedQAv1', 'CMedQAv2', 'T2Reranking', 'MMarcoReranking', 'CMedQAv1', 'CMedQAv2', 'T2Retrieval', 'MMarcoRetrieval', 'DuRetrieval', 'CovidRetrieval', 'CmedqaRetrieval', 'EcomRetrieval', 'MedicalRetrieval', 'VideoRetrieval', 'T2Retrieval', 'MMarcoRetrieval', 'DuRetrieval', 'CovidRetrieval', 'CmedqaRetrieval', 'EcomRetrieval', 'MedicalRetrieval', 'VideoRetrieval', 'STS22', 'ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STSB', 'AFQMC', 'QBQTC', 'ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STSB', 'AFQMC', 'QBQTC']
    
     
    #task_names=['T2Reranking']
    #task_names=['CmedqaRetrieval']
    #task_names=['JDReview']
    #task_names=['DuRetrieval']
    task_names=choose_task0[:1]+choose_task1 ###用其训练数据微调
    task_names+=['ThuNewsClusteringS2S','OnlineShopping','MedicalRetrieval','STSB'] # 没用其训练数据微调
    #task_names=['Ocnli']
  
    
    
    

    for task in task_names:
        # if task not in ChineseTaskList:
        #     continue
        if task in ['T2Retrieval', 'MMarcoRetrieval', 'DuRetrieval',
                    'CovidRetrieval', 'CmedqaRetrieval',
                    'EcomRetrieval', 'MedicalRetrieval', 'VideoRetrieval',
                    'T2Reranking', 'MMarcoReranking', 'CMedQAv1', 'CMedQAv2']:
            if args.model_name_or_path not in query_instruction_for_retrieval_dict:
                if args.add_instruction:
                    instruction = "为这个句子生成表示以用于检索相关文章："
                else:
                    instruction = None
                print(f"{args.model_name_or_path} not in query_instruction_for_retrieval_dict, set instruction={instruction}")
            else:
                instruction = query_instruction_for_retrieval_dict[args.model_name_or_path]
        else:
            instruction = None

            
            
        
        model.query_instruction_for_retrieval = instruction
        
        
        try:
       # if 1:

            evaluation = MTEB(tasks=[task], task_langs=['zh', 'zh-CN'])
            evaluation.run(model, output_folder=f"{args.outname}/lora{task_lora}_d{dim_trunc}_len{maxlen1}_",
                          ###overwrite_results=True
                          )
        except:
            print(f'failing ...{task}\n\n')




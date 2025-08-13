import json
import numpy as np




name2key={'CMedQAv2.json':'test#map',
         'Cmnli.json':'validation#cos_sim#accuracy',
         'MedicalRetrieval.json':'dev#map_at_1',
         'Ocnli.json':'validation#cos_sim#accuracy',
         'OnlineShopping.json':'test#accuracy',
         'STSB.json':'test#cos_sim#pearson',
         'ThuNewsClusteringS2S.json':'test#v_measure'}
def main(f):
    numll=[]
    for name,k in name2key.items():
        name1=f+name
        o=open(name1)
        d=json.load(o)
        kll=k.split('#')
        for k in kll:
            d=d[k]
        print(name,d)
        numll.append(float(d))
    ##
    print('mean',np.mean(numll))

if __name__=='__main__':
    f = '/root/autodl-tmp/code/eval/zh_results_dimN_ft_2024127_jinaclip_3epoch_nGPU/loraNone_d1024_len512_/'
    f = '/root/autodl-tmp/code/eval/zh_results_dimN_ft_2024127_jinaclip_3epoch_nGPU_grad/loraNone_d1024_len512_/'
    f = '/root/autodl-tmp/code/eval/zh_results_dimN_ft_2024127_jinaclip_3epoch_nGPU_1205/loraNone_d1024_len512_/'
    f = '/root/autodl-tmp/code/eval/zh_results_dimN_ft_2024127_jinaclip_3epoch_nGPU_1209/loraNone_d1024_len512_/'
    f='/root/autodl-tmp/code/eval/zh_results_dimN_ft_20241212_jinaclip_3epoch_nGPU_1212/loraNone_d1024_len512_/'
    main(f)

---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- common_voice
model-index:
- name: model-960hfacebook-2022.06.08
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# model-960hfacebook-2022.06.08

This model is a fine-tuned version of [facebook/wav2vec2-large-960h-lv60-self](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self) on the common_voice dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2907
- Wer: 0.1804

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 25
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Wer    |
|:-------------:|:-----:|:-----:|:---------------:|:------:|
| 6.7634        | 0.21  | 300   | 2.9743          | 0.9998 |
| 1.6536        | 0.43  | 600   | 0.8605          | 0.7529 |
| 0.9823        | 0.64  | 900   | 0.6600          | 0.6286 |
| 0.8708        | 0.86  | 1200  | 0.5780          | 0.5736 |
| 0.7878        | 1.07  | 1500  | 0.5386          | 0.5326 |
| 0.7033        | 1.29  | 1800  | 0.4986          | 0.4992 |
| 0.681         | 1.5   | 2100  | 0.4575          | 0.4778 |
| 0.6537        | 1.72  | 2400  | 0.4591          | 0.4482 |
| 0.6263        | 1.93  | 2700  | 0.4317          | 0.4353 |
| 0.5811        | 2.14  | 3000  | 0.4149          | 0.4159 |
| 0.5565        | 2.36  | 3300  | 0.4170          | 0.3956 |
| 0.5501        | 2.57  | 3600  | 0.4007          | 0.3929 |
| 0.5444        | 2.79  | 3900  | 0.3930          | 0.3851 |
| 0.5177        | 3.0   | 4200  | 0.4006          | 0.3630 |
| 0.4682        | 3.22  | 4500  | 0.3707          | 0.3713 |
| 0.4805        | 3.43  | 4800  | 0.3564          | 0.3583 |
| 0.4715        | 3.65  | 5100  | 0.3596          | 0.3434 |
| 0.4482        | 3.86  | 5400  | 0.3555          | 0.3394 |
| 0.4407        | 4.07  | 5700  | 0.3680          | 0.3312 |
| 0.4134        | 4.29  | 6000  | 0.3534          | 0.3328 |
| 0.4165        | 4.5   | 6300  | 0.3294          | 0.3259 |
| 0.4196        | 4.72  | 6600  | 0.3353          | 0.3214 |
| 0.4117        | 4.93  | 6900  | 0.3266          | 0.3211 |
| 0.3847        | 5.15  | 7200  | 0.3365          | 0.3156 |
| 0.3687        | 5.36  | 7500  | 0.3233          | 0.3014 |
| 0.376         | 5.58  | 7800  | 0.3345          | 0.2979 |
| 0.3732        | 5.79  | 8100  | 0.3105          | 0.2882 |
| 0.3705        | 6.0   | 8400  | 0.3252          | 0.2935 |
| 0.3311        | 6.22  | 8700  | 0.3266          | 0.2911 |
| 0.3386        | 6.43  | 9000  | 0.2975          | 0.2765 |
| 0.337         | 6.65  | 9300  | 0.3070          | 0.2826 |
| 0.3458        | 6.86  | 9600  | 0.3090          | 0.2766 |
| 0.3218        | 7.08  | 9900  | 0.3117          | 0.2748 |
| 0.3041        | 7.29  | 10200 | 0.2989          | 0.2651 |
| 0.3031        | 7.51  | 10500 | 0.3210          | 0.2672 |
| 0.3037        | 7.72  | 10800 | 0.3040          | 0.2667 |
| 0.3126        | 7.93  | 11100 | 0.2867          | 0.2613 |
| 0.3005        | 8.15  | 11400 | 0.3075          | 0.2610 |
| 0.2802        | 8.36  | 11700 | 0.3129          | 0.2608 |
| 0.2785        | 8.58  | 12000 | 0.3002          | 0.2579 |
| 0.2788        | 8.79  | 12300 | 0.3063          | 0.2476 |
| 0.286         | 9.01  | 12600 | 0.2971          | 0.2495 |
| 0.2534        | 9.22  | 12900 | 0.2766          | 0.2452 |
| 0.2542        | 9.44  | 13200 | 0.2893          | 0.2405 |
| 0.2576        | 9.65  | 13500 | 0.3038          | 0.2518 |
| 0.2552        | 9.86  | 13800 | 0.2851          | 0.2429 |
| 0.2487        | 10.08 | 14100 | 0.2858          | 0.2356 |
| 0.2441        | 10.29 | 14400 | 0.2999          | 0.2364 |
| 0.2345        | 10.51 | 14700 | 0.2907          | 0.2373 |
| 0.2352        | 10.72 | 15000 | 0.2885          | 0.2402 |
| 0.2464        | 10.94 | 15300 | 0.2896          | 0.2339 |
| 0.2219        | 11.15 | 15600 | 0.2999          | 0.2351 |
| 0.2257        | 11.37 | 15900 | 0.2930          | 0.2326 |
| 0.2184        | 11.58 | 16200 | 0.2980          | 0.2353 |
| 0.2182        | 11.79 | 16500 | 0.2832          | 0.2296 |
| 0.2224        | 12.01 | 16800 | 0.2797          | 0.2285 |
| 0.1991        | 12.22 | 17100 | 0.2810          | 0.2296 |
| 0.1993        | 12.44 | 17400 | 0.2949          | 0.2253 |
| 0.2042        | 12.65 | 17700 | 0.2864          | 0.2207 |
| 0.2083        | 12.87 | 18000 | 0.2860          | 0.2278 |
| 0.1998        | 13.08 | 18300 | 0.2872          | 0.2232 |
| 0.1919        | 13.3  | 18600 | 0.2894          | 0.2247 |
| 0.1925        | 13.51 | 18900 | 0.3007          | 0.2234 |
| 0.1966        | 13.72 | 19200 | 0.2831          | 0.2176 |
| 0.1942        | 13.94 | 19500 | 0.2811          | 0.2161 |
| 0.1778        | 14.15 | 19800 | 0.2901          | 0.2196 |
| 0.1755        | 14.37 | 20100 | 0.2864          | 0.2188 |
| 0.1795        | 14.58 | 20400 | 0.2927          | 0.2170 |
| 0.1817        | 14.8  | 20700 | 0.2846          | 0.2156 |
| 0.1754        | 15.01 | 21000 | 0.3036          | 0.2137 |
| 0.1674        | 15.23 | 21300 | 0.2876          | 0.2156 |
| 0.171         | 15.44 | 21600 | 0.2812          | 0.2106 |
| 0.1603        | 15.65 | 21900 | 0.2692          | 0.2093 |
| 0.1663        | 15.87 | 22200 | 0.2745          | 0.2094 |
| 0.1608        | 16.08 | 22500 | 0.2807          | 0.2043 |
| 0.1555        | 16.3  | 22800 | 0.2872          | 0.2036 |
| 0.1546        | 16.51 | 23100 | 0.2837          | 0.2049 |
| 0.1515        | 16.73 | 23400 | 0.2746          | 0.2031 |
| 0.1571        | 16.94 | 23700 | 0.2767          | 0.2047 |
| 0.1498        | 17.16 | 24000 | 0.2837          | 0.2050 |
| 0.143         | 17.37 | 24300 | 0.2745          | 0.2038 |
| 0.1471        | 17.58 | 24600 | 0.2787          | 0.2004 |
| 0.1442        | 17.8  | 24900 | 0.2779          | 0.2005 |
| 0.1481        | 18.01 | 25200 | 0.2906          | 0.2021 |
| 0.1318        | 18.23 | 25500 | 0.2936          | 0.1991 |
| 0.1396        | 18.44 | 25800 | 0.2913          | 0.1984 |
| 0.144         | 18.66 | 26100 | 0.2806          | 0.1953 |
| 0.1341        | 18.87 | 26400 | 0.2896          | 0.1972 |
| 0.1375        | 19.09 | 26700 | 0.2937          | 0.2002 |
| 0.1286        | 19.3  | 27000 | 0.2929          | 0.1954 |
| 0.1242        | 19.51 | 27300 | 0.2968          | 0.1962 |
| 0.1305        | 19.73 | 27600 | 0.2879          | 0.1944 |
| 0.1287        | 19.94 | 27900 | 0.2850          | 0.1937 |
| 0.1286        | 20.16 | 28200 | 0.2910          | 0.1961 |
| 0.121         | 20.37 | 28500 | 0.2908          | 0.1912 |
| 0.1264        | 20.59 | 28800 | 0.2853          | 0.1904 |
| 0.1238        | 20.8  | 29100 | 0.2913          | 0.1926 |
| 0.117         | 21.02 | 29400 | 0.2907          | 0.1922 |
| 0.1154        | 21.23 | 29700 | 0.2902          | 0.1888 |
| 0.1142        | 21.44 | 30000 | 0.2854          | 0.1907 |
| 0.1168        | 21.66 | 30300 | 0.2918          | 0.1873 |
| 0.1168        | 21.87 | 30600 | 0.2897          | 0.1873 |
| 0.1105        | 22.09 | 30900 | 0.2951          | 0.1856 |
| 0.1134        | 22.3  | 31200 | 0.2842          | 0.1847 |
| 0.1111        | 22.52 | 31500 | 0.2884          | 0.1829 |
| 0.1088        | 22.73 | 31800 | 0.2991          | 0.1840 |
| 0.1139        | 22.94 | 32100 | 0.2876          | 0.1839 |
| 0.1078        | 23.16 | 32400 | 0.2899          | 0.1830 |
| 0.1087        | 23.37 | 32700 | 0.2927          | 0.1803 |
| 0.1076        | 23.59 | 33000 | 0.2924          | 0.1801 |
| 0.11          | 23.8  | 33300 | 0.2877          | 0.1804 |
| 0.1067        | 24.02 | 33600 | 0.2918          | 0.1799 |
| 0.1104        | 24.23 | 33900 | 0.2908          | 0.1809 |
| 0.1023        | 24.45 | 34200 | 0.2939          | 0.1807 |
| 0.0993        | 24.66 | 34500 | 0.2925          | 0.1802 |
| 0.1053        | 24.87 | 34800 | 0.2907          | 0.1804 |


### Framework versions

- Transformers 4.17.0
- Pytorch 1.8.1+cu111
- Datasets 2.2.1
- Tokenizers 0.12.1
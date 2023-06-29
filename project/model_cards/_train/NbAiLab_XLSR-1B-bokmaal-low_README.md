---
tags:
- generated_from_trainer
model-index:
- name: XLSR-1B-bokmaal-low
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# XLSR-1B-bokmaal-low

This model was trained from scratch on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1579
- Wer: 0.0722

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1.7e-05
- train_batch_size: 12
- eval_batch_size: 12
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 24
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 34.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Wer    |
|:-------------:|:-----:|:-----:|:---------------:|:------:|
| 0.434         | 0.24  | 500   | 0.1704          | 0.1378 |
| 0.2833        | 0.48  | 1000  | 0.1638          | 0.1324 |
| 0.2478        | 0.72  | 1500  | 0.1606          | 0.1240 |
| 0.2276        | 0.97  | 2000  | 0.1562          | 0.1212 |
| 0.2208        | 1.21  | 2500  | 0.1576          | 0.1172 |
| 0.2148        | 1.45  | 3000  | 0.1502          | 0.1119 |
| 0.1994        | 1.69  | 3500  | 0.1409          | 0.1110 |
| 0.1932        | 1.93  | 4000  | 0.1432          | 0.1112 |
| 0.2122        | 2.17  | 4500  | 0.1443          | 0.1098 |
| 0.2177        | 2.42  | 5000  | 0.1329          | 0.1102 |
| 0.2058        | 2.66  | 5500  | 0.1403          | 0.1070 |
| 0.2216        | 2.9   | 6000  | 0.1342          | 0.1067 |
| 0.1984        | 3.14  | 6500  | 0.1370          | 0.1030 |
| 0.2056        | 3.38  | 7000  | 0.1371          | 0.1041 |
| 0.1735        | 3.62  | 7500  | 0.1296          | 0.1003 |
| 0.203         | 3.87  | 8000  | 0.1301          | 0.1005 |
| 0.1835        | 4.11  | 8500  | 0.1310          | 0.1004 |
| 0.178         | 4.35  | 9000  | 0.1300          | 0.0959 |
| 0.1585        | 4.59  | 9500  | 0.1277          | 0.0966 |
| 0.1848        | 4.83  | 10000 | 0.1260          | 0.0974 |
| 0.169         | 5.07  | 10500 | 0.1281          | 0.0969 |
| 0.1666        | 5.32  | 11000 | 0.1291          | 0.1003 |
| 0.1552        | 5.56  | 11500 | 0.1271          | 0.0959 |
| 0.2736        | 5.8   | 12000 | 0.1320          | 0.0935 |
| 0.2845        | 6.04  | 12500 | 0.1299          | 0.0921 |
| 0.1536        | 6.28  | 13000 | 0.1282          | 0.0927 |
| 0.1491        | 6.52  | 13500 | 0.1240          | 0.0906 |
| 0.1579        | 6.77  | 14000 | 0.1208          | 0.0921 |
| 0.16          | 7.01  | 14500 | 0.1182          | 0.0903 |
| 0.1367        | 7.25  | 15000 | 0.1214          | 0.0922 |
| 0.1499        | 7.49  | 15500 | 0.1232          | 0.0916 |
| 0.148         | 7.73  | 16000 | 0.1184          | 0.0896 |
| 0.1426        | 7.97  | 16500 | 0.1201          | 0.0889 |
| 0.1471        | 8.22  | 17000 | 0.1256          | 0.0882 |
| 0.1358        | 8.46  | 17500 | 0.1265          | 0.0909 |
| 0.1245        | 8.7   | 18000 | 0.1263          | 0.0886 |
| 0.1407        | 8.94  | 18500 | 0.1226          | 0.0885 |
| 0.1289        | 9.18  | 19000 | 0.1315          | 0.0873 |
| 0.1326        | 9.42  | 19500 | 0.1233          | 0.0868 |
| 0.1305        | 9.67  | 20000 | 0.1237          | 0.0870 |
| 0.1432        | 9.91  | 20500 | 0.1234          | 0.0857 |
| 0.1205        | 10.15 | 21000 | 0.1303          | 0.0858 |
| 0.1248        | 10.39 | 21500 | 0.1252          | 0.0858 |
| 0.1251        | 10.63 | 22000 | 0.1253          | 0.0869 |
| 0.1143        | 10.87 | 22500 | 0.1266          | 0.0860 |
| 0.1155        | 11.12 | 23000 | 0.1219          | 0.0862 |
| 0.1227        | 11.36 | 23500 | 0.1329          | 0.0864 |
| 0.1229        | 11.6  | 24000 | 0.1244          | 0.0855 |
| 0.1112        | 11.84 | 24500 | 0.1356          | 0.0851 |
| 0.2163        | 12.08 | 25000 | 0.1252          | 0.0847 |
| 0.1146        | 12.32 | 25500 | 0.1211          | 0.0837 |
| 0.1058        | 12.57 | 26000 | 0.1247          | 0.0843 |
| 0.1099        | 12.81 | 26500 | 0.1189          | 0.0833 |
| 0.1028        | 13.05 | 27000 | 0.1303          | 0.0815 |
| 0.1092        | 13.29 | 27500 | 0.1305          | 0.0838 |
| 0.1076        | 13.53 | 28000 | 0.1276          | 0.0842 |
| 0.1074        | 13.77 | 28500 | 0.1268          | 0.0844 |
| 0.0971        | 14.02 | 29000 | 0.1322          | 0.0839 |
| 0.1109        | 14.26 | 29500 | 0.1287          | 0.0821 |
| 0.0991        | 14.5  | 30000 | 0.1289          | 0.0831 |
| 0.1095        | 14.74 | 30500 | 0.1273          | 0.0822 |
| 0.1015        | 14.98 | 31000 | 0.1326          | 0.0816 |
| 0.1051        | 15.22 | 31500 | 0.1337          | 0.0814 |
| 0.0894        | 15.47 | 32000 | 0.1331          | 0.0802 |
| 0.1           | 15.71 | 32500 | 0.1304          | 0.0798 |
| 0.0957        | 15.95 | 33000 | 0.1293          | 0.0824 |
| 0.0921        | 16.19 | 33500 | 0.1382          | 0.0808 |
| 0.0986        | 16.43 | 34000 | 0.1301          | 0.0788 |
| 0.098         | 16.67 | 34500 | 0.1305          | 0.0795 |
| 0.0974        | 16.92 | 35000 | 0.1325          | 0.0796 |
| 0.0886        | 17.16 | 35500 | 0.1332          | 0.0796 |
| 0.0892        | 17.4  | 36000 | 0.1327          | 0.0785 |
| 0.0917        | 17.64 | 36500 | 0.1304          | 0.0793 |
| 0.0919        | 17.88 | 37000 | 0.1353          | 0.0791 |
| 0.1007        | 18.12 | 37500 | 0.1340          | 0.0791 |
| 0.0831        | 18.37 | 38000 | 0.1327          | 0.0786 |
| 0.0862        | 18.61 | 38500 | 0.1343          | 0.0792 |
| 0.0837        | 18.85 | 39000 | 0.1334          | 0.0777 |
| 0.0771        | 19.09 | 39500 | 0.1456          | 0.0778 |
| 0.0841        | 19.33 | 40000 | 0.1365          | 0.0784 |
| 0.0874        | 19.57 | 40500 | 0.1379          | 0.0779 |
| 0.0773        | 19.82 | 41000 | 0.1359          | 0.0776 |
| 0.0771        | 20.06 | 41500 | 0.1392          | 0.0776 |
| 0.0861        | 20.3  | 42000 | 0.1395          | 0.0774 |
| 0.0773        | 20.54 | 42500 | 0.1356          | 0.0775 |
| 0.069         | 20.78 | 43000 | 0.1399          | 0.0765 |
| 0.0823        | 21.02 | 43500 | 0.1469          | 0.0774 |
| 0.0747        | 21.27 | 44000 | 0.1415          | 0.0768 |
| 0.0703        | 21.51 | 44500 | 0.1405          | 0.0778 |
| 0.0776        | 21.75 | 45000 | 0.1492          | 0.0778 |
| 0.0833        | 21.99 | 45500 | 0.1448          | 0.0767 |
| 0.0796        | 22.23 | 46000 | 0.1434          | 0.0761 |
| 0.0613        | 22.47 | 46500 | 0.1446          | 0.0768 |
| 0.0753        | 22.72 | 47000 | 0.1439          | 0.0757 |
| 0.076         | 22.96 | 47500 | 0.1402          | 0.0759 |
| 0.0619        | 23.2  | 48000 | 0.1473          | 0.0767 |
| 0.1322        | 23.44 | 48500 | 0.1431          | 0.0766 |
| 0.0691        | 23.68 | 49000 | 0.1452          | 0.0753 |
| 0.061         | 23.92 | 49500 | 0.1452          | 0.0752 |
| 0.0716        | 24.17 | 50000 | 0.1429          | 0.0756 |
| 0.074         | 24.41 | 50500 | 0.1440          | 0.0746 |
| 0.0696        | 24.65 | 51000 | 0.1459          | 0.0756 |
| 0.081         | 24.89 | 51500 | 0.1443          | 0.0751 |
| 0.0754        | 25.13 | 52000 | 0.1483          | 0.0755 |
| 0.0864        | 25.37 | 52500 | 0.1467          | 0.0757 |
| 0.0662        | 25.62 | 53000 | 0.1471          | 0.0748 |
| 0.109         | 25.86 | 53500 | 0.1472          | 0.0759 |
| 0.0682        | 26.1  | 54000 | 0.1539          | 0.0748 |
| 0.0655        | 26.34 | 54500 | 0.1469          | 0.0743 |
| 0.0651        | 26.58 | 55000 | 0.1553          | 0.0748 |
| 0.0666        | 26.82 | 55500 | 0.1520          | 0.0744 |
| 0.0724        | 27.07 | 56000 | 0.1526          | 0.0738 |
| 0.067         | 27.31 | 56500 | 0.1489          | 0.0738 |
| 0.0658        | 27.55 | 57000 | 0.1518          | 0.0738 |
| 0.0581        | 27.79 | 57500 | 0.1518          | 0.0739 |
| 0.0639        | 28.03 | 58000 | 0.1495          | 0.0736 |
| 0.0606        | 28.27 | 58500 | 0.1549          | 0.0739 |
| 0.0641        | 28.52 | 59000 | 0.1513          | 0.0735 |
| 0.0612        | 28.76 | 59500 | 0.1524          | 0.0739 |
| 0.0536        | 29.0  | 60000 | 0.1565          | 0.0741 |
| 0.0574        | 29.24 | 60500 | 0.1541          | 0.0741 |
| 0.057         | 29.48 | 61000 | 0.1555          | 0.0741 |
| 0.0624        | 29.72 | 61500 | 0.1590          | 0.0736 |
| 0.0531        | 29.97 | 62000 | 0.1590          | 0.0734 |
| 0.0661        | 30.21 | 62500 | 0.1599          | 0.0732 |
| 0.0641        | 30.45 | 63000 | 0.1576          | 0.0730 |
| 0.0562        | 30.69 | 63500 | 0.1593          | 0.0734 |
| 0.0527        | 30.93 | 64000 | 0.1604          | 0.0730 |
| 0.0579        | 31.17 | 64500 | 0.1571          | 0.0734 |
| 0.0508        | 31.42 | 65000 | 0.1603          | 0.0733 |
| 0.0524        | 31.66 | 65500 | 0.1588          | 0.0726 |
| 0.0564        | 31.9  | 66000 | 0.1571          | 0.0727 |
| 0.0551        | 32.14 | 66500 | 0.1584          | 0.0728 |
| 0.0564        | 32.38 | 67000 | 0.1565          | 0.0726 |
| 0.0628        | 32.62 | 67500 | 0.1558          | 0.0725 |
| 0.0561        | 32.87 | 68000 | 0.1582          | 0.0727 |
| 0.0553        | 33.11 | 68500 | 0.1591          | 0.0726 |
| 0.0504        | 33.35 | 69000 | 0.1590          | 0.0725 |
| 0.0539        | 33.59 | 69500 | 0.1582          | 0.0723 |
| 0.0576        | 33.83 | 70000 | 0.1579          | 0.0722 |


### Framework versions

- Transformers 4.17.0.dev0
- Pytorch 1.10.0+cu113
- Datasets 1.18.3
- Tokenizers 0.10.3
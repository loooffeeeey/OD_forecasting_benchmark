# OD forecasting benchmark

![Illustration of OD construction](assets/problem_for.png)

This is the OD (origin-destination) forecasting benchmark.


```
dgl==1.1.0
matplotlib==3.1.1
numpy==1.21.6
pandas==1.1.5
Pillow==9.5.0
scikit_learn==0.24.2
scipy==1.3.1
tensorflow==2.12.0
tf_slim==1.1.0
torch==1.13.1
torch_geometric==2.3.1
tqdm==4.62.3
```

## Systematic Comparison of Methods

| model | Spatial Topology Construction | Spatial Feature Modeling | Temporal Modeling | Learning |
| ----- | -----                         | -----                    | -----             | -----             |
| [GEML](./models/GEML/)  | grids as nodes <br> geo-adjacency graph <br> POI-similarity graph | GCN                         |  LSTM             | multi-task learning              |
| [MPGCN](./models/MPGCN/) | regions as nodes<br>distance-based graph<br>POI-similarity graph<br>OD flow-based graph | 2DGCN | LSTM | MSELoss |
| [Gallet](./models/Gallet/) | regions as nodes<br>OD flow-based graph<br>distance-based graph | spatial attention | temporal attention | MSELoss |
| [gMHC-STA](./models/gMHC-STA/) | region-pairs as nodes<br>fully-connected graph | GCN + spatial attention | self-attention | MSELoss |
|ST-VGCN | region-pairs as nodes<br>OD flow-based graph | GCN + gated mechanism | GRU | MSELoss|
| MVPF | stations as nodes<br>distance-based graph | GAT | GRU | MSELoss |
| Hex D-GCN | hexagonal grids as nodes<br>taxi path-based dynamic graph | GCN | GRU | MSELoss |
|CWGAN-GP | OD matrix as an image | CNN | CNN | GAN-based training |
| SEHNN | stations as nodes<br>geo-adjacency graph| GCN | LSTM | VAE-based training |
| HC-LSTM | grids as nodes<br>OD flow-based graph<br>in/out flow as an image<br>OD matrix as an image | CNN + GCN | LSTM | MSELoss |
| ST-GDL | regions as nodes<br>distance-based graph |CNN + GCN | CNN | MSELoss |
| PGCM | region pairs as nodes<br>OD flow-based graph | GCN + gated mechanism | none | probabilistic inference<br>with Monte Carlo |
| MF-ResNet | OD matrix as an image | CNN | none | MSELoss |
| TS-STN | stations as nodes<br>OD flow-based graph | temporally shifted<br>graph convolution | LSTM + attention | Partially MSELoss |
| DMGC-GAN | regions as nodes<br>geo-adjacency graph<br>OD flow-based graph<br>in/out flow-based graph |GCN | GRU | GAN-based training |
| DNEAT | regions as nodes<br>geo-adjacency graph<br>OD flow-based graph | attention | attention | MSELoss |
| CAS-CNN | OD matrix as image | CNN | channel-wise attention | masked MSELoss |
|ST-ED-RMGC | region pairs as nodes<br>fully-connected graph<br>geo-adjacency graph<br>POI-based graph<br>disntance-based graph<br>OD flow-based graph | GCN | LSTM | MSELoss |
| HSTN | regions as nodes<br>geo-adjacency graph<br>in/out flow-based graph | GCN | GRU+Seq2Seq | MSELoss |
| BGARN | grid clusters as nodes<br>distance-based graph<br>OD flow-based graph | GCN + attention | LSTM | MSELoss |
| HMOD | regions as nodes<br>OD flow-based graph | random walk for embedding | GRU | MSELoss |
| STHAN | regions as nodes<br>geo-adjacency graph<br>POI-based graph<br>OD flow-based graph | convolution by meta-paths + attention | GRU | MSELoss |
| ODformer | regions as nodes | 2D-GCN within Transformer | none | MSELoss |
| CMOD | stations as nodes<br>passengers as edges | multi-level information aggregation | multi-level information aggregation | continous time forecasting |
| MIAM | stations as nodes<br>railway-based graph | GCGRU | Transformer | online forecasting |
| DAGNN | regions as nodes<br>fully-connected graph | subgraph + GCN | TCN | MSELoss |





## Performance Comparison


| model | RMSE | NRMSE | MAE | MAPE | sMAPE |
| ----- | ----- | ----- | ----- | ----- | ----- | 
| [LSTNet](./models/LSTNet/)  |  |       |24.5363|0.5161|       |
| [GCRN](./models/GCRN/)  | 120.2321 |       |24.5363|0.5161|       |
| [GEML](./models/GEML/)  | 113.8526 |       |39.5888|3.1885|       |
| [MPGCN](./models/MPGCN/) | 1.1421 |       |     |      |       |
| [PGCN](./models/PGCN/) |      |       |     |      |       |
| [ST-GDL](./models/ST-GDL/) |      |       |     |      |       |
| [Gallet](./models/Gallet/) | 1081.1332|       |355.7162|0.6623|       |
| [Hex D-GCN](./models/Hex_DGCN/) |      |       |     |      |       |
| [BGARN](./models/BGARN/) | 52.2182|       |10.3148|0.5017|       |
| [CMOD](./models/CMOD/) |      |       |     |      |       |
| [AEST](./models/AEST/) |      |       |     |      |       |
| [HMOD](./models/HMOD/) |      |       |     |      |       |
| [MVPF](./models/MVPF/) |      |       |     |      |       |
| [DDW](./models/DDW/) |      |       |     |      |       |
| [ST-VGCN](./models/STVGCN/) |      |       |     |      |       |
| [CA-SATCN](./models/CA-SATCN/) |      |       |     |      |       |
| [ODformer](./models/ODformer/) |      |       |     |      |       |
| [HIAM](./models/HIAM/)|      |       |     |      |       |
| [STHAN](./models/STHAN/) |      |       |     |      |       |

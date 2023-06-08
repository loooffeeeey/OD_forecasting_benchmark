# OD forecasting benchmark

![Illustration of OD construction](assets/problem_for.png)

This is the OD (origin-destination) forecasting benchmark.


## Systematic Comparison of Methods

| model | Spatial Topology Construction | Spatial Feature Modeling | Temporal Modeling | Learning |
| ----- | -----                         | -----                    | -----             | -----             |
| GEML  | <font size=1>grids as nodes</font> <br><font size=1>geo-adjacency graph</font><br><font size=1>POI-based graph</font> | GCN                         |  LSTM             | multi-task learning              |
| MPGCN | <font size=1>regions as nodes</font><br><font size=1>distance-based graph</font><br><font size=1>POI-similarity graph</font><br><font size=1>OD flow-based graph</font> | 2DGCN | LSTM | MSELoss |
| Gallet | <font size=1>regions as nodes</font><br><font size=1>OD flow-based graph</font><br><font size=1>distance-based graph</font> | spatial attention | temporal attention | MSELoss |
| gMHC-STA | <font size=1>region-pairs as nodes</font><br><font size=1>fully-connected graph</font> | GCN + spatial attention | self-attention | MSELoss |
|ST-VGCN | <font size=1>region-pairs as nodes</font><br><font size=1>OD flow-based graph</font> | GCN + gated mechanism | GRU | MSELoss|
| MVPF | <font size=1>stations as nodes</font><br><font size=1>distance-based graph</font> | GAT | GRU | MSELoss |
| Hex D-GCN | <font size=1>hexagonal grids as nodes</font><br><font size=1>taxi path-based dynamic graph</font> | GCN | GRU | MSELoss |
|CWGAN-GP | <font size=1>OD matrix as an image</font> | CNN | CNN | GAN-based training |
| SEHNN | <font size=1>stations as nodes</font><br><font size=1>geo-adjacency graph</font>| GCN | LSTM | VAE-based training |
| HC-LSTM | <font size=1>grids as nodes</font><br><font size=1>OD flow-based graph</font><br><font size=1>in/out flow as an image</font><br><font size=1>OD matrix as an image</font> | CNN + GCN | LSTM | MSELoss |
| ST-GDL | <font size=1>regions as nodes</font><br><font size=1>distance-based graph</font> |CNN + GCN | CNN | MSELoss |
| PGCM | <font size=1>region pairs as nodes</font><br><font size=1>OD flow-based graph</font> | GCN + gated mechanism | none | probabilistic inference<br>with Monte Carlo |
| MF-ResNet | <font size=1>OD matrix as an image</font> | CNN | none | MSELoss |
| TS-STN | <font size=1>stations as nodes</font><br><font size=1>OD flow-based graph</font> | temporally shifted<br>graph convolution | LSTM + attention | Partially MSELoss |
| DMGC-GAN | <font size=1>regions as nodes</font><br><font size=1>geo-adjacency graph</font><br><font size=1>OD flow-based graph</font><br><font size=1>in/out flow-based graph</font> |GCN | GRU | GAN-based training |
| DNEAT | <font size=1>regions as nodes</font><br><font size=1>geo-adjacency graph</font><br><font size=1>OD flow-based graph</font> | attention | attention | MSELoss |
| CAS-CNN | <font size=1>OD matrix as image</font> | CNN | channel-wise attention | masked MSELoss |
|ST-ED-RMGC | <font size=1>region pairs as nodes</font><br><font size=1>fully-connected graph</font><br><font size=1>geo-adjacency graph</font><br><font size=1>POI-based graph</font><br><font size=1>disntance-based graph</font><br><font size=1>OD flow-based graph</font> | GCN | LSTM | MSELoss |
| HSTN | <font size=1>regions as nodes</font><br><font size=1>geo-adjacency graph</font><br><font size=1>in/out flow-based graph</font> | GCN | GRU+Seq2Seq | MSELoss |
| BGARN | <font size=1>grid clusters as nodes</font><br><font size=1>distance-based graph</font><br><font size=1>OD flow-based graph</font> | GCN + attention | LSTM | MSELoss |
| HMOD | <font size=1>regions as nodes</font><br><font size=1>OD flow-based graph</font> | random walk for embedding | GRU | MSELoss |
| STHAN | <font size=1>regions as nodes</font><br><font size=1>geo-adjacency graph</font><br><font size=1>POI-based graph</font><br><font size=1>OD flow-based graph</font> | convolution by meta-paths + attention | GRU | MSELoss |
| ODformer | <font size=1>regions as nodes</font> | 2D-GCN within Transformer | none | MSELoss |
| CMOD | <font size=1>stations as nodes</font><br><font size=1>passengers as edges</font> | multi-level information aggregation | multi-level information aggregation | continous time forecasting |
| MIAM | <font size=1>stations as nodes</font><br><font size=1>railway-based graph</font> | GCGRU | Transformer | online forecasting |
| DAGNN | <font size=1>regions as nodes</font><br><font size=1>fully-connected graph</font> | subgraph + GCN | TCN | MSELoss |





## Performance Comparison

| model | RMSE | NRMSE | MAE | MAPE | sMAPE |
| ----- | ----- | ----- | ----- | ----- | ----- | 
| GCRN  | 120.2321 |       |24.5363|0.5161|       |
| GEML  | 113.8526 |       |39.5888|3.1885|       |
| MPGCN | 1.1421 |       |     |      |       |
| Gallet| 1081.1332|       |355.7162|0.6623|       |
|RefGaaRN| 52.2182|       |10.3148|0.5017|       |






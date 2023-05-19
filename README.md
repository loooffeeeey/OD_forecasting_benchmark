# OD forecasting benchmark











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
| [C-AHGCSP](./models/CAHGCSP/) |      |       |     |      |       |
| [ODformer](./models/ODformer/) |      |       |     |      |       |
| [HIAM](./models/HIAM/)|      |       |     |      |       |
| [STHAN](./models/STHAN/) |      |       |     |      |       |




| model | Spatial Topology Construction | Spatial Feature Modeling | Temporal Modeling | External Features |
| ----- | -----                         | -----                    | -----             | -----             |
| [LSTNet](./models/LSTNet/) |      |       |     |      |       |
| [GCRN](./models/GCRN/) |      |       |     |      |       |
| [GEML](./models/GEML/) | grids as nodes <br> geo-adjacency graph <br> POI-similarity graph | GCN                         |  LSTM             | none              |
| [MPGCN](./models/MPGCN/) |      |       |     |      |       |
| [PGCN](./models/PGCN/) |      |       |     |      |       |
| [ST-GDL](./models/ST-GDL/) |      |       |     |      |       |
| [Gallet](./models/Gallet/) |      |       |     |      |       |
| [Hex D-GCN](./models/Hex_DGCN/) |      |       |     |      |       |
| [BGARN](./models/BGARN/) |      |       |     |      |       |
| [CMOD](./models/CMOD/) |      |       |     |      |       |
| [AEST](./models/AEST/) |      |       |     |      |       |
| [HMOD](./models/HMOD/) |      |       |     |      |       |
| [MVPF](./models/MVPF/) |      |       |     |      |       |
| [DDW](./models/DDW/) |      |       |     |      |       |
| [ST-VGCN](./models/STVGCN/) |      |       |     |      |       |
| [C-AHGCSP](./models/CAHGCSP/) |      |       |     |      |       |
| [ODformer](./models/ODformer/) |      |       |     |      |       |
| [HIAM](./models/HIAM/)|      |       |     |      |       |
| [STHAN](./models/STHAN/) |      |       |     |      |       |

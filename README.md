# OD forecasting benchmark










| model | RMSE | NRMSE | MAE | MAPE | sMAPE |
| ----- | ----- | ----- | ----- | ----- | ----- | 
| GEML  |      |       |     |      |       |
| MPGCN | 1.1421 |       |     |      |       |
| Gallet|      |       |     |      |       |








| model | Spatial Topology Construction | Spatial Feature Modeling | Temporal Modeling | External Features |
| ----- | -----                         | -----                    | -----             | -----             |
| GEML  | grids as nodes <br> geo-adjacency graph <br> POI-similarity graph | GCN                         |  LSTM             | none              |
| Gallet |

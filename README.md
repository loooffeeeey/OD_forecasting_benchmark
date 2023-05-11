# OD forecasting benchmark










| model | RMSE | NRMSE | MAE | MAPE | sMAPE |
| ----- | ----- | ----- | ----- | ----- | ----- | 
| GEML  |      |       |     |      |       |
| MPGCN | 1.1421 |       |     |      |       |
| Gallet| 1081.1332|       |355.7162|0.6623|       |








| model | Spatial Topology Construction | Spatial Feature Modeling | Temporal Modeling | External Features |
| ----- | -----                         | -----                    | -----             | -----             |
| GEML  | grids as nodes <br> geo-adjacency graph <br> POI-similarity graph | GCN                         |  LSTM             | none              |
| Gallet |

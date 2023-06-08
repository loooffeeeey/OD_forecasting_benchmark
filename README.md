# OD forecasting benchmark

![Illustration of OD construction](assets/problem_for.png)

This is the OD (origin-destination) forecasting benchmark.








| model | RMSE | NRMSE | MAE | MAPE | sMAPE |
| ----- | ----- | ----- | ----- | ----- | ----- | 
| GCRN  | 120.2321 |       |24.5363|0.5161|       |
| GEML  | 113.8526 |       |39.5888|3.1885|       |
| MPGCN | 1.1421 |       |     |      |       |
| Gallet| 1081.1332|       |355.7162|0.6623|       |
|RefGaaRN| 52.2182|       |10.3148|0.5017|       |







| model | Spatial Topology Construction | Spatial Feature Modeling | Temporal Modeling | External Features |
| ----- | -----                         | -----                    | -----             | -----             |
| GEML  | grids as nodes <br> geo-adjacency graph <br> POI-similarity graph | GCN                         |  LSTM             | none              |
| Gallet |

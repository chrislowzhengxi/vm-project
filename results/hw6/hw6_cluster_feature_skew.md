|                    |   raw feature skewness (train) |   after transform (train) | transform            |
|:-------------------|-------------------------------:|--------------------------:|:---------------------|
| cores              |                           4.13 |                      1.74 | log2                 |
| memory_gb          |                           1.15 |                      0.01 | log2                 |
| sub_hist_mean      |                          -0.46 |                     -0.46 | /100                 |
| sub_hist_median    |                          -0.22 |                     -0.22 | /100                 |
| sub_hist_frac_le75 |                          -0.48 |                      0.48 | 1 - share            |
| sub_hist_frac_le50 |                           0.31 |                     -0.31 | 1 - share            |
| sub_n_hist         |                           2.67 |                     -0.52 | log1p                |
| dep_hist_mean      |                          -0.28 |                     -0.28 | /100                 |
| hour               |                          -0.09 |                    nan    | sin/cos of 2*pi*h/24 |

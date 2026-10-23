| check                                                            |      n |    MAE |    RMSE |      R2 | note                                |
|:-----------------------------------------------------------------|-------:|-------:|--------:|--------:|:------------------------------------|
| RF C, all test                                                   | 426846 | 18.582 |  24.835 |   0.531 |                                     |
| RF C, validation period (days 21-24)                             | 293872 | 19.093 |  24.674 |   0.514 |                                     |
| RF C, test day 25                                                |  87796 | 18.841 |  25.073 |   0.519 | share right-censored=0.016          |
| RF C, test day 26                                                |  97585 | 17.450 |  24.072 |   0.580 | share right-censored=0.019          |
| RF C, test day 27                                                |  91418 | 19.038 |  25.192 |   0.516 | share right-censored=0.040          |
| RF C, test day 28                                                |  91851 | 18.315 |  24.873 |   0.530 | share right-censored=0.046          |
| RF C, test day 29                                                |  58196 | 19.790 |  25.106 |   0.469 | share right-censored=0.152          |
| RF C, lifetime <= 5 min                                          |  25997 | 25.601 |  31.804 |  -0.906 | mean p95=19.6                       |
| RF C, lifetime 5 min - 1 h                                       | 261842 | 17.061 |  24.015 |   0.552 | mean p95=40.9                       |
| RF C, lifetime 1 h - 1 day                                       | 118618 | 19.596 |  24.137 |   0.405 | mean p95=66.1                       |
| RF C, lifetime > 1 day                                           |  20389 | 23.255 |  28.747 |   0.163 | mean p95=37.4                       |
| RF C, test excl. top-10 training subscriptions                   | 328561 | 17.148 |  23.585 |   0.574 |                                     |
| Train-mean baseline, test excl. top-10 subs                      | 328561 | 32.729 |  36.554 |  -0.023 |                                     |
| Subscription-macro-averaged MAE (each test sub weighted equally) |   1726 | 21.254 | nan     | nan     | train-mean baseline macro MAE=30.58 |
| LEAKY: RF with lenient sub history (A + running-VM p95)          | 426846 | 18.384 |  24.658 |   0.538 | compare with RF B                   |
| LEAKY: RF C + lifetime + vmcategory                              | 426846 | 15.707 |  21.797 |   0.639 | illustrates leakage, not usable     |
| RF C without dep_batch_size                                      | 426846 | 18.710 |  24.971 |   0.526 |                                     |
| RF C trained only on labels observable by day 21                 | 426846 | 18.778 |  24.977 |   0.526 | train rows=1,703,020                |

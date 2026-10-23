|                                 |     1 |    2 |    5 |    10 |
|:--------------------------------|------:|-----:|-----:|------:|
| Ridge naive threshold           | -22.6 | 10.7 | 14.5 | -10.4 |
| Ridge + margin (val-tuned)      |  -0.4 | 11.7 | 19.7 |  14.4 |
| CART naive threshold            | -22.1 | 11.2 | 15.3 |  -8.9 |
| CART + margin (val-tuned)       |  -0.9 | 12.5 | 19.5 |  13.9 |
| RF naive threshold              | -21.1 | 11.6 | 15.2 |  -9.9 |
| RF + margin (val-tuned)         |  -0.3 | 12.6 | 19.6 |  15.2 |
| Direct cost-aware CART          |   0.0 | 12.1 | 18.8 |  14.5 |
| Direct cost-aware RF            |   0.0 | 13.0 | 21.5 |  16.6 |
| Depth-3 policy tree             |   0.0 | 11.6 | 19.4 |  13.3 |
| Oracle (hindsight cost-optimal) |   0.0 | 27.7 | 41.1 |  40.5 |

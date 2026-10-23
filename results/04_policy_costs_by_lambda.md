|                                 |      1 |      2 |      5 |     10 |
|:--------------------------------|-------:|-------:|-------:|-------:|
| Best fixed (val)                | 0.5349 | 0.7895 | 1.0000 | 1.0000 |
| Ridge naive threshold           | 0.6556 | 0.7054 | 0.8548 | 1.1039 |
| Ridge + margin (val-tuned)      | 0.5373 | 0.6971 | 0.8032 | 0.8559 |
| CART naive threshold            | 0.6530 | 0.7015 | 0.8469 | 1.0892 |
| CART + margin (val-tuned)       | 0.5400 | 0.6907 | 0.8053 | 0.8609 |
| RF naive threshold              | 0.6477 | 0.6979 | 0.8484 | 1.0992 |
| RF + margin (val-tuned)         | 0.5364 | 0.6900 | 0.8042 | 0.8484 |
| Direct cost-aware CART          | 0.5349 | 0.6940 | 0.8116 | 0.8551 |
| Direct cost-aware RF            | 0.5349 | 0.6866 | 0.7852 | 0.8338 |
| Depth-3 policy tree             | 0.5349 | 0.6982 | 0.8058 | 0.8674 |
| Oracle (hindsight cost-optimal) | 0.5349 | 0.5708 | 0.5890 | 0.5952 |

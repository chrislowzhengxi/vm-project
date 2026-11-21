|                                                                                 |     2 |     5 |    10 |
|:--------------------------------------------------------------------------------|------:|------:|------:|
| ('OPT-exact', 'A: size + time (7)')                                             |  4.53 |  6.00 |  0.93 |
| ('OPT-exact', 'B: A + subscription history (top 8)')                            | 11.54 | 20.31 | 15.06 |
| ('OPT-exact', 'C: B + deployment history (top 8, main)')                        | 11.53 | 20.10 | 15.32 |
| ('Greedy', 'A: size + time (7)')                                                |  3.38 |  5.40 |  0.00 |
| ('Greedy', 'B: A + subscription history (top 8)')                               | 11.32 | 20.19 | 14.96 |
| ('Greedy', 'C: B + deployment history (top 8, main)')                           | 11.37 | 19.96 | 14.76 |
| ('Cost-aware RF (all set features)', 'A: size + time (7)')                      |  5.07 |  7.19 |  1.15 |
| ('Cost-aware RF (all set features)', 'B: A + subscription history (top 8)')     | 12.58 | 20.89 | 16.38 |
| ('Cost-aware RF (all set features)', 'C: B + deployment history (top 8, main)') | 12.94 | 21.59 | 16.74 |

|                                                                      |   cost |   % saving vs best fixed |   E[r] % |   P(Y>r) % |   E[(Y-r)+] pp |   P(Y-r>20pp) % |
|:---------------------------------------------------------------------|-------:|-------------------------:|---------:|-----------:|---------------:|----------------:|
| ('VM-weighted', 'H1 customer mean + margin')                         |   0.81 |                    19.14 |    72.51 |      11.65 |           1.67 |            3.49 |
| ('VM-weighted', 'RF point -> tier')                                  |   0.86 |                    14.22 |    59.04 |      29.96 |           5.35 |           13.13 |
| ('VM-weighted', 'RF point + margin')                                 |   0.80 |                    19.77 |    73.21 |       9.95 |           1.40 |            2.77 |
| ('VM-weighted', 'Greedy depth 3')                                    |   0.80 |                    19.96 |    74.35 |       8.11 |           1.14 |            2.24 |
| ('VM-weighted', 'OPT depth 3')                                       |   0.80 |                    20.10 |    73.41 |       9.24 |           1.30 |            2.55 |
| ('VM-weighted', 'OPT 4-leaf (depth 2)')                              |   0.80 |                    19.81 |    74.62 |       7.71 |           1.11 |            2.20 |
| ('VM-weighted', 'OPT leaf-penalized (val-selected)')                 |   0.80 |                    20.38 |    73.55 |       8.57 |           1.21 |            2.43 |
| ('VM-weighted', 'OPT depth 4 (certified)')                           |   0.79 |                    20.50 |    73.79 |       8.41 |           1.14 |            2.01 |
| ('VM-weighted', 'Cost-aware RF')                                     |   0.78 |                    21.60 |    72.48 |       9.11 |           1.18 |            2.19 |
| ('VM-weighted', 'Forest-weighted prescription')                      |   0.79 |                    21.44 |    72.49 |       9.17 |           1.21 |            2.19 |
| ('equal-subscription-weighted', 'H1 customer mean + margin')         |   0.84 |                    16.07 |    74.71 |      11.71 |           1.84 |            3.71 |
| ('equal-subscription-weighted', 'RF point -> tier')                  |   0.85 |                    15.40 |    61.27 |      26.34 |           4.66 |           10.12 |
| ('equal-subscription-weighted', 'RF point + margin')                 |   0.84 |                    16.26 |    76.83 |       9.71 |           1.38 |            2.62 |
| ('equal-subscription-weighted', 'Greedy depth 3')                    |   0.82 |                    18.31 |    72.46 |      11.37 |           1.85 |            3.65 |
| ('equal-subscription-weighted', 'OPT depth 3')                       |   0.82 |                    17.60 |    71.85 |      13.06 |           2.11 |            4.16 |
| ('equal-subscription-weighted', 'OPT 4-leaf (depth 2)')              |   0.82 |                    18.17 |    72.40 |      11.21 |           1.89 |            3.87 |
| ('equal-subscription-weighted', 'OPT leaf-penalized (val-selected)') |   0.82 |                    18.49 |    72.12 |      11.69 |           1.88 |            3.78 |
| ('equal-subscription-weighted', 'OPT depth 4 (certified)')           |   0.82 |                    18.11 |    72.03 |      12.40 |           1.97 |            3.81 |
| ('equal-subscription-weighted', 'Cost-aware RF')                     |   0.81 |                    18.93 |    73.98 |      10.28 |           1.42 |            2.68 |
| ('equal-subscription-weighted', 'Forest-weighted prescription')      |   0.81 |                    19.02 |    73.23 |      11.01 |           1.55 |            2.90 |

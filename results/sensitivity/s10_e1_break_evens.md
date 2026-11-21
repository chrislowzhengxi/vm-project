|                                                                                |   break-even beta | in [1, 20]   | cheaper above break-even       |
|:-------------------------------------------------------------------------------|------------------:|:-------------|:-------------------------------|
| ('OPT-exact [full]', 'Greedy [full]', 'validation')                            |             5.248 | True         | Greedy [full]                  |
| ('OPT-exact [full]', 'Greedy [full]', 'test')                                  |             5.838 | True         | Greedy [full]                  |
| ('OPT-exact [full]', 'Cost-aware RF', 'validation')                            |            -0.841 | False        | Cost-aware RF                  |
| ('OPT-exact [full]', 'Cost-aware RF', 'test')                                  |            -7.501 | False        | Cost-aware RF                  |
| ('OPT-exact [full]', 'CART-MSE (26 feat.) [full]', 'validation')               |             4.064 | True         | OPT-exact [full]               |
| ('OPT-exact [full]', 'CART-MSE (26 feat.) [full]', 'test')                     |             3.318 | True         | OPT-exact [full]               |
| ('OPT-exact [full]', 'H1: customer mean P95 + margin', 'validation')           |             2.344 | True         | OPT-exact [full]               |
| ('OPT-exact [full]', 'H1: customer mean P95 + margin', 'test')                 |             2.421 | True         | OPT-exact [full]               |
| ('OPT-exact [full]', 'RF predict + margin', 'validation')                      |            -6.053 | False        | RF predict + margin            |
| ('OPT-exact [full]', 'RF predict + margin', 'test')                            |             1.757 | True         | OPT-exact [full]               |
| ('OPT-exact [full]', 'Always 100%', 'validation')                              |            16.963 | True         | Always 100%                    |
| ('OPT-exact [full]', 'Always 100%', 'test')                                    |            20.411 | False        | Always 100%                    |
| ('Greedy [full]', 'Cost-aware RF', 'validation')                               |            -7.974 | False        | Cost-aware RF                  |
| ('Greedy [full]', 'Cost-aware RF', 'test')                                     |            45.816 | False        | Greedy [full]                  |
| ('Greedy [full]', 'CART-MSE (26 feat.) [full]', 'validation')                  |             4.482 | True         | Greedy [full]                  |
| ('Greedy [full]', 'CART-MSE (26 feat.) [full]', 'test')                        |             4.035 | True         | Greedy [full]                  |
| ('Greedy [full]', 'H1: customer mean P95 + margin', 'validation')              |             3.338 | True         | Greedy [full]                  |
| ('Greedy [full]', 'H1: customer mean P95 + margin', 'test')                    |             3.450 | True         | Greedy [full]                  |
| ('Greedy [full]', 'RF predict + margin', 'validation')                         |             6.278 | True         | Greedy [full]                  |
| ('Greedy [full]', 'RF predict + margin', 'test')                               |             4.249 | True         | Greedy [full]                  |
| ('Greedy [full]', 'Always 100%', 'validation')                                 |            18.946 | True         | Always 100%                    |
| ('Greedy [full]', 'Always 100%', 'test')                                       |            22.441 | False        | Always 100%                    |
| ('Cost-aware RF', 'CART-MSE (26 feat.) [full]', 'validation')                  |             1.598 | True         | Cost-aware RF                  |
| ('Cost-aware RF', 'CART-MSE (26 feat.) [full]', 'test')                        |             0.832 | False        | Cost-aware RF                  |
| ('Cost-aware RF', 'H1: customer mean P95 + margin', 'validation')              |             0.780 | False        | Cost-aware RF                  |
| ('Cost-aware RF', 'H1: customer mean P95 + margin', 'test')                    |            -0.004 | False        | Cost-aware RF                  |
| ('Cost-aware RF', 'RF predict + margin', 'validation')                         |            -0.595 | False        | Cost-aware RF                  |
| ('Cost-aware RF', 'RF predict + margin', 'test')                               |            -3.245 | False        | Cost-aware RF                  |
| ('Cost-aware RF', 'Always 100%', 'validation')                                 |            23.492 | False        | Always 100%                    |
| ('Cost-aware RF', 'Always 100%', 'test')                                       |            23.228 | False        | Always 100%                    |
| ('CART-MSE (26 feat.) [full]', 'H1: customer mean P95 + margin', 'validation') |           -33.560 | False        | CART-MSE (26 feat.) [full]     |
| ('CART-MSE (26 feat.) [full]', 'H1: customer mean P95 + margin', 'test')       |            14.004 | True         | H1: customer mean P95 + margin |
| ('CART-MSE (26 feat.) [full]', 'RF predict + margin', 'validation')            |             3.623 | True         | RF predict + margin            |
| ('CART-MSE (26 feat.) [full]', 'RF predict + margin', 'test')                  |             3.849 | True         | RF predict + margin            |
| ('CART-MSE (26 feat.) [full]', 'Always 100%', 'validation')                    |            14.257 | True         | Always 100%                    |
| ('CART-MSE (26 feat.) [full]', 'Always 100%', 'test')                          |            16.393 | True         | Always 100%                    |
| ('H1: customer mean P95 + margin', 'RF predict + margin', 'validation')        |             1.994 | True         | RF predict + margin            |
| ('H1: customer mean P95 + margin', 'RF predict + margin', 'test')              |             2.672 | True         | RF predict + margin            |
| ('H1: customer mean P95 + margin', 'Always 100%', 'validation')                |            13.782 | True         | Always 100%                    |
| ('H1: customer mean P95 + margin', 'Always 100%', 'test')                      |            16.437 | True         | Always 100%                    |
| ('RF predict + margin', 'Always 100%', 'validation')                           |            17.245 | True         | Always 100%                    |
| ('RF predict + margin', 'Always 100%', 'test')                                 |            19.061 | True         | Always 100%                    |
| ('OPT-exact [full]', 'Always 25%', 'validation')                               |             1.698 | True         | OPT-exact [full]               |
| ('OPT-exact [full]', 'Always 50%', 'validation')                               |             1.793 | True         | OPT-exact [full]               |
| ('OPT-exact [full]', 'Always 75%', 'validation')                               |             0.529 | False        | OPT-exact [full]               |
| ('OPT-exact [full]', 'Always 100%', 'validation')                              |            16.963 | True         | Always 100%                    |
| ('OPT-exact [full]', 'Always 25%', 'test')                                     |             1.779 | True         | OPT-exact [full]               |
| ('OPT-exact [full]', 'Always 50%', 'test')                                     |             1.775 | True         | OPT-exact [full]               |
| ('OPT-exact [full]', 'Always 75%', 'test')                                     |            -0.452 | False        | OPT-exact [full]               |
| ('OPT-exact [full]', 'Always 100%', 'test')                                    |            20.411 | False        | Always 100%                    |

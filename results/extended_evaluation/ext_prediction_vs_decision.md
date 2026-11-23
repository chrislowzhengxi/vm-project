|                                                             |   test R2 |   test RMSE (pp) |   % saving b=2 |   % saving b=5 |   % saving b=10 |
|:------------------------------------------------------------|----------:|-----------------:|---------------:|---------------:|----------------:|
| ('Ridge', 'point -> tier')                                  |     0.494 |           25.794 |         10.627 |         14.684 |          -9.904 |
| ('Ridge', 'point + val-tuned margin')                       |     0.494 |           25.794 |         11.727 |         19.776 |          14.480 |
| ('CART', 'point -> tier')                                   |     0.491 |           25.864 |         10.796 |         11.496 |         -18.625 |
| ('CART', 'point + val-tuned margin')                        |     0.491 |           25.864 |         12.058 |         18.839 |          13.088 |
| ('RF', 'point -> tier')                                     |     0.525 |           24.993 |         11.696 |         14.242 |         -12.487 |
| ('RF', 'point + val-tuned margin')                          |     0.525 |           24.993 |         12.366 |         19.768 |          15.307 |
| ('RF', 'forest-weighted prescription (same RF)')            |     0.525 |          nan     |         13.250 |         21.434 |          16.505 |
| ('-', 'cost-aware RF (separate forest, shortfall targets)') |   nan     |          nan     |         12.945 |         21.593 |          16.737 |
| ('-', 'OPT depth 3 (no prediction)')                        |   nan     |          nan     |         11.529 |         20.098 |          15.323 |

| policy                                                |   vCPU-hour-weighted cost |   % reduction vs weighted best fixed |
|:------------------------------------------------------|--------------------------:|-------------------------------------:|
| A: point prediction -> tier [val-selected forest]     |                    0.8676 |                              13.2408 |
| B: point prediction + margin [val-selected forest]    |                    0.8286 |                              17.1353 |
| C: forest-weighted prescription [val-selected forest] |                    0.8027 |                              19.7270 |
| A: point prediction -> tier [main forest, msl=50]     |                    0.8560 |                              14.4012 |
| B: point prediction + margin [main forest, msl=50]    |                    0.8299 |                              17.0134 |
| C: forest-weighted prescription [main forest, msl=50] |                    0.8027 |                              19.7270 |
| Best fixed (val) (existing)                           |                    1.0000 |                               0.0000 |
| Greedy policy tree (existing)                         |                    0.8208 |                              17.9155 |
| OPT-exact (existing)                                  |                    0.8238 |                              17.6163 |
| Cost-aware RF (existing)                              |                    0.8067 |                              19.3337 |

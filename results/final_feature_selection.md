| feature             | label                                  |   val cost increase when permuted (beta=5) | selected for trees   |
|:--------------------|:---------------------------------------|-------------------------------------------:|:---------------------|
| sub_hist_mean       | subscription mean past P95 (%)         |                                     0.0448 | True                 |
| sub_hist_median     | subscription median past P95 (%)       |                                     0.0265 | True                 |
| sub_hist_frac_le75  | share of past VMs with P95 <= 75%      |                                     0.0249 | True                 |
| cores               | vCPU bucket                            |                                     0.0230 | True                 |
| sub_hist_frac_le50  | share of past VMs with P95 <= 50%      |                                     0.0112 | True                 |
| memory_gb           | memory bucket (GB)                     |                                     0.0094 | True                 |
| mem_per_core        | GB per vCPU                            |                                     0.0077 | True                 |
| dep_hist_mean       | deployment mean past P95 (%)           |                                     0.0062 | True                 |
| sub_hist_std        | subscription SD of past P95            |                                     0.0048 | False                |
| dep_hist_median     | deployment median past P95 (%)         |                                     0.0047 | False                |
| dep_batch_size      | VMs requested in same deployment slot  |                                     0.0039 | False                |
| sub_hist_frac_le25  | share of past VMs with P95 <= 25%      |                                     0.0037 | False                |
| sub_hist_last5_mean | mean P95 of last 5 completed VMs (%)   |                                     0.0033 | False                |
| hour                | creation hour                          |                                     0.0029 | False                |
| dep_n_prior_created | # earlier VMs (deployment)             |                                     0.0023 | False                |
| dep_n_hist          | # completed earlier VMs (deployment)   |                                     0.0020 | False                |
| sub_n_hist          | # completed earlier VMs (subscription) |                                     0.0020 | False                |
| sub_n_prior_created | # earlier VMs (subscription)           |                                     0.0018 | False                |
| sub_hist_last       | last completed VM's P95 (%)            |                                     0.0010 | False                |
| dep_hist_last       | deployment last completed P95 (%)      |                                     0.0006 | False                |
| sub_age_days        | subscription age (days)                |                                     0.0001 | False                |
| sub_preexisting     | subscription active at trace start     |                                     0.0000 | False                |
| dep_cold            | no completed deployment history        |                                     0.0000 | False                |
| sub_cold            | no completed subscription history      |                                     0.0000 | False                |
| day_in_trace        | day in trace                           |                                     0.0000 | False                |
| dow                 | creation weekday                       |                                    -0.0001 | False                |

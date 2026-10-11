# Downstream test: probe trained on the selected 5% of the training pool

3 grouped 70/30 splits, up to 3 sampler seeds each. The probe uses the OTHER model's features than the sampler (cross-model) or the same ones (same-model). Mean [95% bootstrap CI].

## cross-model, tag=scene

| sampler | macro-F1 | rare-class recall |
|---|---|---|
| all_data | 0.375 [0.358, 0.397] | 0.027 [0.000, 0.066] |
| random | 0.284 [0.274, 0.294] | 0.000 [0.000, 0.000] |
| kmeans_flat | 0.287 [0.278, 0.296] | 0.000 [0.000, 0.000] |
| semdedup | 0.284 [0.278, 0.291] | 0.000 [0.000, 0.000] |
| d4 | 0.284 [0.278, 0.291] | 0.000 [0.000, 0.000] |
| proto_hard | 0.278 [0.272, 0.283] | 0.000 [0.000, 0.000] |
| hier_sqrt | 0.291 [0.282, 0.299] | 0.000 [0.000, 0.000] |
| hier_balanced | 0.273 [0.263, 0.282] | 0.000 [0.000, 0.000] |
| kcenter | 0.285 [0.279, 0.292] | 0.000 [0.000, 0.000] |

## cross-model, tag=weather

| sampler | macro-F1 | rare-class recall |
|---|---|---|
| all_data | 0.515 [0.511, 0.518] | 0.000 [0.000, 0.000] |
| random | 0.231 [0.208, 0.252] | 0.000 [0.000, 0.000] |
| kmeans_flat | 0.256 [0.236, 0.275] | 0.000 [0.000, 0.000] |
| semdedup | 0.290 [0.279, 0.302] | 0.000 [0.000, 0.000] |
| d4 | 0.284 [0.270, 0.297] | 0.000 [0.000, 0.000] |
| proto_hard | 0.281 [0.253, 0.298] | 0.000 [0.000, 0.000] |
| hier_sqrt | 0.234 [0.215, 0.255] | 0.000 [0.000, 0.000] |
| hier_balanced | 0.281 [0.263, 0.298] | 0.000 [0.000, 0.000] |
| kcenter | 0.276 [0.262, 0.290] | 0.000 [0.000, 0.000] |

## cross-model, tag=time_of_day

| sampler | macro-F1 | rare-class recall |
|---|---|---|
| all_data | 0.789 [0.779, 0.798] | nan [nan, nan] |
| random | 0.636 [0.634, 0.638] | nan [nan, nan] |
| kmeans_flat | 0.633 [0.632, 0.634] | nan [nan, nan] |
| semdedup | 0.671 [0.659, 0.682] | nan [nan, nan] |
| d4 | 0.654 [0.642, 0.667] | nan [nan, nan] |
| proto_hard | 0.588 [0.490, 0.673] | nan [nan, nan] |
| hier_sqrt | 0.635 [0.633, 0.637] | nan [nan, nan] |
| hier_balanced | 0.656 [0.644, 0.669] | nan [nan, nan] |
| kcenter | 0.642 [0.622, 0.661] | nan [nan, nan] |

## cross-model, tag=condition

| sampler | macro-F1 | rare-class recall |
|---|---|---|
| all_data | 0.293 [0.281, 0.305] | 0.333 [0.322, 0.343] |
| random | 0.115 [0.106, 0.125] | 0.028 [0.016, 0.039] |
| kmeans_flat | 0.123 [0.113, 0.132] | 0.033 [0.024, 0.043] |
| semdedup | 0.131 [0.124, 0.137] | 0.054 [0.050, 0.059] |
| d4 | 0.129 [0.118, 0.138] | 0.060 [0.052, 0.069] |
| proto_hard | 0.117 [0.093, 0.140] | 0.056 [0.042, 0.074] |
| hier_sqrt | 0.118 [0.110, 0.126] | 0.036 [0.028, 0.043] |
| hier_balanced | 0.132 [0.125, 0.139] | 0.096 [0.072, 0.125] |
| kcenter | 0.127 [0.117, 0.136] | 0.054 [0.048, 0.061] |

## same-model, tag=scene

| sampler | macro-F1 | rare-class recall |
|---|---|---|
| all_data | 0.375 [0.358, 0.397] | 0.027 [0.000, 0.066] |
| random | 0.284 [0.274, 0.295] | 0.000 [0.000, 0.000] |
| kmeans_flat | 0.286 [0.272, 0.299] | 0.000 [0.000, 0.000] |
| semdedup | 0.284 [0.272, 0.296] | 0.000 [0.000, 0.000] |
| d4 | 0.284 [0.272, 0.297] | 0.000 [0.000, 0.000] |
| proto_hard | 0.278 [0.261, 0.298] | 0.000 [0.000, 0.000] |
| hier_sqrt | 0.293 [0.280, 0.305] | 0.000 [0.000, 0.000] |
| hier_balanced | 0.275 [0.261, 0.289] | 0.000 [0.000, 0.000] |
| kcenter | 0.280 [0.266, 0.295] | 0.000 [0.000, 0.000] |

## same-model, tag=weather

| sampler | macro-F1 | rare-class recall |
|---|---|---|
| all_data | 0.515 [0.511, 0.518] | 0.000 [0.000, 0.000] |
| random | 0.231 [0.208, 0.252] | 0.000 [0.000, 0.000] |
| kmeans_flat | 0.263 [0.235, 0.288] | 0.000 [0.000, 0.000] |
| semdedup | 0.272 [0.248, 0.299] | 0.000 [0.000, 0.000] |
| d4 | 0.277 [0.257, 0.297] | 0.000 [0.000, 0.000] |
| proto_hard | 0.262 [0.230, 0.294] | 0.000 [0.000, 0.000] |
| hier_sqrt | 0.241 [0.221, 0.261] | 0.000 [0.000, 0.000] |
| hier_balanced | 0.273 [0.247, 0.299] | 0.000 [0.000, 0.000] |
| kcenter | 0.265 [0.243, 0.285] | 0.000 [0.000, 0.000] |

## same-model, tag=time_of_day

| sampler | macro-F1 | rare-class recall |
|---|---|---|
| all_data | 0.789 [0.779, 0.798] | nan [nan, nan] |
| random | 0.636 [0.634, 0.638] | nan [nan, nan] |
| kmeans_flat | 0.636 [0.634, 0.638] | nan [nan, nan] |
| semdedup | 0.680 [0.664, 0.696] | nan [nan, nan] |
| d4 | 0.673 [0.657, 0.690] | nan [nan, nan] |
| proto_hard | 0.578 [0.518, 0.627] | nan [nan, nan] |
| hier_sqrt | 0.635 [0.633, 0.639] | nan [nan, nan] |
| hier_balanced | 0.656 [0.642, 0.669] | nan [nan, nan] |
| kcenter | 0.662 [0.646, 0.678] | nan [nan, nan] |

## same-model, tag=condition

| sampler | macro-F1 | rare-class recall |
|---|---|---|
| all_data | 0.293 [0.281, 0.305] | 0.333 [0.322, 0.343] |
| random | 0.115 [0.106, 0.125] | 0.028 [0.016, 0.039] |
| kmeans_flat | 0.125 [0.114, 0.136] | 0.039 [0.026, 0.053] |
| semdedup | 0.132 [0.122, 0.142] | 0.064 [0.057, 0.071] |
| d4 | 0.132 [0.123, 0.142] | 0.064 [0.057, 0.073] |
| proto_hard | 0.124 [0.110, 0.137] | 0.069 [0.053, 0.091] |
| hier_sqrt | 0.118 [0.109, 0.127] | 0.037 [0.027, 0.047] |
| hier_balanced | 0.132 [0.122, 0.143] | 0.091 [0.067, 0.122] |
| kcenter | 0.129 [0.121, 0.138] | 0.058 [0.050, 0.065] |

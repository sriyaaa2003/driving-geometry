# ZOD downstream test: probe trained on the selected 5% of the training pool

3 grouped 70/30 splits (group = country + day), 3 sampler seeds each. Cross-model: the probe uses the OTHER model's features; same-model: the same model's. Mean [95% bootstrap CI]. `all_data` trains on the whole pool.

## cross-model, tag=condition

| sampler | macro-F1 | rare-class recall |
|---|---|---|
| all_data | 0.807 [0.788, 0.825] | 0.688 [0.659, 0.715] |
| random | 0.734 [0.712, 0.755] | 0.506 [0.473, 0.538] |
| kmeans_flat | 0.736 [0.715, 0.756] | 0.513 [0.483, 0.540] |
| semdedup | 0.603 [0.580, 0.633] | 0.239 [0.169, 0.313] |
| d4 | 0.592 [0.570, 0.616] | 0.212 [0.144, 0.282] |
| proto_hard | 0.523 [0.514, 0.531] | 0.109 [0.003, 0.222] |
| hier_sqrt | 0.728 [0.708, 0.746] | 0.484 [0.448, 0.515] |
| hier_balanced | 0.676 [0.645, 0.706] | 0.384 [0.318, 0.443] |
| kcenter | 0.555 [0.547, 0.564] | 0.140 [0.074, 0.215] |

## cross-model, tag=road_type

| sampler | macro-F1 | rare-class recall |
|---|---|---|
| all_data | 0.624 [0.616, 0.632] | nan [nan, nan] |
| random | 0.594 [0.586, 0.601] | nan [nan, nan] |
| kmeans_flat | 0.590 [0.584, 0.596] | nan [nan, nan] |
| semdedup | 0.527 [0.516, 0.537] | nan [nan, nan] |
| d4 | 0.520 [0.508, 0.532] | nan [nan, nan] |
| proto_hard | 0.467 [0.435, 0.499] | nan [nan, nan] |
| hier_sqrt | 0.585 [0.578, 0.592] | nan [nan, nan] |
| hier_balanced | 0.551 [0.542, 0.560] | nan [nan, nan] |
| kcenter | 0.533 [0.522, 0.544] | nan [nan, nan] |

## cross-model, tag=country

| sampler | macro-F1 | rare-class recall |
|---|---|---|
| all_data | 0.463 [0.428, 0.503] | 0.234 [0.110, 0.354] |
| random | 0.377 [0.367, 0.389] | 0.082 [0.053, 0.111] |
| kmeans_flat | 0.376 [0.367, 0.385] | 0.070 [0.046, 0.094] |
| semdedup | 0.357 [0.347, 0.368] | 0.072 [0.048, 0.094] |
| d4 | 0.356 [0.345, 0.367] | 0.064 [0.041, 0.088] |
| proto_hard | 0.335 [0.313, 0.357] | 0.069 [0.032, 0.114] |
| hier_sqrt | 0.380 [0.367, 0.391] | 0.096 [0.062, 0.129] |
| hier_balanced | 0.362 [0.349, 0.376] | 0.085 [0.050, 0.122] |
| kcenter | 0.355 [0.344, 0.366] | 0.069 [0.044, 0.094] |

## same-model, tag=condition

| sampler | macro-F1 | rare-class recall |
|---|---|---|
| all_data | 0.807 [0.788, 0.825] | 0.688 [0.659, 0.715] |
| random | 0.734 [0.712, 0.755] | 0.506 [0.471, 0.539] |
| kmeans_flat | 0.736 [0.714, 0.757] | 0.512 [0.475, 0.547] |
| semdedup | 0.618 [0.589, 0.650] | 0.276 [0.205, 0.347] |
| d4 | 0.597 [0.579, 0.617] | 0.229 [0.161, 0.297] |
| proto_hard | 0.523 [0.511, 0.534] | 0.116 [0.003, 0.238] |
| hier_sqrt | 0.728 [0.708, 0.747] | 0.482 [0.446, 0.519] |
| hier_balanced | 0.677 [0.649, 0.707] | 0.385 [0.313, 0.450] |
| kcenter | 0.583 [0.563, 0.607] | 0.187 [0.105, 0.271] |

## same-model, tag=road_type

| sampler | macro-F1 | rare-class recall |
|---|---|---|
| all_data | 0.624 [0.616, 0.632] | nan [nan, nan] |
| random | 0.594 [0.586, 0.601] | nan [nan, nan] |
| kmeans_flat | 0.590 [0.584, 0.596] | nan [nan, nan] |
| semdedup | 0.525 [0.517, 0.534] | nan [nan, nan] |
| d4 | 0.514 [0.506, 0.522] | nan [nan, nan] |
| proto_hard | 0.459 [0.429, 0.483] | nan [nan, nan] |
| hier_sqrt | 0.587 [0.581, 0.593] | nan [nan, nan] |
| hier_balanced | 0.550 [0.542, 0.558] | nan [nan, nan] |
| kcenter | 0.531 [0.524, 0.538] | nan [nan, nan] |

## same-model, tag=country

| sampler | macro-F1 | rare-class recall |
|---|---|---|
| all_data | 0.463 [0.428, 0.503] | 0.234 [0.110, 0.354] |
| random | 0.377 [0.366, 0.389] | 0.082 [0.053, 0.109] |
| kmeans_flat | 0.375 [0.365, 0.386] | 0.071 [0.046, 0.094] |
| semdedup | 0.360 [0.345, 0.374] | 0.080 [0.047, 0.112] |
| d4 | 0.363 [0.347, 0.378] | 0.081 [0.046, 0.116] |
| proto_hard | 0.329 [0.304, 0.355] | 0.035 [0.007, 0.064] |
| hier_sqrt | 0.377 [0.366, 0.388] | 0.085 [0.055, 0.111] |
| hier_balanced | 0.361 [0.348, 0.374] | 0.072 [0.042, 0.104] |
| kcenter | 0.360 [0.345, 0.374] | 0.071 [0.045, 0.100] |

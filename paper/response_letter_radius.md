# Response Letter Draft: Hexagonal Radius Control

## Reviewer Comment

> Add a control that shrinks the hexagonal neighborhood radius. If a tighter-radius hexagonal map closes most of the QE gap, the graph topologies are winning on looseness, not structure.

## Response

We thank the reviewer for raising this point. We agree that neighborhood scale is an important potential confound and that the original manuscript did not make the radius treatment explicit enough. However, the topology comparison was not a tuned MST/RNG versus default-radius hexagonal comparison. The Optuna campaign optimized `initial_radius` for every topology family, including hexagonal, using the same search interval and the same tuning budget. Therefore, the hexagonal comparator was already free to adopt a tighter neighborhood radius if that improved quantization error.

To make this clear, we revised the Optuna benchmark protocol and topology-results sections. We now state explicitly that `initial_radius` was included in the Optuna search space for hexagonal, MST, and RNG runs. We also report the distilled selected radii, which show that the tuned hexagonal comparator used the tightest radius among the topology families: 1.03 for hexagonal under full sampling, compared with 1.46 for MST and 1.41 for RNG; and 1.17 for hexagonal under random sampling, compared with 1.82 for MST and 1.77 for RNG.

This addresses the fairness concern: the observed MST/RNG QE gains are not obtained by comparing graph topologies against an untuned or artificially broad hexagonal neighborhood radius. Instead, the results are best-observed-versus-best-observed comparisons under a matched Optuna budget. We have revised the manuscript to avoid leaving this implicit.

## Manuscript Amendments

In Section 4.1, we added:

> "The neighborhood-radius search space was shared across topology families. In particular, `initial_radius` was an Optuna-optimized parameter for hexagonal, MST, and RNG runs, with the same search interval of 0.5 to 10.0 in each case. Thus, the hexagonal topology comparisons below use a tuned hexagonal comparator rather than a default-radius hexagonal baseline."

In Section 5.3, we added:

> "Because `initial_radius` was tuned for every topology family, the topology comparison is a best-observed-versus-best-observed comparison under the same Optuna budget rather than a comparison against an untuned hexagonal radius. The distilled deployable defaults selected tighter hexagonal radii than the graph topologies: under full sampling the selected `initial_radius` values were 1.03 for hexagonal, 1.46 for MST, and 1.41 for RNG, while under random sampling they were 1.17, 1.82, and 1.77, respectively. These values show that the reported MST/RNG $QE$ gains are not explained by evaluating hexagonal only at a broader default neighborhood radius."

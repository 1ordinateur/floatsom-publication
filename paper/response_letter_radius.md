# Response Letter Draft: Reviewer 1

We thank the reviewer for the careful reading and for separating the systems contribution from the quality/topology claims. We agree that the systems evidence is stronger in the submitted version than the topology-quality evidence, and the revision is structured to make that distinction explicit. Where the manuscript already contained the relevant analysis but did not state the point clearly enough, we have clarified the text. Where the reviewer identified a missing control or missing reporting detail, we have added or planned the corresponding amendment below.

## 1. Quality claims rely on QE alone

### Reviewer Comment

> The quality claim runs on QE alone. QE measures distance to the best-matching unit, not neighborhood preservation, which is the point of a SOM. No topographic error or trustworthiness, even though Forest et al., 2020 is cited and implements them.

### Response

We agree with the reviewer that quantization error alone does not test the topology-preservation property of a SOM. The submitted manuscript used $QE$ as the principal deployment metric because the benchmark was designed around large-scale vector quantization and because $QE$ is the metric used in our XPySOM calibration. However, the reviewer is correct that lower $QE$ under MST or RNG could reflect a looser vector-quantizer-like fit rather than better preservation of neighborhood structure.

We will therefore revise the topology comparison to either report preservation metrics alongside $QE$ or, if the additional runs are not completed in time for the revision, explicitly narrow the claim to quantization performance rather than topology preservation. Our preferred revision is to add topographic error and trustworthiness to the topology comparison, using the existing FloatSOM metric infrastructure and the same matched dataset/seed/split units used for the $QE$ analyses.

### Manuscript Amendment

In Section 4.1, we will add:

> "For topology comparisons, we report quantization error together with preservation-oriented diagnostics. Topographic error measures whether the first and second best-matching units are adjacent under the evaluated SOM topology, while trustworthiness measures local neighborhood preservation between the input space and the map representation. These metrics are reported alongside $QE$ because a lower $QE$ alone does not establish that a SOM preserves neighborhood structure."

In Section 5.3, we will revise the opening sentence to:

> "Topology comparisons are reported with $QE$ and preservation-oriented metrics, with the Optuna hexagonal batch setting as the primary regular-topology baseline; Fig. 5 provides a qualitative illustration of the neighborhood structures produced by hexagonal, MST, and RNG."

If the preservation-metric runs are not included in the final revision, we will instead use the narrower wording:

> "Topology comparisons in this version are therefore interpreted as quantization-performance comparisons, not as a complete demonstration of improved topology preservation."

## 2. Hexagonal neighborhood-radius control

### Reviewer Comment

> Add a control that shrinks the hexagonal neighborhood radius. If a tighter-radius hexagonal map closes most of the QE gap, the graph topologies are winning on looseness, not structure.

### Response

We agree that neighborhood radius is an important potential confound. However, the submitted topology comparison was not a tuned MST/RNG versus default-radius hexagonal comparison. The Optuna campaign optimized `initial_radius` for every topology family, including hexagonal, using the same search interval and the same tuning budget. Therefore, the hexagonal comparator was already free to adopt a tighter neighborhood radius if that improved quantization error.

To make this clear, we revised the Optuna benchmark protocol and topology-results sections. We now state explicitly that `initial_radius` was included in the Optuna search space for hexagonal, MST, and RNG runs. We also report the distilled selected radii, which show that the tuned hexagonal comparator used the tightest radius among the topology families: 1.03 for hexagonal under full sampling, compared with 1.46 for MST and 1.41 for RNG; and 1.17 for hexagonal under random sampling, compared with 1.82 for MST and 1.77 for RNG.

This addresses the radius-control concern: the observed MST/RNG $QE$ gains are not obtained by comparing graph topologies against an untuned or artificially broad hexagonal neighborhood radius. Instead, the results are best-observed-versus-best-observed comparisons under a matched Optuna budget.

### Manuscript Amendment

In Section 4.1, we added:

> "The neighborhood-radius search space was shared across topology families. In particular, `initial_radius` was an Optuna-optimized parameter for hexagonal, MST, and RNG runs, with the same search interval of 0.5 to 10.0 in each case. Thus, the hexagonal topology comparisons below use a tuned hexagonal comparator rather than a default-radius hexagonal baseline."

In Section 5.3, we added:

> "Because `initial_radius` was tuned for every topology family, the topology comparison is a best-observed-versus-best-observed comparison under the same Optuna budget rather than a comparison against an untuned hexagonal radius. The distilled deployable defaults selected tighter hexagonal radii than the graph topologies: under full sampling the selected `initial_radius` values were 1.03 for hexagonal, 1.46 for MST, and 1.41 for RNG, while under random sampling they were 1.17, 1.82, and 1.77, respectively. These values show that the reported MST/RNG $QE$ gains are not explained by evaluating hexagonal only at a broader default neighborhood radius."

## 3. Related work and external baselines

### Reviewer Comment

> Related work names prior graph and GPU SOMs but does not benchmark against any. aweSOM (Ha et al., JOSS 2025), a GPU Python SOM in the same N > 10^6 range, is not mentioned.

> Add recent GPU and distributed SOM baselines. Somoclu CUDA, GigaSOM, aweSOM. aweSOM is single-node CPU/GPU with ensemble stacking, so it is a quality baseline, not a distributed competitor.

### Response

We agree that the related-work coverage should be expanded and that aweSOM should be discussed. We will add aweSOM as a recent Python CPU/GPU SOM implementation, but we will not add it as a new executable benchmark. We will instead cite the scaling results reported in the aweSOM paper. Those reported numbers are not close to the FloatSOM large-scale performance regime, and aweSOM is not a stronger runtime comparator than XPySOM for the Python GPU batch-SOM setting considered here.

For executable benchmarking, we retain XPySOM as the direct external implementation baseline because it is the closest Python GPU batch-SOM comparator and because the XPySOM paper already benchmarks against earlier open-source SOM implementations and reports large speed advantages over those alternatives. Since aweSOM's own literature numbers do not exceed XPySOM's relevance as the direct comparator, re-running aweSOM would not change the main systems comparison.

We will discuss Somoclu and GigaSOM more explicitly. Somoclu is an important CUDA/MPI SOM system, but it is not the closest baseline for the Python GPU batch-SOM workflow evaluated here. GigaSOM.jl is also important, especially for large cytometry workloads, but it is implemented in Julia and is not readily comparable as a drop-in Python baseline in our benchmark harness. We therefore discuss GigaSOM as a distributed large-scale SOM system precedent rather than as a directly benchmarked external Python implementation.

### Manuscript Amendment

In Section 2.1, we will revise the related-work paragraph to:

> "Open-source SOM libraries range from lightweight Python implementations to more performance-oriented systems. MiniSom is a compact Python implementation of the classical online regime [@vettigliJustGlowingMinisom2018], whereas XPySOM is a Python-based batch SOM implementation designed for efficient GPU-backed execution [@manciniXPySomHighPerformanceSelfOrganizing2020]. aweSOM is a recent Python CPU/GPU SOM implementation with ensemble stacking that targets large single-node workloads; we discuss it through its reported literature scaling rather than as a direct benchmark because its published performance regime is below the large-scale distributed setting evaluated here and it is not a stronger direct comparator than XPySOM. At larger scales, Somoclu and GigaSOM provide mature parallel SOM systems for large workloads [@wittekSomocluEfficientParallel2017; @kratochvilGigaSOMjlHighperformanceClustering2020]."

In Section 4.1 or Section 7, we will add:

> "We selected XPySOM as the direct external implementation baseline because it is the closest Python GPU batch-SOM comparator and because prior XPySOM benchmarking already established strong runtime performance relative to earlier open-source SOM implementations. Somoclu, GigaSOM, and aweSOM are therefore discussed as important systems precedents using their reported literature results, but they are not directly benchmarked in this Python-centered deployment comparison; in particular, GigaSOM.jl is Julia-based and not readily integrated into the same benchmark harness."

## 4. Numbers should be embedded in the paper

### Reviewer Comment

> Quality numbers are not in the paper. Per-dataset effect sizes exist only as external .tsv paths (Tables S4 to S11 are captions pointing at files). They cannot be read off the forest plots either.

> Put the numbers in the paper. Embed the per-dataset effect-size tables instead of external .tsv paths.

### Response

We agree. The submitted version included the tables as reproducibility artifacts but did not embed their numerical contents in the manuscript, which makes the paper harder to evaluate independently. We will replace the path-only supplementary table captions with embedded tables for the per-dataset effect summaries and runtime summaries. The external `.tsv` files can remain as machine-readable artifacts, but the manuscript should contain the actual values needed to evaluate the claims.

### Manuscript Amendment

In the Supplementary Tables section, we will replace the path-only captions with embedded tables. For example, the revised captions will read:

> "Supplementary Table S7. Paired topology comparison p-values and effect summaries for hexagonal versus MST and hexagonal versus RNG across balanced $QE$, holdout $QE$, and train $QE$. Rows list metric/dataset entries, including the OVERALL row; the numerical values are embedded below and the corresponding machine-readable table is also provided in `assets/tables/supp_table_topology_hex_vs_mst_rng_pvalues.tsv`."

For the deployment comparison, we will revise the caption to:

> "Supplementary Table S8. Figure 13 deployment comparison percent summary for tuned FloatSOM RNG versus default hexagonal XPySOM across $QE_B$, $QE_H$, and $QE_T$. Rows list per-dataset and `GLOBAL_OVERALL` entries with the plotted median percent change and 95% confidence interval; the numerical values are embedded below and the corresponding machine-readable table is also provided in `assets/tables/supp_table_figure_13_xpysom_rng_deployment_summary.tsv`."

## 5. Deployment comparison separates topology, tuning, and implementation

### Reviewer Comment

> Deployment comparison (Section 7, Figure 13) is a different claim. Tuned FloatSOM RNG against untuned hexagonal XPySOM. One number, three changes: topology, tuning, implementation. The 14.5, 9.1, 22.5 percent gains cannot be credited to topology. Sections 5.1, 5.3, and 5.4 already have the pieces to separate them.

> Separate topology from tuning and implementation in the deployment comparison, or tune both systems.

### Response

We agree that Fig. 13 should not be read as attributing the full gain to topology alone. Our intent was to show the integrated deployment comparison between a practical default XPySOM run and the recommended tuned FloatSOM RNG configuration. The submitted manuscript already separates several components: Section 5.1 calibrates FloatSOM against XPySOM under matched hexagonal settings, Section 5.3 compares topologies inside FloatSOM under a matched Optuna budget, and Section 5.4 evaluates tuning relative to an untuned reference. However, Section 7 did not state this decomposition clearly enough.

We will revise Section 7 and the Fig. 13 caption so that the 14.5%, 9.1%, and 22.5% improvements are described as an integrated deployment effect, not a topology-only effect. We will also point readers to the existing tuned-hexagonal and tuned-MST deployment figures in the Supplementary material so that the effect of topology can be read separately from the effect of tuning and implementation.

### Manuscript Amendment

In Section 7, we will revise the opening paragraph to:

> "Fig. 13 is an integrated deployment comparison rather than a topology-only attribution. It compares the untuned default hexagonal XPySOM workflow against the recommended tuned FloatSOM RNG workflow, so the reported difference includes implementation, hyperparameter tuning, and topology choice. The components are separated in the preceding analyses: Section 5.1 calibrates FloatSOM and XPySOM under matched hexagonal settings, Section 5.3 compares hexagonal, MST, and RNG inside FloatSOM under the same Optuna budget, and Section 5.4 evaluates tuned configurations against the untuned reference."

We will revise the Fig. 13 caption to:

> "Figure 13. Integrated deployment comparison of default hexagonal XPySOM versus tuned FloatSOM RNG. The comparison intentionally combines implementation, tuning, and topology choice and should not be interpreted as attributing the full difference to topology alone."

## 6. MST and RNG novelty claims

### Reviewer Comment

> Drop "novel" for the topologies. Jang et al., 2009, already in the references, put MSTs on SOMs. Frame this as the first version that scales on GPUs.

### Response

We agree that MST itself should not be described as a novel SOM topology because prior work, including Jang et al., used MSTs in SOMs. We will revise the manuscript to remove broad novelty language for MSTs and instead frame the contribution as a scalable GPU implementation and large-scale evaluation of dynamic graph-based SOM topologies.

For RNG, we will retain a narrower and qualified novelty claim. We have not identified prior work applying dynamically refreshed Relative Neighborhood Graphs as the neighborhood topology in SOM training. To avoid overclaiming, we will phrase this as "to our knowledge" and distinguish it from the non-novel MST component.

### Manuscript Amendment

In the Abstract, we will replace:

> "novel topologies beyond regular lattices"

with:

> "scalable graph-based topologies beyond regular lattices"

In Sections 1 and 3.2, we will add:

> "Prior work has used MST-based neighborhoods in SOMs, so we do not claim MST itself as a novel topology. Our contribution for MST is a scalable GPU-compatible dynamic implementation and large-scale evaluation. In contrast, to our knowledge, dynamically refreshed Relative Neighborhood Graph neighborhoods have not previously been used as a SOM training topology."

In the Discussion, we will replace broad "novel graph-based topology" wording with:

> "FloatSOM combines scalable graph-based topology support with out-of-memory execution and distributed multi-GPU training."

## 7. Runtime cost of RNG recommendation

### Reviewer Comment

> Measure the runtime cost of the recommended topology. Section 8.5 recommends RNG. At grid size 64, RNG is 27x hexagonal, MST is 8x. The MST distances-to-CPU step for Kruskal is the cheap part. The all-pairs hop-distance step both share plus RNG's blocker test is the expensive part. Put a number on the RNG cost.

### Response

We agree that any recommendation of RNG must be paired with its runtime cost. No additional end-to-end benchmarking is needed to answer this reviewer request because the manuscript already reports this cost in Fig. 12 and the topology-runtime discussion: at grid size 64, the 8-GPU mean runtime was 32.54 s for hexagonal, 266.45 s for MST, and 880.83 s for RNG, corresponding to 8.19x and 27.07x the hexagonal runtime for MST and RNG, respectively. The needed revision is to move this number into the recommendation/discussion context so the cost is visible at the point where RNG is recommended. A separate component-level microbenchmark of the RNG blocker test and all-pairs hop-distance step would be useful for diagnosis, but it is not necessary to satisfy the reviewer's request to "put a number on the RNG cost."

### Manuscript Amendment

In Section 8.5, we will add:

> "The RNG recommendation is conditional on the user accepting its topology-construction overhead. In the grid-size scaling benchmark, the largest grid size tested (64) required 32.54 s for hexagonal, 266.45 s for MST, and 880.83 s for RNG on 8 GPUs, making RNG 27.07x slower than hexagonal and MST 8.19x slower than hexagonal at that point. RNG should therefore be preferred when quantization performance is the priority and graph-refresh cost is acceptable; hexagonal remains the appropriate default when throughput dominates."

## 8. Multiple-comparison correction

### Reviewer Comment

> State whether the paired t-tests were corrected for multiple comparisons. About 42 per topology, 14 datasets by 3 metrics. Seed counts and intervals are already there.

### Response

We agree that the statistical reporting should state whether the paired tests were multiplicity-corrected. On checking the analysis code, the publication-figure pipeline already computes Benjamini-Hochberg adjusted `q_value` fields for dataset-level summaries. However, the current manuscript-facing topology p-value table exports only formatted raw `p_value` entries, and some figure significance markers still threshold raw `p_value` even when `q_value` is available. The revision will therefore expose the already computed adjusted values and state the correction convention explicitly.

### Manuscript Amendment

In Section 4.4, we will add:

> "For dataset-level families of related paired tests, we compute Benjamini-Hochberg adjusted q-values in addition to raw paired $t$-test p-values. The adjustment is applied across non-global dataset rows within each comparison family. Global pooled rows are reported separately as overall summaries and are not included in the dataset-level adjustment family."

In the supplementary topology table captions, we will revise:

> "Rows report raw paired $t$-test p-values and Benjamini-Hochberg adjusted q-values for the corresponding dataset-level comparison family; OVERALL rows are pooled summaries and are shown separately from the dataset-level adjustment."

## 9. Dead-node and node-utilization reporting

### Reviewer Comment

> Add a dead-node or node-utilization count across the three topologies. Cheap, and it bears on the preservation question.

### Response

We agree. Node utilization is a useful diagnostic for determining whether lower $QE$ comes from broadly used map capacity or from uneven allocation in which some nodes are effectively unused. We will add a node-utilization summary across hexagonal, MST, and RNG using the same matched units as the topology comparison.

### Manuscript Amendment

In Section 4.1, we will add:

> "We additionally report node utilization, defined as the fraction of SOM nodes selected as the best-matching unit by at least one evaluated sample. The complementary dead-node fraction is one minus this utilization. This diagnostic helps distinguish broadly used map capacity from solutions in which lower $QE$ is accompanied by unused nodes."

In Section 5.3, we will add:

> "Node-utilization summaries are reported alongside the topology metrics to test whether graph-based improvements coincide with higher or lower use of the available SOM nodes."

## 10. HDSSSOM framing

### Reviewer Comment

> Keep the HDSSSOM results framed as the pilot they already are.

### Response

We agree and will preserve the current framing. The submitted manuscript already describes HDSSSOM as a focused screening pilot and not as part of the main full-versus-random sampling benchmark. We will keep that language and avoid broadening the claim beyond the pilot configuration.

### Manuscript Amendment

The current Section 5.2 text already states:

> "We first report a focused HDSSSOM pilot as an elimination comparison rather than as part of the broader sampling benchmark. This pilot used a smaller, more limited configuration than the later full versus random analysis and is summarized in Supplementary Table S2."

We will retain this framing and, if needed, add:

> "These HDSSSOM results should therefore be interpreted as a pilot screen under the stated configuration rather than as a comprehensive evaluation of all possible HDSSSOM schedules."

## 11. Scaling efficiency and extrapolated 1-GPU baselines

### Reviewer Comment

> Some scaling points are extrapolated 1-GPU values, not measured (Section 4.2). Stated in the paper. Still limits the efficiency panels.

### Response

We agree. The submitted manuscript already states that some efficiency denominators are locally extrapolated from the last available 1-GPU point when direct 1-GPU runs were unavailable. We will keep this limitation visible in the results and discussion and avoid interpreting the above-100% efficiency panels as pure compute scaling.

### Manuscript Amendment

The current Section 6.2 text states:

> "When a direct 1-GPU baseline was unavailable at a given axis value, the efficiency denominator was constructed by local linear extrapolation from the last available 1-GPU point on that curve (Section 4.2), so some values should be interpreted with care if the underlying 1-GPU runtime is nonlinear over that range."

We will retain this statement and add to the Discussion:

> "Efficiency values above 100% should be interpreted as a combined consequence of parallelism and a changed memory/data-staging regime, not as evidence of superlinear compute scaling."

# Response Letter Draft: Reviewer 1

We thank the reviewer for the careful reading and for separating the systems contribution from the quality/topology claims. We agree that the systems evidence is stronger in the submitted version than the topology-quality evidence, and the revision is structured to make that distinction explicit. Where the manuscript already contained the relevant analysis but did not state the point clearly enough, we have clarified the text. Where the reviewer identified a missing control or missing reporting detail, we have added the corresponding amendment below.

## 1. Quality claims rely on QE alone

### Reviewer Comment

> The quality claim runs on QE alone. QE measures distance to the best-matching unit, not neighborhood preservation, which is the point of a SOM. No topographic error or trustworthiness, even though Forest et al., 2020 is cited and implements them.

### Response

We agree with the reviewer that quantization error alone does not test the topology-preservation property of a SOM. The submitted manuscript used $QE$ as the principal deployment metric because the benchmark was designed around large-scale vector quantization and because $QE$ is the metric used in our XPySOM calibration. However, the reviewer is correct that lower $QE$ under MST or RNG could reflect a looser vector-quantizer-like fit rather than better preservation of neighborhood structure.

We have therefore revised the topology comparison to report a preservation metric alongside $QE$. We do not use raw topographic error as the primary cross-topology statistic, because raw topographic error defines an error by whether the first and second best-matching units are immediate neighbors on the map. That adjacency relation is itself topology-dependent. Even when map size and output dimensionality are fixed, hexagonal, MST, and RNG maps have different connectivity and degree structure, so raw topographic error would partly measure the graph's one-hop neighbor convention rather than only topology preservation.

This is not only a concern raised by Ramos et al. [@ramosROLELATTICEDIMENSIONALITY2018]. Neme and Miramontes further showed that topographic error is affected by statistical properties of the neuron lattice, including path length, clustering, and connectivity length [@nemeStatisticalPropertiesLattices2005]. Other SOM comparisons, including Machon-Gonzalez and Lopez-Garcia, also caution that topographic-error comparisons require the same map size because the errors depend on map design [@machon-gonzalezFLSOMIndividualKernel2010]. These results support the narrower point relevant here: holding map size fixed is necessary, but holding map size or output dimensionality fixed is not sufficient when the adjacency graph itself changes.

Instead, we added Mean Tied Rank (MTR), following the tied-rank logic proposed by Ramos et al. for comparing SOMs with different topologies [@ramosROLELATTICEDIMENSIONALITY2018]. For each sample, we compute the first and second BMUs, rank all non-winning units by graph shortest-path distance from the first BMU, assign average ranks to tied graph-distance groups, and record the tied rank of the second BMU. Lower MTR indicates that the second-best prototype is topologically close to the winning prototype. Because all topology comparisons use the same map size, no map-size normalization is required. We report MTR next to $QE$ and interpret the two metrics separately.

We also added matched post hoc topology diagnostics for the final trained FloatSOM maps. These diagnostics evaluate the same dataset, seed, sampling, and topology units, so the reviewer can see whether graph topologies improve $QE$ while preserving local topology and using map nodes effectively.

### Manuscript Amendment

In Section 4.1, we added:

> "For cross-topology preservation diagnostics, we report Mean Tied Rank (MTR), following the tied-rank approach proposed for comparing SOMs with different topologies [@ramosROLELATTICEDIMENSIONALITY2018], rather than raw topographic error. For each sample $x_i$, let $b_i^{(1)}$ and $b_i^{(2)}$ denote the first and second best-matching units. We rank all non-winning units by graph shortest-path distance from $b_i^{(1)}$, assigning the average ordinal rank to units tied at the same graph-distance shell. If $b_i^{(2)}$ lies in shell $S_d$ and $L_d$ non-winning units are in closer shells, its tied rank is $\tau_i=L_d+(|S_d|+1)/2$, and $MTR=N^{-1}\sum_i \tau_i$. Lower MTR indicates that the second-best prototype remains topologically close to the winning prototype. We do not use raw topographic error as the primary cross-topology statistic because its one-hop adjacency criterion changes with the evaluated graph's connectivity and degree structure; prior work has shown that topographic error depends on map topology, lattice statistical properties, and map design choices such as size [@ramosROLELATTICEDIMENSIONALITY2018; @nemeStatisticalPropertiesLattices2005; @machon-gonzalezFLSOMIndividualKernel2010]."

In Section 5.3, we revised the opening sentence to:

> "Topology comparisons retain $QE$ as the primary optimized endpoint, with the Optuna hexagonal batch setting as the primary regular-topology baseline; Fig. 5 provides a qualitative illustration of the neighborhood structures produced by hexagonal, MST, and RNG. To test whether graph-topology $QE$ gains reflect useful topology behavior rather than only looser vector quantization, the same final maps are also evaluated post hoc for MTR, node utilization, and dead-node fraction under matched tuned and untuned/default profiles."

In the Discussion, we added:

> "$QE$ and MTR are interpreted as complementary quantities: $QE$ measures vector-quantization fidelity, whereas MTR evaluates whether the two closest prototypes for a sample remain close under the topology-induced graph distance. We therefore avoid treating a lower $QE$ alone as evidence of improved topology preservation."

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

We agree that the related-work coverage should be expanded and that aweSOM should be discussed. We attempted to benchmark aweSOM on our standard speed workload ($10^7$ samples, 50 dimensions, and a $32 \times 32$ map). Using aweSOM's standard training configuration and only one online update step per sample ($N$ updates), the run consistently reached our 30-minute timeout (1800 s). A fully matched serial online comparison would require $10N$ updates to mirror the 10 full batch iterations used in FloatSOM, so we proceeded with XPySOM as the executable external baseline. We now include this additional benchmark information in the manuscript.

For executable benchmarking, we therefore retain XPySOM as the direct external implementation baseline because it is the closest Python GPU batch-SOM comparator and because the XPySOM paper already benchmarks against earlier open-source SOM implementations and reports large speed advantages over those alternatives.

We now discuss Somoclu and GigaSOM more explicitly. Somoclu is an important CUDA/MPI SOM system, but it is not the closest baseline for the Python GPU batch-SOM workflow evaluated here. GigaSOM.jl is also important, especially for large cytometry workloads, but it is implemented in Julia and was not readily executable in our benchmark environment because the required Julia package and dependency conditions were not met in our setup. We therefore discuss GigaSOM as a distributed large-scale SOM system precedent rather than as a directly benchmarked external Python implementation.

### Manuscript Amendment

In Section 2.1, we revised the related-work discussion to:

> "Open-source SOM libraries range from lightweight to more performance-oriented implementations. MiniSom is a compact Python implementation of the classical online regime [@vettigliJustGlowingMinisom2018], whereas XPySOM is a Python-based batch SOM implementation designed for efficient GPU-backed execution [@manciniXPySomHighPerformanceSelfOrganizing2020]. aweSOM is a recent Python CPU/GPU SOM implementation with ensemble stacking that targets large single-node workloads [@haAweSOMCPUGPUaccelerated2025], but it follows a serial online SOM training regime rather than the batch SOM formulation evaluated here. At larger scales, Somoclu and GigaSOM provide mature parallel SOM systems for large workloads [@wittekSomocluEfficientParallel2017; @kratochvilGigaSOMjlHighperformanceClustering2020]."

We also added:

> "We use XPySOM as the direct executable external baseline because it is the closest Python GPU batch-SOM comparator and because the XPySOM study already benchmarks against earlier open-source SOM implementations. We also attempted to benchmark aweSOM on our standard speed workload ($10^7$ samples, 50 dimensions, and a $32 \times 32$ map). Using aweSOM's standard training configuration and only one online update step per sample ($N$ updates), the run consistently reached our 30-minute timeout (1800 s). A fully matched serial online comparison would require $10N$ updates to mirror the 10 full batch iterations used in FloatSOM, so we proceeded with XPySOM as the executable external baseline. Somoclu remains an important CUDA/MPI SOM system, but it is less directly aligned with the Python GPU batch-SOM deployment setting evaluated here. GigaSOM.jl is an important large-scale cytometry-oriented SOM system, but it is implemented in Julia and was not readily executable in our Python/CUDA/Ray benchmark environment because the required Julia package and dependency conditions were not met in our setup. We therefore discuss GigaSOM as a large-scale systems precedent rather than as a directly benchmarked Python baseline. An important systems gap remains: XPySOM is limited to a single GPU and requires the full dataset to fit in VRAM, while GigaSOM does not provide a drop-in distributed GPU training baseline within the broadly used Python workflow targeted by FloatSOM."

## 4. Numbers should be embedded in the paper

### Reviewer Comment

> Quality numbers are not in the paper. Per-dataset effect sizes exist only as external .tsv paths (Tables S4 to S11 are captions pointing at files). They cannot be read off the forest plots either.

> Put the numbers in the paper. Embed the per-dataset effect-size tables instead of external .tsv paths.

### Response

We agree. The submitted version included the tables as reproducibility artifacts but did not embed their numerical contents in the manuscript, which makes the paper harder to evaluate independently. We have replaced the path-only supplementary table captions with embedded tables for the XPySOM calibration summaries, topology p-value summary, deployment effect summaries, and topology runtime summary. The external `.tsv` files remain as machine-readable artifacts, but the manuscript now contains the numerical values needed to evaluate the corresponding figures.

### Manuscript Amendment

In the Supplementary Tables section, we replaced the path-only captions with embedded tables. For example, the revised captions now read:

> "Supplementary Table S7. Paired topology comparison p-values for hexagonal versus MST and hexagonal versus RNG across balanced QE, holdout QE, and train QE. Rows list metric/dataset entries, including the OVERALL row. The MST and RNG columns report p-values using the manuscript reporting convention. The embedded table is reproduced from `assets/tables/supp_table_topology_hex_vs_mst_rng_pvalues.tsv`."

For the deployment comparison, the revised caption now reads:

> "Supplementary Table S8. Figure 13 deployment comparison percent summary for tuned FloatSOM RNG versus default hexagonal XPySOM across $QE_B$, $QE_H$, and $QE_T$. Rows list per-dataset and `GLOBAL_OVERALL` entries with the plotted median percent change and 95% confidence interval. The embedded table is reproduced from `assets/tables/supp_table_figure_13_xpysom_rng_deployment_summary.tsv`."

## 5. Deployment comparison separates topology, tuning, and implementation

### Reviewer Comment

> Deployment comparison (Section 7, Figure 13) is a different claim. Tuned FloatSOM RNG against untuned hexagonal XPySOM. One number, three changes: topology, tuning, implementation. The 14.5, 9.1, 22.5 percent gains cannot be credited to topology. Sections 5.1, 5.3, and 5.4 already have the pieces to separate them.

> Separate topology from tuning and implementation in the deployment comparison, or tune both systems.

### Response

We agree that Fig. 13 should not be read as attributing the full gain to topology alone. Our intent was to show the integrated deployment comparison between a practical default XPySOM run and the recommended tuned FloatSOM RNG configuration. The submitted manuscript already separates several components: Section 5.1 calibrates FloatSOM against XPySOM under matched hexagonal settings, Section 5.3 compares topologies inside FloatSOM under a matched Optuna budget, and Section 5.4 evaluates tuning relative to an untuned reference. However, Section 7 did not state this decomposition clearly enough.

We revised Section 7 and the Fig. 13 caption so that the 14.5%, 9.1%, and 22.5% improvements are described as an integrated deployment effect, not a topology-only effect. We also point readers to the tuned-hexagonal and tuned-MST deployment figures in the Supplementary material so that the effect of topology can be read separately from the effect of tuning and implementation.

### Manuscript Amendment

In Section 7, we revised the opening paragraph to:

> "Fig. 13 is an integrated deployment comparison rather than a topology-only attribution. It compares the untuned default hexagonal XPySOM workflow against the recommended tuned FloatSOM RNG workflow, so the reported difference includes implementation, hyperparameter tuning, and topology choice [@manciniXPySomHighPerformanceSelfOrganizing2020]. The components are separated in the preceding analyses: Section 5.1 calibrates FloatSOM and XPySOM under matched hexagonal settings, Section 5.3 compares hexagonal, MST, and RNG inside FloatSOM under the same Optuna budget, and Section 5.4 evaluates tuned configurations against the untuned reference. Supplementary Figures S12-S13 provide the corresponding tuned hexagonal and tuned MST deployment comparisons against default hexagonal XPySOM."

We revised the Fig. 13 caption to:

> "Figure 13. Integrated deployment comparison of default hexagonal XPySOM versus tuned FloatSOM RNG. The comparison intentionally combines implementation, hyperparameter tuning, and topology choice and should not be interpreted as attributing the full difference to topology alone."

## 6. MST and RNG novelty claims

### Reviewer Comment

> Drop "novel" for the topologies. Jang et al., 2009, already in the references, put MSTs on SOMs. Frame this as the first version that scales on GPUs.

### Response

We agree that MST itself should not be described as a novel SOM topology because prior work, including Jang et al., used MSTs in SOMs. We revised the manuscript to remove broad novelty language for MSTs and instead frame the contribution as a scalable GPU implementation and large-scale evaluation of dynamic graph-based SOM topologies.

For RNG, we retain a narrower and qualified novelty claim. We have not identified prior work applying dynamically refreshed Relative Neighborhood Graphs as the neighborhood topology in SOM training. To avoid overclaiming, we now phrase this as "to our knowledge" and distinguish it from the non-novel MST component.

### Manuscript Amendment

In the Abstract, we replaced:

> "novel topologies beyond regular lattices"

with:

> "scalable graph-based topologies beyond regular lattices"

In Section 2.3, we added:

> "The SOM literature has also explored alternatives to fixed lattices, including dynamic maps and graph-structured neighborhoods [@vasighiDirectedBatchGrowing2017; @spanakisAMSOMAdaptiveMoving2016; @kangasVariantsSelforganizingMaps1990; @jangUseMinimalSpanning2009]. In particular, prior work has used MST-based neighborhoods in SOMs [@jangUseMinimalSpanning2009], so we do not claim MST itself as a novel SOM topology. Our MST contribution is a scalable GPU-compatible dynamic implementation and large-scale evaluation."

We also added:

> "Relative Neighborhood Graphs (RNGs) [@toussaintRelativeNeighbourhoodGraph1980] are of particular interest here. To our knowledge, dynamically refreshed RNG neighborhoods have not previously been used as a SOM training topology; we return to the full rationale and implementation for RNG in Section 3.2.2."

In the Discussion, we replaced broad "novel graph-based topology" wording with:

> "This manuscript presents FloatSOM as a unified large-scale SOM framework that combines scalable graph-based topology support with sampling options, optimised hyperparameters, and distributed out-of-memory GPU execution."

## 7. Runtime cost of RNG recommendation

### Reviewer Comment

> Measure the runtime cost of the recommended topology. Section 8.5 recommends RNG. At grid size 64, RNG is 27x hexagonal, MST is 8x. The MST distances-to-CPU step for Kruskal is the cheap part. The all-pairs hop-distance step both share plus RNG's blocker test is the expensive part. Put a number on the RNG cost.

### Response

We agree that any recommendation of RNG must be paired with its runtime cost. No additional end-to-end benchmarking is needed to answer this reviewer request because the manuscript already reports this cost in Fig. 12 and the topology-runtime discussion: at grid size 64, the 8-GPU mean runtime was 32.54 s for hexagonal, 266.45 s for MST, and 880.83 s for RNG, corresponding to 8.19x and 27.07x the hexagonal runtime for MST and RNG, respectively. We revised the recommendation/discussion text so the cost is visible at the point where RNG is recommended, and specifically frame the caveat as most important for very large grids.

### Manuscript Amendment

In Section 8.5, we added:

> "Overall, these results support a practical deployment strategy that uses RNG with topology-aware tuned defaults when $QE$ is the priority and topology-construction overhead is acceptable. That recommendation is conditional on grid size. In the grid-size scaling benchmark, the largest tested grid size (64) required 32.54 s for hexagonal, 266.45 s for MST, and 880.83 s for RNG on 8 GPUs, making MST 8.19x and RNG 27.07x slower than hexagonal at that point. For workloads dominated by very large grids, hexagonal remains the appropriate throughput-oriented default, and MST can be a practical compromise when graph-based topology is desired but RNG's blocker-test cost is too high."

## 8. Multiple-comparison correction

### Reviewer Comment

> State whether the paired t-tests were corrected for multiple comparisons. About 42 per topology, 14 datasets by 3 metrics. Seed counts and intervals are already there.

### Response

We agree that the statistical reporting should state whether the paired tests were multiplicity-corrected. On checking the analysis code, the publication-figure pipeline already computes Benjamini-Hochberg adjusted `q_value` fields for dataset-level summaries. We have revised the manuscript-facing topology p-value table to report both raw `p_value` and adjusted `q_value` entries, and we have updated the plotting code so significance markers use `q_value` when available, falling back to `p_value` only when no adjusted value exists.

### Manuscript Amendment

In Section 4.4, we added:

> "For dataset-level families of related paired tests, we compute Benjamini-Hochberg adjusted q-values in addition to raw paired $t$-test p-values. The adjustment is applied across non-global dataset rows within each comparison family. Global pooled rows are reported separately as overall summaries and are not included in the dataset-level adjustment family; these pooled rows therefore retain raw p-values only."

In the Fig. 6 and Fig. 7 result text, we added:

> "Dataset-level Benjamini-Hochberg adjusted q-values for the same Fig. 6 comparisons are reported in Supplementary Table S7; q<0.05 in 5/14 balanced QE rows, 7/14 holdout QE rows, and 5/14 train QE rows."

and:

> "Dataset-level Benjamini-Hochberg adjusted q-values for the same Fig. 7 comparisons are reported in Supplementary Table S7; q<0.05 in 8/14 balanced QE rows, 7/14 holdout QE rows, and 8/14 train QE rows."

In the Fig. 6 and Fig. 7 captions, we added:

> "Dataset-level raw p-values and Benjamini-Hochberg adjusted q-values are reported in Supplementary Table S7."

In the supplementary topology table caption, we revised:

> "Supplementary Table S7. Paired topology comparison p-values for hexagonal versus MST and hexagonal versus RNG across balanced QE, holdout QE, and train QE. Rows list metric/dataset entries, including the OVERALL row. The MST and RNG columns report raw p-values and Benjamini-Hochberg q-values for the corresponding dataset-level comparison family. OVERALL rows are pooled summaries and are shown separately from the dataset-level adjustment, so their q-values are reported as `NA`. The embedded table is reproduced from `assets/tables/supp_table_topology_hex_vs_mst_rng_pvalues.tsv`."

## 9. Dead-node and node-utilization reporting

### Reviewer Comment

> Add a dead-node or node-utilization count across the three topologies. Cheap, and it bears on the preservation question.

### Response

We agree. Node utilization is a useful diagnostic for determining whether lower $QE$ comes from broadly used map capacity or from uneven allocation in which some nodes are effectively unused. We added node-utilization and dead-node-fraction diagnostics to the matched topology benchmark outputs, using the same matched units as the topology comparison.

### Manuscript Amendment

In Section 4.1, we added:

> "We also report node utilization diagnostics for the same fitted maps. Node utilization is the fraction of SOM nodes selected as a best-matching unit by at least one sample in the evaluated split, and dead-node fraction is its complement. MTR, node utilization, and dead-node fraction are computed for both training and holdout splits and summarized with the same balanced train-holdout convention used for $QE$; these diagnostics are not optimized by Optuna."

In Section 5.3, we added:

> "Topology comparisons retain $QE$ as the primary optimized endpoint, with the Optuna hexagonal batch setting as the primary regular-topology baseline; Fig. 5 provides a qualitative illustration of the neighborhood structures produced by hexagonal, MST, and RNG. To test whether graph-topology $QE$ gains reflect useful topology behavior rather than only looser vector quantization, the same final maps are also evaluated post hoc for MTR, node utilization, and dead-node fraction under matched tuned and untuned/default profiles."

## 10. HDSSSOM framing

### Reviewer Comment

> Keep the HDSSSOM results framed as the pilot they already are.

### Response

We agree and preserve the current framing. The submitted manuscript already describes HDSSSOM as a focused screening pilot and not as part of the main full-versus-random sampling benchmark. We kept that language and added a sentence to avoid broadening the claim beyond the pilot configuration.

### Manuscript Amendment

The current Section 5.2 text already states:

> "We first report a focused HDSSSOM pilot as an elimination comparison rather than as part of the broader sampling benchmark. This pilot used a smaller, more limited configuration than the later full versus random analysis and is summarized in Supplementary Table S2."

We added:

> "These HDSSSOM results should therefore be interpreted as a pilot screen under the stated configuration rather than as a comprehensive evaluation of all possible HDSSSOM schedules."

## 11. Scaling efficiency and extrapolated 1-GPU baselines

### Reviewer Comment

> Some scaling points are extrapolated 1-GPU values, not measured (Section 4.2). Stated in the paper. Still limits the efficiency panels.

### Response

We agree. The submitted manuscript already states that some efficiency denominators are locally extrapolated from the last available 1-GPU point when direct 1-GPU runs were unavailable. We retained this limitation and added explicit wording that above-100% efficiency should not be interpreted as pure superlinear compute scaling.

### Manuscript Amendment

The current Section 6.2 text states:

> "When a direct 1-GPU baseline was unavailable at a given axis value, the efficiency denominator was constructed by local linear extrapolation from the last available 1-GPU point on that curve (Section 4.2), so some values should be interpreted with care if the underlying 1-GPU runtime is nonlinear over that range."

In Section 6.2.2, we added:

> "Efficiencies above 100% should also be interpreted as a combined consequence of parallelism and a changed memory/data-staging regime, not as evidence of superlinear compute scaling."

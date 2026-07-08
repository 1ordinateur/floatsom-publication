# Response Letter Draft: Reviewer 1

We thank the reviewer for the careful reading and for separating the systems contribution from the quality/topology claims. We agree that the systems evidence is stronger in the submitted version than the topology-quality evidence, and the revision is structured to make that distinction explicit. Where the manuscript already contained the relevant analysis but did not state the point clearly enough, we have clarified the text. Where the reviewer identified a missing control or missing reporting detail, we have added the corresponding amendment below.

## 1. Quality and topology diagnostics

### Reviewer Comment

> The quality claim runs on QE alone. QE measures distance to the best-matching unit, not neighborhood preservation, which is the point of a SOM. No topographic error or trustworthiness, even though Forest et al., 2020 is cited and implements them.

### Response

The submitted manuscript used $QE$ as the principal deployment metric because the benchmark was designed around large-scale vector quantization and because $QE$ is the metric used in our XPySOM calibration. We have expanded the topology comparison by adding a local BMU-neighborhood ordering diagnostic alongside $QE$.

Raw topographic error is not the primary cross-topology statistic because it defines an error by whether the first and second best-matching units are immediate neighbors on the map. That adjacency relation is itself topology-dependent. Even when map size and output dimensionality are fixed, hexagonal, MST, and RNG maps have different connectivity and degree structure, so raw topographic error would partly measure the graph's one-hop neighbor convention.

This is not only a concern raised by Ramos et al. [@ramosROLELATTICEDIMENSIONALITY2018]. Neme and Miramontes further showed that topographic error is affected by statistical properties of the neuron lattice, including path length, clustering, and connectivity length [@nemeStatisticalPropertiesLattices2005]. Other SOM comparisons, including Machon-Gonzalez and Lopez-Garcia, also caution that topographic-error comparisons require the same map size because the errors depend on map design [@machon-gonzalezFLSOMIndividualKernel2010]. These results support the narrower point relevant here: holding map size fixed is necessary, but holding map size or output dimensionality fixed is not sufficient when the adjacency graph itself changes.

Instead, we added Mean Tied Rank (MTR), following the tied-rank logic proposed by Ramos et al. for comparing SOMs with different topologies [@ramosROLELATTICEDIMENSIONALITY2018]. For each sample, we compute the first and second BMUs, rank all non-winning units by graph shortest-path distance from the first BMU, assign average ranks to tied graph-distance groups, and record the tied rank of the second BMU. Lower MTR indicates that the second-best prototype is topologically close to the winning prototype. Because all topology comparisons use the same map size, no map-size normalization is required. We report MTR next to $QE$ in the cross-topology local BMU-neighborhood ordering diagnostics.

We also added matched post hoc topology diagnostics for the final trained FloatSOM maps. These diagnostics evaluate the same dataset, seed, sampling, and topology units, and report both MTR and node-use summaries.

We distinguish this matched fixed diagnostic benchmark from the existing Optuna-budget topology figure. Fig. 7 remains the tuned topology $QE$ comparison, asking whether tuned RNG can achieve lower $QE$ than tuned hexagonal maps under the same optimization protocol. The matched fixed diagnostic benchmark reruns deployable fixed tuned and untuned profiles under matched dataset, seed, topology, and split keys, then evaluates the resulting final maps for MTR and node use.

The matched diagnostics show that the graph-topology improvement is not strictly a tuning artifact. Under the untuned profile, both MST and RNG improved balanced $QE$ relative to hexagonal maps, but RNG gave the clearer local BMU-neighborhood ordering result: RNG improved balanced MTR by 2.44 tied-rank positions relative to hexagonal maps (95% CI 2.20 to 2.68; p=5.16e-57; 269/11 matched pairs favoring RNG), whereas MST did not clearly improve MTR relative to hexagonal maps. Under the tuned profile, RNG improved balanced $QE$ relative to tuned hexagonal maps by 0.0515 (95% CI 0.0318 to 0.0712; p=4.94e-07) and improved balanced MTR by 25.67 tied-rank positions (95% CI 24.77 to 26.56; p=2.17e-154; 280/0 matched pairs favoring RNG).

We also added the tuned-versus-untuned MTR comparison because it directly addresses whether $QE$ optimization itself changes topology preservation. Tuning improved balanced $QE$ for hexagonal, MST, and RNG, but worsened balanced MTR in each topology. The MTR penalty was much larger for hexagonal (24.34 tied-rank positions) than for MST (1.43) or RNG (1.11). We interpret this as evidence that the fixed hexagonal lattice reaches lower optimized $QE$ at a much larger cost to local BMU-neighborhood ordering than the graph topologies do.

Node utilization results support this interpretation, as the reviewer kindly pointed out. Considering this, we calculated node utilization and dead-node fraction for the same matched maps. For example, tuned RNG increased balanced node utilization by 0.0197 relative to tuned hexagonal maps (95% CI 0.0158 to 0.0235; p=1.38e-20) and reduced balanced dead-node fraction by the same amount. These node-use diagnostics indicate that the MTR result is not accompanied by reduced map utilization. We embedded the compact diagnostic summary in the main text and the full paired diagnostic and by-dataset diagnostic tables in the manuscript supplement rather than leaving them as external TSV-only outputs.

### Manuscript Amendment

In Section 4.1, we added:

> "For cross-topology local BMU-neighborhood ordering diagnostics, we report Mean Tied Rank (MTR), following the tied-rank approach proposed for comparing SOMs with different topologies [@ramosROLELATTICEDIMENSIONALITY2018], rather than raw topographic error. For each sample $x_i$, let $b_i^{(1)}$ and $b_i^{(2)}$ denote the first and second best-matching units. We rank all non-winning units by graph shortest-path distance from $b_i^{(1)}$, assigning the average ordinal rank to units tied at the same graph-distance shell. If $b_i^{(2)}$ lies in shell $S_d$ and $L_d$ non-winning units are in closer shells, its tied rank is $\tau_i=L_d+(|S_d|+1)/2$, and $MTR=N^{-1}\sum_i \tau_i$. MTR is therefore reported in tied-rank positions over SOM nodes rather than feature-space units or graph-edge counts. Lower MTR indicates that the second-best prototype remains topologically close to the winning prototype. Raw topographic error is not used as the cross-topology statistic because its one-hop adjacency criterion changes with the evaluated graph's connectivity and degree structure; prior work has shown that topographic error depends on map topology, lattice statistical properties, and map design choices such as size [@ramosROLELATTICEDIMENSIONALITY2018; @nemeStatisticalPropertiesLattices2005; @machon-gonzalezFLSOMIndividualKernel2010]."

In Section 4.3, we clarified the fixed tuned rerun design:

> "To be clear, these fixed-configuration reruns are distinct from the top-$k$ Optuna summaries used for search-budget comparisons. The Optuna top-$k$ summaries represent the maximal observed performance of each algorithmic setting within the tuning campaign, whereas the fixed tuned reruns represent likely general usage. The fixed tuned configurations use the average numeric hyperparameters and modal categorical hyperparameters selected from the tuned runs, then rerun that single deployable configuration under matched dataset, seed, topology, and split keys."

In Section 5.3, we revised the opening sentence to:

> "Topology comparisons retain $QE$ as the primary optimized endpoint, with the Optuna hexagonal batch setting as the primary regular-topology baseline; Fig. 5 provides a qualitative illustration of the neighborhood structures produced by hexagonal, MST, and RNG. Separately, fixed tuned and untuned deployable reruns are evaluated post hoc for MTR and node use. Node utilization and dead-node fraction are reported as confirmatory capacity-use checks under matched tuned and untuned profiles."

We also added the tuned and untuned RNG diagnostic result:

> "Fig. 7 and the matched fixed diagnostic benchmark address complementary questions. Fig. 7 is the Optuna-budget topology comparison and asks whether tuned RNG can achieve lower $QE$ than tuned hexagonal maps under the same optimization protocol. The matched fixed diagnostic benchmark reruns deployable fixed tuned and untuned profiles under matched dataset, seed, topology, and split keys, then evaluates the resulting final maps for MTR and node use."

> "The matched diagnostic benchmark indicates that the graph-topology advantage is not strictly a tuning artifact (Table \ref{tab:matched_topology_diagnostics_summary}). Under the untuned profile, both MST and RNG improved balanced $QE$ relative to hexagonal maps, but the MTR result distinguished the two graph topologies: MST did not clearly improve balanced MTR relative to hexagonal maps, whereas RNG lowered balanced MTR by 2.44 tied-rank positions. Under the tuned profile, both MST and RNG improved balanced $QE$ relative to hexagonal maps. RNG was statistically comparable with MST for balanced $QE$, while lowering balanced MTR by 2.96 tied-rank positions relative to MST. Thus, MST remains a strong $QE$ topology, whereas RNG has the strongest paired $QE$/MTR profile."

In Section 5.4, we added:

> "The matched diagnostics show that this $QE$ gain has different local BMU-neighborhood ordering costs across topology families. Tuning improved balanced $QE$ for hexagonal, MST, and RNG, but worsened balanced MTR in each case. The MTR increase was much larger for hexagonal (24.34 tied-rank positions) than for MST (1.43) or RNG (1.11), indicating that the fixed hexagonal lattice pays a substantially larger local BMU-neighborhood ordering cost to achieve lower $QE$. Node-use diagnostics indicate that the MTR result is not accompanied by reduced map utilization."

In the Discussion, we added:

> "$QE$ and MTR summarize complementary aspects of the fitted maps: $QE$ reports vector-quantization fidelity, while MTR reports the graph-rank separation between first and second BMUs. We therefore interpret the topology comparisons using both quantities."

> "In the matched diagnostics, RNG showed lower MTR than hexagonal under both untuned and tuned profiles."

> "The tuned-versus-untuned comparisons also suggest a topology-specific trade-off between the $QE$ objective and MTR. Tuning improved balanced $QE$ for hexagonal, MST, and RNG, but the MTR penalty was much larger for hexagonal: tuned hexagonal maps worsened balanced MTR by 24.34 tied-rank positions relative to untuned hexagonal maps, whereas tuned MST and tuned RNG worsened balanced MTR by 1.43 and 1.11 tied-rank positions, respectively. This pattern suggests that the fixed hexagonal lattice reaches lower optimized $QE$ at a much larger cost to the local BMU-neighborhood ordering measured by MTR, while the graph topologies absorb the same $QE$-oriented tuning with a much smaller MTR cost."

## 2. Hexagonal neighborhood-radius control

### Reviewer Comment

> Add a control that shrinks the hexagonal neighborhood radius. If a tighter-radius hexagonal map closes most of the QE gap, the graph topologies are winning on looseness, not structure.

### Response

We agree that neighborhood radius is an important potential confound. However, the submitted topology comparison was not a tuned MST/RNG versus default-radius hexagonal comparison. The Optuna campaign optimized `initial_radius` for every topology family, including hexagonal, using the same search interval and the same tuning budget. Therefore, the hexagonal comparator was already free to adopt a tighter neighborhood radius if that improved quantization error.

To make this clear, we revised the Optuna benchmark protocol and topology-results sections. We now state explicitly that `initial_radius` was included in the Optuna search space for hexagonal, MST, and RNG runs. We also report the distilled selected radii, which show that tuning selected substantially tighter neighborhood settings than the untuned radius of 5.0 for all topology families. The tuned hexagonal comparator used the tightest radius among the topology families: 1.03 for hexagonal under full sampling, compared with 1.46 for MST and 1.41 for RNG; and 1.17 for hexagonal under random sampling, compared with 1.82 for MST and 1.77 for RNG.

This means the control proposed by the reviewer is already present in the Optuna design: the hexagonal map is allowed to shrink its neighborhood radius, and it in fact does so. The revised MTR diagnostics show that this tighter optimized setting still leaves a topology-specific MTR penalty. Tuning tightens the effective neighborhood settings for hexagonal, MST, and RNG, but the MTR cost is disproportionate for the fixed hexagonal lattice: the tuned-versus-untuned diagnostics show a much larger MTR penalty for hexagonal than for MST or RNG, whereas the graph topologies absorb the same QE-oriented tuning with a smaller local BMU-neighborhood ordering cost. This rules out comparison against an untuned broad-radius hexagonal baseline, although it does not isolate radius from the other tuned hyperparameters.

This addresses the radius-control concern: the observed MST/RNG $QE$ gains are not obtained by comparing graph topologies against an untuned or artificially broad hexagonal neighborhood radius. Instead, the results are best-observed-versus-best-observed comparisons under a matched Optuna budget.

### Manuscript Amendment

In Section 4.1, we added:

> "The neighborhood-radius search space was shared across topology families. In particular, `initial_radius` was an Optuna-optimized parameter for hexagonal, MST, and RNG runs, with the same search interval of 0.5 to 10.0 in each case. Thus, the hexagonal topology comparisons below use a tuned hexagonal comparator rather than a default-radius hexagonal baseline."

In Section 5.3, we added:

> "Because `initial_radius` was tuned for every topology family, the topology comparison is a best-observed-versus-best-observed comparison under the same Optuna budget rather than a comparison against an untuned hexagonal radius. The distilled deployable configurations selected tighter hexagonal radii than the graph topologies: under full sampling the selected `initial_radius` values were 1.03 for hexagonal, 1.46 for MST, and 1.41 for RNG, while under random sampling they were 1.17, 1.82, and 1.77, respectively. These values show that the reported MST/RNG $QE$ gains are not explained by evaluating hexagonal only at a broader default neighborhood radius, although they do not isolate radius from the other tuned hyperparameters."

## 3. Related work and external baselines

### Reviewer Comment

> Related work names prior graph and GPU SOMs but does not benchmark against any. aweSOM (Ha et al., JOSS 2025), a GPU Python SOM in the same N > 10^6 range, is not mentioned.

> Add recent GPU and distributed SOM baselines. Somoclu CUDA, GigaSOM, aweSOM. aweSOM is single-node CPU/GPU with ensemble stacking, so it is a quality baseline, not a distributed competitor.

### Response

We agree that the related-work coverage should be expanded and that aweSOM should be discussed. We now introduce aweSOM in the Introduction and Section 2.1 as, to our knowledge, the strongest contemporary open-source serial-online SOM implementation. This makes clear why aweSOM is a relevant serial-online comparator while also distinguishing it from FloatSOM, which is a batch SOM algorithm.

We attempted to benchmark aweSOM on our standard speed workload ($10^7$ samples, 50 dimensions, and a $32 \times 32$ map). Using aweSOM's standard training configuration and only one online update step per sample ($N$ updates), the run consistently ($n = 5$) reached our 30-minute timeout ($1800 s$), with all attempts timing out before 60% of the requested online updates had completed. A fully matched serial-online comparison would require $10N$ updates to mirror the 10 full batch iterations used in FloatSOM. As aweSOM's serial-online algorithm scales linearly with additional updates, this would require approximately 10 times as many pointwise updates as a setting that already timed out. Therefore, we proceeded with XPySOM as the external baseline. We now include this additional aweSOM benchmark information in the manuscript.

Additionally, we now discuss Somoclu and GigaSOM more explicitly. Somoclu is an important CUDA/MPI SOM system, but it is not the closest baseline for the Python GPU batch-SOM workflow evaluated here. XPySOM was directly benchmarked against Somoclu in its original evaluation and reported substantial speedups in that benchmark setting. Consequently, we do not believe that additional benchmarking against Somoclu is necessary. We have added this clarification into the body of the manuscript and thank the reviewer for their insights into this.

We also added a short note on GigaSOM.jl. Its flow cytometry example trained a $32 \times 32$ SOM on 1,167,129,317 cells as part of a full Julia workflow completed in under 25 minutes on an 11-node, 256-core CPU cluster [@kratochvilGigaSOMjlHighperformanceClustering2020]. Because that result differs in language, hardware, feature dimensionality, epoch count, and included workflow stages, we treat it as related systems context rather than a directly benchmarked Python/CUDA/Ray baseline.

### Manuscript Amendment

In the Introduction, we added:

> "Recent tools have improved single-node SOM execution. In particular, aweSOM is, to our knowledge, the strongest contemporary open-source serial-online SOM implementation, combining CPU/GPU acceleration with ensemble stacking for large single-node datasets [@haAweSOMCPUGPUaccelerated2025]. However, serial-online training remains algorithmically different from batch SOM training and requires pointwise updates, so it does not address the distributed batch and out-of-core training setting targeted here."

In Section 2.1, we revised the related-work discussion to:

> "Open-source SOM libraries range from lightweight implementations to systems-oriented packages. One family follows the classical serial-online regime, where the map is updated immediately after each sampled point. MiniSom is a compact Python implementation of this regime [@vettigliJustGlowingMinisom2018]. aweSOM is, to our knowledge, the strongest contemporary open-source serial-online SOM implementation: it combines CPU/GPU acceleration with statistically combined ensemble stacking, targets large single-node workloads, and reports good performance up to approximately $10^8$ points [@haAweSOMCPUGPUaccelerated2025]. Because aweSOM is serial online, its training cost scales with the number of pointwise updates rather than with batch iterations over aggregated assignments."

> "A second family targets batch or parallel SOM execution. XPySOM is a Python-based batch SOM implementation designed for efficient GPU-backed execution [@manciniXPySomHighPerformanceSelfOrganizing2020]. Its original evaluation directly compared against MiniSom, Somoclu, and TensorFlow SOM, and reported substantial speedups over the strongest open-source multicore and GPU-accelerated comparators in that benchmark setting [@manciniXPySomHighPerformanceSelfOrganizing2020]."

We also added:

> "For executable external benchmarking, this training-regime distinction determines which systems are directly comparable. We use XPySOM as the direct executable external baseline because it is the closest Python GPU batch-SOM comparator and because the XPySOM study already benchmarks against earlier open-source SOM implementations, including Somoclu. We also attempted to benchmark aweSOM because it is the strongest serial-online comparator we are aware of. On our standard speed workload ($10^7$ samples, 50 dimensions, and a $32 \times 32$ map), using aweSOM's standard training configuration and only one online update step per sample ($N$ updates), the run consistently reached our 30-minute timeout across five attempts ($n=5$; 1800 s each), with all attempts timing out before 60% of the requested online updates had completed. A fully matched serial-online comparison would require $10N$ updates to mirror the 10 full batch iterations used in FloatSOM. Because serial-online training scales with the number of pointwise updates, this would require approximately 10 times as many updates as a setting that already timed out, so we proceeded with XPySOM as the executable external baseline."

> "GigaSOM.jl reports a large-scale Julia workflow that trained a $32 \times 32$ SOM on 1,167,129,317 cells as part of a full analysis completed in under 25 minutes on an 11-node, 256-core CPU cluster [@kratochvilGigaSOMjlHighperformanceClustering2020]. Because that result differs in language, hardware, feature dimensionality, epoch count, and included workflow stages, we treat it as related systems context rather than a directly benchmarked Python/CUDA/Ray baseline."

## 4. Numbers should be embedded in the paper

### Reviewer Comment

> Quality numbers are not in the paper. Per-dataset effect sizes exist only as external table paths (Tables S4 to S11 are captions pointing at files). They cannot be read off the forest plots either.

> Put the numbers in the paper. Embed the per-dataset effect-size tables instead of external paths.

### Response

We agree. The submitted version included the tables as reproducibility artifacts but did not embed their numerical contents in the manuscript, which makes the paper harder to evaluate independently. We have replaced the path-only supplementary table captions with embedded tables for the XPySOM calibration summaries, topology q-value summary, deployment effect summaries, and topology runtime summary. The manuscript now contains the numerical values needed to evaluate the corresponding figures.

### Manuscript Amendment

In the Supplementary Tables section, we replaced the path-only captions with embedded tables. For example, the revised captions now read:

> "Supplementary Table S7. Paired topology comparison q-values for hexagonal versus MST and hexagonal versus RNG across balanced QE, holdout QE, and train QE. Rows list metric/dataset entries, including the OVERALL row. The MST_q and RNG_q columns report Benjamini-Hochberg q-values for the corresponding dataset-level comparison family; raw p-values are retained in the MST_p and RNG_p columns for audit. OVERALL rows are pooled summaries and are shown separately from the dataset-level adjustment, so their q-values are reported as `NA`."

For the deployment comparison, the revised caption now reads:

> "Supplementary Table S8. Figure 13 deployment comparison percent summary for tuned FloatSOM RNG versus untuned hexagonal XPySOM across $QE_B$, $QE_H$, and $QE_T$. Rows list per-dataset and `GLOBAL_OVERALL` entries with the plotted median percent change and 95% confidence interval."

## 5. Deployment comparison separates topology, tuning, and implementation

### Reviewer Comment

> Deployment comparison (Section 7, Figure 13) is a different claim. Tuned FloatSOM RNG against untuned hexagonal XPySOM. One number, three changes: topology, tuning, implementation. The 14.5, 9.1, 22.5 percent gains cannot be credited to topology. Sections 5.1, 5.3, and 5.4 already have the pieces to separate them.

> Separate topology from tuning and implementation in the deployment comparison, or tune both systems.

### Response

We agree that Fig. 13 should not be read as attributing the full gain to topology alone. Our intent was to show the integrated deployment comparison between a practical untuned XPySOM run and the recommended tuned FloatSOM RNG configuration. The submitted manuscript already separates several components: Section 5.1 calibrates FloatSOM against XPySOM under matched hexagonal settings, Section 5.3 compares topologies inside FloatSOM under a matched Optuna budget, and Section 5.4 evaluates tuning relative to an untuned reference. However, Section 7 did not state this decomposition clearly enough.

We revised Section 7 and the Fig. 13 caption so that the 14.5%, 9.1%, and 22.5% improvements are described as an integrated deployment effect, not a topology-only effect. We also point readers to the tuned-hexagonal and tuned-MST deployment figures in the Supplementary material so that the effect of topology can be read separately from the effect of tuning and implementation.

### Manuscript Amendment

In Section 7, we revised the opening paragraph to:

> "Fig. 13 is an integrated deployment comparison rather than a topology-only attribution. It compares the untuned hexagonal XPySOM workflow against the recommended tuned FloatSOM RNG workflow, so the reported difference includes implementation, hyperparameter tuning, and topology choice [@manciniXPySomHighPerformanceSelfOrganizing2020]. The components are separated in the preceding analyses: Section 5.1 calibrates FloatSOM and XPySOM under matched hexagonal settings, Section 5.3 compares hexagonal, MST, and RNG inside FloatSOM under the same Optuna budget, and Section 5.4 evaluates tuned configurations against the untuned reference. Supplementary Figures S12-S13 provide the corresponding tuned hexagonal and tuned MST deployment comparisons against untuned hexagonal XPySOM."

We revised the Fig. 13 caption to:

> "Figure 13. Integrated deployment comparison of untuned hexagonal XPySOM versus tuned FloatSOM RNG. The comparison intentionally combines implementation, hyperparameter tuning, and topology choice and should not be interpreted as attributing the full difference to topology alone."

## 6. MST and RNG novelty claims

### Reviewer Comment

> Drop "novel" for the topologies. Jang et al., 2009, already in the references, put MSTs on SOMs. Frame this as the first version that scales on GPUs.

### Response

We agree that the novelty claim needed to distinguish MST-on-SOM analyses from MST-as-training-topology SOMs. Jang et al. used MSTs on SOMs, but that use is not equivalent to the FloatSOM topology path: in our implementation, the MST is the operative neighborhood graph during training, is recalculated from the evolving node-weight geometry, and directly changes the SOM update influence matrix. This differs from superimposing or embedding an MST on an already trained map for interpretation, map-shape assessment, or related post hoc analysis.

We also avoid overclaiming. Earlier SOM variants discussed MST-defined neighborhoods during learning, so we do not claim that the graph object alone is new. The contribution is the scalable GPU-compatible refreshed MST training topology and its large-scale quantitative evaluation. For RNG, we retain a narrower and qualified novelty claim: we have not identified prior work applying dynamically refreshed Relative Neighborhood Graphs as the neighborhood topology in SOM training.

### Manuscript Amendment

In the Abstract, we replaced:

> "novel topologies beyond regular lattices"

with:

> "scalable training-time graph topologies beyond regular lattices"

In Section 2.3, we added:

> "The SOM literature has also explored alternatives to fixed lattices, including dynamic maps and graph-structured neighborhoods [@vasighiDirectedBatchGrowing2017; @spanakisAMSOMAdaptiveMoving2016; @kangasVariantsSelforganizingMaps1990; @jangUseMinimalSpanning2009]. MSTs have previously appeared in SOM analyses, including as structures superimposed on an already trained map for interpretation, subnode embedding, or map-shape assessment [@jangUseMinimalSpanning2009]. This is distinct from the training-time role used here: in FloatSOM, the MST is the operative neighborhood graph during SOM updates, is recalculated from the evolving node-weight geometry, and directly changes the update influence matrix used during learning. Earlier SOM variants also discussed MST-defined neighborhoods during learning [@kangasVariantsSelforganizingMaps1990], but these alternatives have not been assessed on large-scale datasets and do not have implementations that are either publicly available or suitable for distributed GPU computation. FloatSOM's MST contribution is therefore a scalable GPU-compatible implementation of refreshed MST-based SOM training and a large-scale quantification of its effect, rather than a post hoc MST overlay on a conventional trained SOM."

We also added:

> "Relative Neighborhood Graphs (RNGs) [@toussaintRelativeNeighbourhoodGraph1980] are of particular interest here. To our knowledge, dynamically refreshed RNG neighborhoods have not previously been used as a SOM training topology; we return to the full rationale and implementation for RNG in Section 3.2.2."

In Section 3.2.1, we clarified that the MST topology changes training rather than only visualization:

> "The MST topology replaces fixed lattice neighborhood distance with graph hop distance on a minimum spanning tree built from current SOM nodes. The MST is therefore part of the SOM training rule: it determines which nodes receive neighborhood influence during each topology-refresh interval, rather than serving as a visualization or analysis layer after training."

In the Discussion, we replaced broad "novel graph-based topology" wording with:

> "Because MST and RNG are used as training-time neighborhoods, these differences reflect changes to the SOM update dynamics rather than post hoc graph summaries placed over an unchanged trained map."

and:

> "This manuscript presents FloatSOM as a unified large-scale SOM framework that combines scalable training-time graph topology support with sampling options, optimised hyperparameters, and distributed out-of-memory GPU execution."

## 7. Runtime cost of RNG recommendation

### Reviewer Comment

> Measure the runtime cost of the recommended topology. Section 8.5 recommends RNG. At grid size 64, RNG is 27x hexagonal, MST is 8x. The MST distances-to-CPU step for Kruskal is the cheap part. The all-pairs hop-distance step both share plus RNG's blocker test is the expensive part. Put a number on the RNG cost.

### Response

We agree that any recommendation of RNG must be paired with its runtime cost. No additional end-to-end benchmarking is needed to answer this reviewer request because the manuscript already reports this cost in Fig. 12 and the topology-runtime discussion: at grid size 64, the 8-GPU mean runtime was 32.54 s for hexagonal, 266.45 s for MST, and 880.83 s for RNG, corresponding to 8.19x and 27.07x the hexagonal runtime for MST and RNG, respectively. We revised the recommendation/discussion text so the cost is visible at the point where RNG is recommended, and specifically frame the caveat as most important for very large grids.

### Manuscript Amendment

In Section 8.5, we added:

> "Overall, these results support a practical deployment strategy that uses RNG with topology-aware tuned configurations when $QE$ is the priority and topology-construction overhead is acceptable. That recommendation is conditional on grid size. In the grid-size scaling benchmark, the largest tested grid size (64) required 32.54 s for hexagonal, 266.45 s for MST, and 880.83 s for RNG on 8 GPUs, making MST 8.19x and RNG 27.07x slower than hexagonal at that point. For workloads dominated by very large grids, hexagonal remains the appropriate throughput-oriented default, and MST can be a practical compromise when graph-based topology is desired but RNG's blocker-test cost is too high."

## 8. Multiple-comparison correction

### Reviewer Comment

> State whether the paired t-tests were corrected for multiple comparisons. About 42 per topology, 14 datasets by 3 metrics. Seed counts and intervals are already there.

### Response

We agree that the statistical reporting should state whether the paired tests were multiplicity-corrected. On checking the analysis code, the publication-figure pipeline already computes Benjamini-Hochberg adjusted `q_value` fields for dataset-level summaries. We have revised the manuscript-facing topology q-value table to report adjusted `q_value` entries alongside raw `p_value` entries retained for audit, and all affected forest-plot significance legends now use q-value notation rather than p-value notation.

We also audited the figure-level significance annotations against the corrected q-value convention. The correction did not change the substantive interpretation of the results: most affected annotations only changed star level while remaining significant. Only two dataset-level points crossed the significance threshold after correction, both in Fig. 7B for the holdout-QE hexagonal versus RNG comparison: `blobs` changed from p=0.0358 (`*`) to q=0.0626 (`ns`), and `iris` changed from p=0.0444 (`*`) to q=0.0691 (`ns`). We have corrected these figure annotations accordingly.

### Manuscript Amendment

In Section 4.4, we added:

> "For dataset-level families of related paired tests, we compute Benjamini-Hochberg adjusted q-values in addition to raw paired $t$-test p-values. Dataset-level figure significance markers and dataset-level significance counts use these adjusted q-values. The adjustment is applied across non-global dataset rows within each comparison family. Global pooled rows are reported separately as overall summaries and are not included in the dataset-level adjustment family; these pooled rows therefore retain raw p-values only."

In the Fig. 4 caption, we added:

> "Figure significance markers use Benjamini-Hochberg adjusted q-values."

In the Fig. 6 and Fig. 7 result text, we added:

> "Dataset-level Benjamini-Hochberg adjusted q-values for the same Fig. 6 comparisons are reported in Supplementary Table S7; q<0.05 in 5/14 balanced QE rows, 7/14 holdout QE rows, and 5/14 train QE rows."

and:

> "Dataset-level Benjamini-Hochberg adjusted q-values for the same Fig. 7 comparisons are reported in Supplementary Table S7; q<0.05 in 8/14 balanced QE rows, 7/14 holdout QE rows, and 8/14 train QE rows."

In the Fig. 6 and Fig. 7 captions, we added:

> "Figure significance markers use dataset-level Benjamini-Hochberg adjusted q-values, which are reported in Supplementary Table S7 alongside raw p-values retained for audit."

In the supplementary topology table caption, we revised:

> "Supplementary Table S7. Paired topology comparison q-values for hexagonal versus MST and hexagonal versus RNG across balanced QE, holdout QE, and train QE. Rows list metric/dataset entries, including the OVERALL row. The MST_q and RNG_q columns report Benjamini-Hochberg q-values for the corresponding dataset-level comparison family; raw p-values are retained in the MST_p and RNG_p columns for audit. OVERALL rows are pooled summaries and are shown separately from the dataset-level adjustment, so their q-values are reported as `NA`."

## 9. Dead-node and node-utilization reporting

### Reviewer Comment

> Add a dead-node or node-utilization count across the three topologies. Cheap, and it bears on the preservation question.

### Response

We agree, and thank the reviewer for pointing this out. Node utilization is a useful confirmatory diagnostic for checking whether the MTR result is accompanied by broadly used map capacity rather than uneven allocation in which some nodes are effectively unused. Considering this, we added node-utilization and dead-node-fraction diagnostics to the matched topology benchmark outputs, using the same matched units as the topology comparison.

The matched tuned-profile result favored RNG rather than indicating poorer map use. Relative to tuned hexagonal maps, tuned RNG increased balanced node utilization by 0.0197 (95% CI 0.0158 to 0.0235; p=1.38e-20) and reduced balanced dead-node fraction by the same amount. We report the split-specific and balanced diagnostics separately and present node utilization as a confirmatory diagnostic alongside MTR.

### Manuscript Amendment

In Section 4.1, we added:

> "We also report node utilization diagnostics for the same fitted maps. Node utilization is the fraction of SOM nodes selected as a best-matching unit by at least one sample in the evaluated split, and dead-node fraction is its complement. MTR, node utilization, and dead-node fraction are computed for both training and holdout splits and summarized with the same balanced train-holdout convention used for $QE$; these diagnostics are not optimized by Optuna."

In Section 5.3, we added:

> "Topology comparisons retain $QE$ as the primary optimized endpoint, with the Optuna hexagonal batch setting as the primary regular-topology baseline; Fig. 5 provides a qualitative illustration of the neighborhood structures produced by hexagonal, MST, and RNG. Separately, fixed tuned and untuned deployable reruns are evaluated post hoc for MTR and node use. Node utilization and dead-node fraction are reported as confirmatory capacity-use checks under matched tuned and untuned profiles."

We also embedded the paired diagnostic table:

> "Supplementary Table S12. Matched topology diagnostic paired summaries across tuned and untuned profiles. Rows report pooled paired effects from the final matched random-seed topology diagnostic benchmark. `Effect favoring comparator` is oriented so positive values favor the comparator after applying each metric's directionality; lower is better for QE, MTR, and dead-node fraction, while higher is better for node utilization. Because these are pre-specified pooled paired summaries rather than dataset-level test families, the table reports raw paired-test p-values."

> "Supplementary Table S13. Full matched topology diagnostic means by profile, dataset, and topology. Rows report the mean value across matched seeds for each profile, dataset, topology, and full-sampling setting."

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

# Response Letter Draft: Reviewer 2

We thank the reviewer for the careful and constructive assessment. We agree with the central framing: FloatSOM is primarily a systems and tooling contribution, and the manuscript must keep the scope of its claims aligned with the evidence shown in the main text. In revision, we have focused on making the topology, scaling, and statistical claims more explicit, and on separating evidence that was directly demonstrated from evidence that should be treated as supporting context or a limitation.

## 1. Direct MST-versus-RNG evidence in the main text

### Reviewer Comment

> The paper's principal empirical conclusion is that RNG is the strongest topology, yet the main text never directly establishes that ordering. Section 5.3 compares MST and RNG independently against the hexagonal baseline, but the direct MST-versus-RNG comparison appears only in the supplementary material (Figs. S4-S7). Because the Discussion ultimately recommends RNG over the other topology options, this comparison is central to the paper's argument and should be presented in the main text rather than relegated to the appendix.

### Response

We agree that the direct MST-versus-RNG comparison is necessary to support our topology recommendation, and we have promoted it from the supplement to the main topology results as Fig. 8. In the matched full-sampling Optuna top-$k$ analysis, RNG achieves significantly lower balanced and train $QE$ than MST, while no significant holdout-$QE$ difference is detected. The hyperparameter analysis also gives RNG the lowest overall stability score across seeds and datasets.

Taken together, these results support RNG as our quality-first default: it provides the strongest attainable $QE$ performance and the most reproducible tuned settings. MST remains a lower-cost alternative for runtime-constrained workloads, particularly at large grid sizes. We therefore do not claim that RNG dominates every endpoint or every deployment regime.

We also revised the geometric explanation so that the RNG preference is not reduced to graph density alone. MST minimizes neighborhood coupling and permits substantial redistribution along irregular structures, but a tree cannot represent multiple locally appropriate connections in a dense region. RNG permits such connections only when supported by the evolving prototype geometry, whereas the hexagonal lattice imposes a uniform neighbor pattern before learning. The observed $QE$ results are presented as consistent with this account rather than as identifying a unique causal mechanism.

### Manuscript Amendment

In Section 5.3, we added Fig. 8 and a direct MST-versus-RNG paragraph after the separate hexagonal-versus-MST and hexagonal-versus-RNG comparisons:

> "Fig. 8 directly compares MST and RNG under full sampling. In the matched Optuna top-$k$ analysis, RNG achieves significantly lower balanced and train $QE$ than MST, while no significant holdout-$QE$ difference is detected. Across the topology comparisons, RNG therefore provides the strongest attainable $QE$ performance among the evaluated topologies."

## 2. Section 6.2 versus Section 6.3 scaling language

### Reviewer Comment

> Section 6.2 claims "broadly consistent scaling... regardless of topology," citing Figs. 11 and S11, but Fig. 11 itself shows RNG only -- the hexagonal/MST curves substantiating this sit in the supplement. More importantly, "regardless of topology" could be read as runtime parity, yet Section 6.3 reports MST and RNG running 8.19x and 27.07x slower than hexagonal at large grid sizes. If Section 6.2 means only that scaling-efficiency shape is similar across topologies (not absolute runtime), that should be stated explicitly, since as written the two sections read as contradictory.

### Response

We agree and have qualified Section 6.2 to distinguish dimension and sample scaling from grid-size scaling. Across the dimension and sample workloads, the topologies show similar runtime and GPU-efficiency profiles, whereas increasing the number of SOM nodes produces large topology-dependent runtime differences. This makes explicit that the dimension- and sample-scaling patterns and the grid-size runtime penalties are compatible findings.

### Manuscript Amendment

In Section 6.2, we revised:

> "Across these increasingly demanding loads, runtime and efficiency show a broadly consistent scaling pattern, with stable behaviour regardless of topology (Figs. 11 and S11)."

to:

> "Across dimension and sample scaling, hexagonal, MST, and RNG show similar qualitative runtime and GPU-efficiency trends, with similar absolute runtime levels at the largest tested axis values (4.70% pairwise spread for dimension scaling and 3.26% for sample scaling; Fig. 13A,B,D,E; Fig. S6). Grid-size scaling shows a different pattern: topology-dependent runtime and scaling behaviour diverge as the number of SOM nodes increases. At grid size 64, MST and RNG take 8.19x and 27.07x the hexagonal runtime, respectively; this grid-size regime is analyzed in Section 6.3."

In Section 6.2.2, we also revised the Fig. S6 sentence to state that the dimension- and sample-scaling panels show similar qualitative runtime and GPU-efficiency trends across topologies, but that this pattern does not extend to grid-size scaling.

## 3. Somoclu and GigaSOM comparison

### Reviewer Comment

> No system named as closest related work (Somoclu, GigaSOM) is empirically compared against; only single-GPU XPySOM is benchmarked, despite the introduction's "systems gap" argument being framed explicitly around these tools.

### Response

We agree that the baseline selection required clearer justification. We expanded the related-work discussion, attempted an aweSOM benchmark, and now explain why XPySOM was retained as the executable comparator. XPySOM most closely matches FloatSOM's Python/GPU batch-training pathway, while differences in execution model, language, hardware, and published benchmark design prevent controlled direct comparisons with Somoclu and GigaSOM in the present study. The manuscript now states this limitation explicitly.

### Manuscript Amendment

In Section 2.1, we added the baseline-selection rationale:

> "Open-source SOM libraries range from lightweight implementations to systems-oriented packages. MiniSom implements classical serial-online training [@vettigliJustGlowingMinisom2018], while aweSOM adds CPU/GPU acceleration and ensemble stacking for large single-node workloads [@haAweSOMCPUGPUaccelerated2025]. XPySOM instead provides GPU-accelerated batch training and is the closest executable comparator to FloatSOM's Python/GPU training pathway [@manciniXPySomHighPerformanceSelfOrganizing2020]."

> "Somoclu and GigaSOM provide additional parallel-systems context [@wittekSomocluEfficientParallel2017; @kratochvilGigaSOMjlHighperformanceClustering2020]."

We discuss Somoclu and GigaSOM in further detail in the Discussion (Section 8), including the published systems context and why these results are not controlled head-to-head benchmarks.

In Section 4.2, we added the baseline selection and aweSOM attempt:

> "XPySOM was selected as the executable external baseline because it most closely matches FloatSOM's Python/GPU batch-training regime. We also attempted aweSOM on the standard speed workload, but all five runs reached the 1800-s timeout before completing 60% of $N$ online updates. Matching FloatSOM's 10 batch iterations would require $10N$ pointwise updates, so aweSOM was not included in the timed comparison."

In the Discussion (Section 8), we added the detailed Somoclu and GigaSOM comparison and clarified why these published results are not controlled head-to-head benchmarks:

> "Somoclu and GigaSOM are important parallel-systems references, but neither is a controlled head-to-head baseline in this study. Somoclu was already benchmarked against XPySOM in the XPySOM study, which reported a speed difference of more than 10-fold in favor of XPySOM [@manciniXPySomHighPerformanceSelfOrganizing2020]. Because we conduct our own benchmarks against XPySOM, a further direct Somoclu comparison would be unlikely to provide much additional performance information. GigaSOM is a CPU-based algorithm implemented in Julia. In the large IMPC example, the published code selected and scaled 18 input columns, applying the asinh transform to 12 marker columns, and completed a workflow that trained a $32 \times 32$ SOM on 1,167,129,317 cells in under 25 minutes on an 11-node, 256-core CPU cluster [@kratochvilGigaSOMjlHighperformanceClustering2020]. FloatSOM trained a same-sized 1024-node SOM on 1,000,000,000 samples with 50 features in 6.16 minutes using 8 GPUs across two HPC nodes, representing an overall input dataset approximately 2.4 times larger. Note that these results are not a fair head-to-head comparison because the hardware requirements, programming languages, training regimes, timed workloads, and benchmark designs differ. Conducting a full systems benchmark of CPU-based systems against GPU-based systems, or reimplementing GigaSOM in Python to enable one, is outside the scope of this paper."

## 4. Multiple-comparison correction

### Reviewer Comment

> With a large number of paired significance tests reported (14 datasets x 3 metrics x multiple comparisons), there is no correction for multiple comparisons and no statement of whether per-dataset significance stars are intended as primary evidence or as descriptive/exploratory annotation.

### Response

We agree that the statistical reporting should state the correction policy. The revised manuscript now reports Benjamini-Hochberg adjusted q-values for dataset-level families of related paired tests, while retaining raw p-values for audit. For the topology comparisons, the q-values are computed over exactly the family raised by the reviewer: 42 non-global dataset-level tests per topology comparison, corresponding to 14 datasets across $QE_B$, $QE_H$, and $QE_T$, separately for hexagonal-versus-MST and hexagonal-versus-RNG. Dataset-level figure significance markers and dataset-level significance counts use the adjusted q-values. Global pooled rows are reported separately as overall summaries and are not included in the dataset-level adjustment family.

The added radius-control analysis applies a separate prespecified correction across 54 within-topology radius--metric tests. Its post hoc cross-topology analysis applies Benjamini--Hochberg correction across all 21 topology-pair--radius $QE_B$ contrasts, reported with estimates and confidence intervals in Supplementary Table S15.

### Manuscript Amendment

In Section 4.4, we added:

> "We report Benjamini-Hochberg q-values alongside raw p-values for related dataset-level tests, and use q-values for figure markers and significance counts. Each topology contrast in Figs. 6-7 forms a separate family of 42 tests (14 datasets across $QE_B$, $QE_H$, and $QE_T$). Pooled overall tests are reported separately with raw p-values."

Figure captions and Supplementary Table S7 were updated to use q-value language for dataset-level significance markers.

## 5. In-memory quality path versus distributed scaling path

### Reviewer Comment

> The quality benchmarks (Figs. 3-9) run on the in-memory batch path, while the scaling benchmarks (Figs. 10-13) run on the distributed path; the joint claim of "better QE and better scalability" is never shown on the same execution path, and this decoupling is not discussed.

### Response

We agree that the submitted version did not make this execution-path distinction prominent enough. The revised Methods now explicitly state that the Optuna quality runs use the standard in-memory batch path, whereas the speed-scaling results use the Ray-orchestrated distributed execution layer. We also revised the deployment comparison language so that Fig. 15 is described as an integrated deployment comparison rather than a pure topology-only attribution.

To test whether the diagnostic conclusions depended on the execution path, we added a matched benchmark comparing local CuPy with Ray streaming under identical datasets, seeds, topologies, and fixed tuned configurations. No comparison reached statistical significance before or after Benjamini-Hochberg correction, and the estimated differences were small (Supplementary Table S14). We report this diagnostic after the tuning and stability results and before the Ray-pathway performance benchmarks.

### Manuscript Amendment

In Section 4.1, we state:

> "The reported runs used the standard in-memory batch path rather than the Ray-distributed execution stack."

In Section 4.2, we state:

> "Accordingly, the scaling figures in Sections 6.1-6.2 and the runtime/scaling comparison reported later against XPySOM should be interpreted as distributed-execution results rather than the in-memory Optuna path."

In Section 7, we revised the opening framing:

> "Fig. 15 compares untuned hexagonal XPySOM with tuned FloatSOM RNG and therefore combines implementation, tuning, and topology effects [@manciniXPySomHighPerformanceSelfOrganizing2020]."

In Section 5.5, we added:

> "The quality analyses above used the in-memory pathway. Repeating the tuned diagnostics through Ray streaming with matched data and configurations produced small differences, none of which reached significance before or after Benjamini-Hochberg correction (Supplementary Table S14). The remaining differences may reflect floating-point accumulation order. The following speed benchmarks use the Ray pathway."

In Section 4.3.4, we added the complete concordance protocol: 14 datasets, seeds 42--51, three topologies, full sampling, one V100, identical fixed tuned configurations and data splits, Ray-minus-local paired differences, 95% paired confidence intervals, and Benjamini--Hochberg correction across the 12 topology--metric tests.

We also added Supplementary Table S14, which reports the mean paired difference, standard deviation of the paired difference, 95% confidence interval, and Benjamini-Hochberg adjusted q-value for the local CuPy versus Ray streaming execution-path comparison.

## 6. HDSSSOM pilot scope

### Reviewer Comment

> HDSSSOM is eliminated from the rest of the study (Section 5.2) using a smaller pilot (10 datasets/5 seeds) than the main protocol (14/10) used everywhere else. This should be validated at full scale, or at minimum flagged as provisional, before being used to justify dropping a whole sampling method from subsequent analysis.

### Response

We agree that the smaller protocol should not be presented as a full-scale elimination study. The revised manuscript frames HDSSSOM as a focused screening pilot and explicitly states that the result should be interpreted under the stated pilot configuration rather than as a comprehensive evaluation of all possible HDSSSOM schedules.

### Manuscript Amendment

In Section 5.2, we added:

> "These HDSSSOM results should therefore be interpreted as a pilot screen under the stated configuration rather than as a comprehensive evaluation of all possible HDSSSOM schedules."

## 7. RNG runtime penalty in the practical recommendation

### Reviewer Comment

> RNG's 27x runtime penalty over hexagonal at large grid sizes (Section 6.3) is reduced to a brief caveat in the Discussion's general recommendation to use RNG (Section 8.5); given the size of the effect, this deserves a clearer statement of where the recommendation breaks down.

### Response

We agree that the RNG recommendation must be explicitly conditional on runtime budget and grid size. The revised Discussion identifies RNG as the default topology for most datasets, while qualifying this recommendation for very large grids. It identifies MST as the preferable graph alternative when RNG's topology overhead is prohibitive and reports the measured hexagonal, MST, and RNG runtimes at the largest tested grid size.

The geometric rationale makes that conditional recommendation more precise: RNG retains more geometry-supported local connections than MST, which may benefit concentrated regions, but the same additional topology work contributes to its higher grid-size cost. Thus MST's greater freedom and lower graph cost make it the practical compromise when RNG's mesh-like local connectivity is not worth the runtime penalty.

### Manuscript Amendment

In Section 8, we revised the recommendation to:

> "Taken together, the topology results support RNG as the default topology for most datasets. This recommendation is caveated for workloads requiring very large grid sizes, and hence very large numbers of nodes. At grid size 64, hexagonal, MST, and RNG required 32.54, 266.45, and 880.83 s on 8 GPUs, respectively. RNG's graph-construction and all-pairs path calculations therefore scale much more sharply with grid size, so MST is preferable when a large graph is required and this topology overhead is prohibitive."

## 8. Discussion structure

### Reviewer Comment

> Sections 5.3-5.4 and especially Discussion Section 8 (8.1-8.5) fragment an otherwise continuous, well-sequenced argument into short subsections of 2-3 sentences each. Consolidating into flowing prose with topic-sentence transitions, particularly in the Discussion, would let the paper's coherent narrative come through more clearly.

### Response

We agree that the Discussion should read as a continuous argument rather than a list of short independent subsections. We consolidated it into a continuous sequence covering sampling, topology, tuning, stability, and systems limits, while removing repeated interpretation already given in the Results.

### Manuscript Amendment

Section 8 now combines the previous short subsections into a continuous sequence of topic-led paragraphs, retaining the key systems interpretation points, topology-runtime cost, and grid-size-dependent recommendations. The matched execution-path validation is now reported in Section 5.5 immediately before the speed-scaling results.

Within that continuous Discussion, we added a single geometric progression from fixed-lattice coupling, through MST's freedom and tree limitation, to RNG's geometry-supported additional connections. We then connect that interpretation cautiously to the observed $QE$ results before turning to the measured runtime limits.

## 9. Appendix redundancy

### Reviewer Comment

> Several supplementary figures (S5-S7, S8-S10) largely re-confirm robustness without adding distinct conclusions, and could be compressed into single multi-panel summaries or a single in-text sentence pointing to a consolidated figure. This would help offset the length added by moving the load-bearing results identified above into the main text.

### Response

We agree that the supplement should prioritize results that materially support the main claims. In revision, we compressed the redundant robustness material into two consolidated supplementary figures: Supplementary Fig. S4 for the topology top-$k$ sensitivity analyses and Supplementary Fig. S5 for the topology-stratified tuned-versus-untuned comparisons. We also moved the direct MST-versus-RNG comparison into the main text as Fig. 8, so the central topology-ordering evidence is no longer relegated to the appendix.

### Manuscript Amendment

Supplementary Fig. S4 now consolidates the topology sensitivity analyses, and Supplementary Fig. S5 now consolidates the topology-stratified tuned-versus-untuned comparisons. This retains the audit trail while reducing repeated figure-level interpretation.

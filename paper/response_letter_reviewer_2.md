# Response Letter Draft: Reviewer 2

We thank the reviewer for the careful and constructive assessment. We agree with the central framing: FloatSOM is primarily a systems and tooling contribution, and the manuscript must keep the scope of its claims aligned with the evidence shown in the main text. In revision, we have focused on making the topology, scaling, and statistical claims more explicit, and on separating evidence that was directly demonstrated from evidence that should be treated as supporting context or a limitation.

## 1. Direct MST-versus-RNG evidence in the main text

### Reviewer Comment

> The paper's principal empirical conclusion is that RNG is the strongest topology, yet the main text never directly establishes that ordering. Section 5.3 compares MST and RNG independently against the hexagonal baseline, but the direct MST-versus-RNG comparison appears only in the supplementary material (Figs. S4-S7). Because the Discussion ultimately recommends RNG over the other topology options, this comparison is central to the paper's argument and should be presented in the main text rather than relegated to the appendix.

### Response

We agree that the direct MST-versus-RNG comparison is necessary to support a topology-ordering claim. We have promoted that comparison from the supplement to the main topology results as Fig. 8. We also narrowed the conclusion: MST and RNG are close competitors on $QE$, while the matched MTR diagnostics provide the additional evidence favoring RNG.

### Manuscript Amendment

In Section 5.3, we added Fig. 8 and a direct MST-versus-RNG paragraph after the separate hexagonal-versus-MST and hexagonal-versus-RNG comparisons:

> "Fig. 8 directly compares MST and RNG under full sampling. The two graph topologies are close on $QE$, and neither dominates across all three endpoints."

## 2. Section 6.2 versus Section 6.3 scaling language

### Reviewer Comment

> Section 6.2 claims "broadly consistent scaling... regardless of topology," citing Figs. 11 and S11, but Fig. 11 itself shows RNG only -- the hexagonal/MST curves substantiating this sit in the supplement. More importantly, "regardless of topology" could be read as runtime parity, yet Section 6.3 reports MST and RNG running 8.19x and 27.07x slower than hexagonal at large grid sizes. If Section 6.2 means only that scaling-efficiency shape is similar across topologies (not absolute runtime), that should be stated explicitly, since as written the two sections read as contradictory.

### Response

We agree and have chosen the first remedy suggested by the reviewer: we qualified Section 6.2 so that it refers specifically to the sample-scaling workload, where the topologies exhibit similar scaling characteristics. We also added a direct contrast with Section 6.3, where increasing the number of SOM nodes produces large topology-dependent runtime differences. This makes explicit that the sample-scaling efficiency pattern and the grid-size runtime penalties are compatible findings.

### Manuscript Amendment

In Section 6.2, we revised:

> "Across these increasingly demanding loads, runtime and efficiency show a broadly consistent scaling pattern, with stable behaviour regardless of topology (Figs. 11 and S11)."

to:

> "Section 6.2 primarily concerns sample scaling. In that setting, the topologies exhibit similar scaling characteristics: as sample count increases, the GPU-count response and efficiency curves have similar qualitative shapes for RNG (Fig. 12B,E) and for the corresponding hexagonal and MST outputs (Fig. S6). Conversely, when the number of SOM nodes is increased (grid-size scaling), the topologies differ substantially. That grid-size regime is analyzed in Section 6.3, where MST and RNG take 8.19x and 27.07x the hexagonal runtime, respectively, at the largest tested grid size."

In Section 6.2.2, we also revised the Fig. S6 sentence to state that Fig. 12 and Fig. S6 support the same qualitative sample-scaling efficiency trend, but that this trend does not extend to grid-size scaling.

## 3. Somoclu and GigaSOM comparison

### Reviewer Comment

> No system named as closest related work (Somoclu, GigaSOM) is empirically compared against; only single-GPU XPySOM is benchmarked, despite the introduction's "systems gap" argument being framed explicitly around these tools.

### Response

We agree that the baseline selection required clearer justification. We expanded the related-work discussion, attempted an aweSOM benchmark, and now explain why XPySOM was retained as the executable comparator. XPySOM most closely matches FloatSOM's Python/GPU batch-training pathway, while differences in execution model, language, hardware, and published benchmark design prevent controlled direct comparisons with Somoclu and GigaSOM in the present study. The manuscript now states this limitation explicitly.

### Manuscript Amendment

In Section 2.1, we added the baseline-selection rationale:

> "Open-source SOM libraries range from lightweight implementations to systems-oriented packages. MiniSom implements classical serial-online training [@vettigliJustGlowingMinisom2018], while aweSOM adds CPU/GPU acceleration and ensemble stacking for large single-node workloads [@haAweSOMCPUGPUaccelerated2025]. XPySOM instead provides GPU-accelerated batch training and is the closest executable comparator to FloatSOM's Python/GPU training pathway [@manciniXPySomHighPerformanceSelfOrganizing2020]."

> "Somoclu and GigaSOM provide additional parallel-systems context [@wittekSomocluEfficientParallel2017; @kratochvilGigaSOMjlHighperformanceClustering2020]. GigaSOM.jl, for example, trained a $32 \times 32$ SOM on 1,167,129,317 cells within a larger Julia analysis completed in under 25 minutes on an 11-node, 256-core CPU cluster [@kratochvilGigaSOMjlHighperformanceClustering2020]. These systems differ from FloatSOM in training regime, execution model, language, or hardware, so we treat their published results as systems context rather than controlled benchmarks."

In Section 4.2, we added the baseline selection and aweSOM attempt:

> "XPySOM was selected as the executable external baseline because it most closely matches FloatSOM's Python/GPU batch-training regime. We also attempted aweSOM on the standard speed workload, but all five runs reached the 1800-s timeout before completing 60% of $N$ online updates. Matching FloatSOM's 10 batch iterations would require $10N$ pointwise updates, so aweSOM was not included in the timed comparison."

We also added the GigaSOM context:

> "GigaSOM.jl, for example, trained a $32 \times 32$ SOM on 1,167,129,317 cells within a larger Julia analysis completed in under 25 minutes on an 11-node, 256-core CPU cluster [@kratochvilGigaSOMjlHighperformanceClustering2020]."

## 4. Multiple-comparison correction

### Reviewer Comment

> With a large number of paired significance tests reported (14 datasets x 3 metrics x multiple comparisons), there is no correction for multiple comparisons and no statement of whether per-dataset significance stars are intended as primary evidence or as descriptive/exploratory annotation.

### Response

We agree that the statistical reporting should state the correction policy. The revised manuscript now reports Benjamini-Hochberg adjusted q-values for dataset-level families of related paired tests, while retaining raw p-values for audit. For the topology comparisons, the q-values are computed over exactly the family raised by the reviewer: 42 non-global dataset-level tests per topology comparison, corresponding to 14 datasets across $QE_B$, $QE_H$, and $QE_T$, separately for hexagonal-versus-MST and hexagonal-versus-RNG. Dataset-level figure significance markers and dataset-level significance counts use the adjusted q-values. Global pooled rows are reported separately as overall summaries and are not included in the dataset-level adjustment family.

### Manuscript Amendment

In Section 4.4, we added:

> "We report Benjamini-Hochberg q-values alongside raw p-values for related dataset-level tests, and use q-values for figure markers and significance counts. Each topology contrast in Figs. 6-7 forms a separate family of 42 tests (14 datasets across $QE_B$, $QE_H$, and $QE_T$). Pooled overall tests are reported separately with raw p-values."

Figure captions and Supplementary Table S7 were updated to use q-value language for dataset-level significance markers.

## 5. In-memory quality path versus distributed scaling path

### Reviewer Comment

> The quality benchmarks (Figs. 3-9) run on the in-memory batch path, while the scaling benchmarks (Figs. 10-13) run on the distributed path; the joint claim of "better QE and better scalability" is never shown on the same execution path, and this decoupling is not discussed.

### Response

We agree that the submitted version did not make this execution-path distinction prominent enough. The revised Methods now explicitly state that the Optuna quality runs use the standard in-memory batch path, whereas the speed-scaling results use the Ray-orchestrated distributed execution layer. We also revised the deployment comparison language so that Fig. 14 is described as an integrated deployment comparison rather than a pure topology-only attribution.

To test whether the diagnostic conclusions depended on the execution path, we added a matched benchmark comparing local CuPy with Ray streaming under identical datasets, seeds, topologies, and fixed tuned configurations. No comparison reached statistical significance before or after Benjamini-Hochberg correction, and the estimated differences were small (Supplementary Table S14). We report this diagnostic after the tuning and stability results and before the Ray-pathway performance benchmarks.

### Manuscript Amendment

In Section 4.1, we state:

> "The reported runs used the standard in-memory batch path rather than the Ray-distributed execution stack."

In Section 4.2, we state:

> "Accordingly, the scaling figures in Sections 6.1-6.2 and the runtime/scaling comparison reported later against XPySOM should be interpreted as distributed-execution results rather than the in-memory Optuna path."

In Section 7, we revised the opening framing:

> "Fig. 14 compares untuned hexagonal XPySOM with tuned FloatSOM RNG and therefore combines implementation, tuning, and topology effects [@manciniXPySomHighPerformanceSelfOrganizing2020]."

In Section 5.5, we added:

> "The quality analyses above used the in-memory pathway. Repeating the tuned diagnostics through Ray streaming with matched data and configurations produced small differences, none of which reached significance before or after Benjamini-Hochberg correction (Supplementary Table S14). The remaining differences may reflect floating-point accumulation order. The following speed benchmarks use the Ray pathway."

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

We agree that the RNG recommendation must be explicitly conditional on runtime budget and grid size. The revised Discussion now states that RNG is the preferred topology only when $QE$ is the priority and topology-construction overhead is acceptable. It also states that for workloads dominated by very large grids, hexagonal remains the throughput-oriented default and MST can be a practical compromise when graph-based topology is desired but RNG's blocker-test cost is too high.

### Manuscript Amendment

In Section 8, we revised the recommendation to:

> "RNG is the preferred topology when $QE$ is the priority and topology overhead is acceptable, but this recommendation depends on grid size. At grid size 64, hexagonal, MST, and RNG required 32.54, 266.45, and 880.83 s on 8 GPUs, respectively. Hexagonal is therefore the throughput-oriented choice for very large grids, while MST provides a compromise when a graph topology is desired."

## 8. Discussion structure

### Reviewer Comment

> Sections 5.3-5.4 and especially Discussion Section 8 (8.1-8.5) fragment an otherwise continuous, well-sequenced argument into short subsections of 2-3 sentences each. Consolidating into flowing prose with topic-sentence transitions, particularly in the Discussion, would let the paper's coherent narrative come through more clearly.

### Response

We agree that the Discussion should read as a continuous argument rather than a list of short independent subsections. We consolidated it into a continuous sequence covering sampling, topology, tuning, stability, and systems limits, while removing repeated interpretation already given in the Results.

### Manuscript Amendment

Section 8 now combines the previous short subsections into a continuous sequence of topic-led paragraphs, retaining the key systems interpretation points, topology-runtime cost, and grid-size-dependent recommendations. The matched execution-path validation is now reported in Section 5.5 immediately before the speed-scaling results.

## 9. Appendix redundancy

### Reviewer Comment

> Several supplementary figures (S5-S7, S8-S10) largely re-confirm robustness without adding distinct conclusions, and could be compressed into single multi-panel summaries or a single in-text sentence pointing to a consolidated figure. This would help offset the length added by moving the load-bearing results identified above into the main text.

### Response

We agree that the supplement should prioritize results that materially support the main claims. In revision, we compressed the redundant robustness material into two consolidated supplementary figures: Supplementary Fig. S4 for the topology top-$k$ sensitivity analyses and Supplementary Fig. S5 for the topology-stratified tuned-versus-untuned comparisons. We also moved the direct MST-versus-RNG comparison into the main text as Fig. 8, so the central topology-ordering evidence is no longer relegated to the appendix.

### Manuscript Amendment

Supplementary Fig. S4 now consolidates the topology sensitivity analyses, and Supplementary Fig. S5 now consolidates the topology-stratified tuned-versus-untuned comparisons. This retains the audit trail while reducing repeated figure-level interpretation.

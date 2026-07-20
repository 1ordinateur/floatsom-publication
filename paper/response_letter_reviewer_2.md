# Response Letter Draft: Reviewer 2

We thank the reviewer for the careful and constructive assessment. We agree with the central framing: FloatSOM is primarily a systems and tooling contribution, and the manuscript must keep the scope of its claims aligned with the evidence shown in the main text. In revision, we have focused on making the topology, scaling, and statistical claims more explicit, and on separating evidence that was directly demonstrated from evidence that should be treated as supporting context or a limitation.

## 1. Direct MST-versus-RNG evidence in the main text

### Reviewer Comment

> The paper's principal empirical conclusion is that RNG is the strongest topology, yet the main text never directly establishes that ordering. Section 5.3 compares MST and RNG independently against the hexagonal baseline, but the direct MST-versus-RNG comparison appears only in the supplementary material (Figs. S4-S7). Because the Discussion ultimately recommends RNG over the other topology options, this comparison is central to the paper's argument and should be presented in the main text rather than relegated to the appendix.

### Response

We agree that the direct MST-versus-RNG comparison is necessary to support our topology recommendation, and we have promoted it from the supplement to the main topology results as Fig. 8. In the matched full-sampling Optuna top-$k$ analysis, RNG achieves significantly lower balanced and train $QE$ than MST, while no significant holdout-$QE$ difference is detected. The hyperparameter analysis also gives RNG the lowest overall stability score across seeds and datasets.

Taken together, these results support RNG as our quality-first default: it provides the strongest attainable $QE$ performance and the most reproducible tuned settings. MST remains a lower-cost alternative for runtime-constrained workloads, particularly at large grid sizes. We therefore do not claim that RNG dominates every endpoint or every deployment regime.

We also revised the geometric explanation so that the RNG preference is not reduced to graph density alone. MST minimizes neighborhood coupling and permits substantial redistribution along irregular structures, but a tree cannot represent multiple locally appropriate connections in a dense region. RNG permits such connections only when supported by the evolving prototype geometry, whereas the hexagonal lattice imposes a uniform neighbor pattern before learning. The observed $QE$ results are presented as consistent with this account rather than as identifying a unique causal mechanism.

### Summary of Manuscript Changes

We moved the direct MST-versus-RNG comparison into the main text as Fig. 8. Section 5.3 now reports that RNG has lower balanced and train $QE$ than MST in the matched Optuna top-$k$ analysis, while no holdout-$QE$ difference was detected, providing direct support for the stated topology ordering.

## 2. Section 6.2 versus Section 6.3 scaling language

### Reviewer Comment

> Section 6.2 claims "broadly consistent scaling... regardless of topology," citing Figs. 11 and S11, but Fig. 11 itself shows RNG only -- the hexagonal/MST curves substantiating this sit in the supplement. More importantly, "regardless of topology" could be read as runtime parity, yet Section 6.3 reports MST and RNG running 8.19x and 27.07x slower than hexagonal at large grid sizes. If Section 6.2 means only that scaling-efficiency shape is similar across topologies (not absolute runtime), that should be stated explicitly, since as written the two sections read as contradictory.

### Response

We agree and have qualified Section 6.2 to distinguish dimension and sample scaling from grid-size scaling. Across the dimension and sample workloads, the topologies show similar runtime and GPU-efficiency profiles, whereas increasing the number of SOM nodes produces large topology-dependent runtime differences. This makes explicit that the dimension- and sample-scaling patterns and the grid-size runtime penalties are compatible findings.

### Summary of Manuscript Changes

We revised the scaling language to distinguish similarity in scaling behavior from parity in absolute runtime. Section 6.2 now points to the topology-specific curves and reports that the maximum pairwise runtime spread remained 4.70% at 5,000 dimensions and 3.26% at one billion samples, while Section 6.3 separately presents the much larger topology-dependent costs that arise as grid size increases.

## 3. Somoclu and GigaSOM comparison

### Reviewer Comment

> No system named as closest related work (Somoclu, GigaSOM) is empirically compared against; only single-GPU XPySOM is benchmarked, despite the introduction's "systems gap" argument being framed explicitly around these tools.

### Response

We agree that the baseline selection required clearer justification. We expanded the related-work discussion, attempted an aweSOM benchmark, and now explain why XPySOM was retained as the executable comparator. Somoclu was not benchmarked against because both Somoclu and SomocluGPU were already compared directly with XPySOM in the XPySOM study and were more than 10 times slower in its reported benchmark [@manciniXPySomHighPerformanceSelfOrganizing2020]. As XPySOM most closely matches FloatSOM's Python/GPU batch-training pathway, we retained it as the controlled executable comparator. Regarding GigaSOM, differences in execution model, language, hardware, and published benchmark design likewise prevent a controlled direct comparison in the present study. The manuscript now states these limitations explicitly.

### Summary of Manuscript Changes

We expanded the related-work and Discussion sections to explain the baseline selection. XPySOM was retained as the controlled executable comparator because it most closely matches FloatSOM's Python/GPU batch pathway. We also now note that XPySOM has already been compared directly with Somoclu and SomocluGPU with favourable findings to XPySOM. Finally, we note the GigaSOM differences in execution model, language, hardware, training regime, and workload design preventing a fair direct comparison with FloatSOM.

## 4. Multiple-comparison correction

### Reviewer Comment

> With a large number of paired significance tests reported (14 datasets x 3 metrics x multiple comparisons), there is no correction for multiple comparisons and no statement of whether per-dataset significance stars are intended as primary evidence or as descriptive/exploratory annotation.

### Response

We agree that the statistical reporting should state the correction policy. The revised manuscript now reports Benjamini-Hochberg adjusted q-values for dataset-level families of related paired tests, while retaining raw p-values for audit. For the topology comparisons, the q-values are computed over exactly the family raised by the reviewer: 42 non-global dataset-level tests per topology comparison, corresponding to 14 datasets across $QE_B$, $QE_H$, and $QE_T$, separately for hexagonal-versus-MST and hexagonal-versus-RNG. Dataset-level figure significance markers and dataset-level significance counts use the adjusted q-values. Global pooled rows are reported separately as overall summaries and are not included in the dataset-level adjustment family.

The added radius-control analysis applies a separate prespecified correction across 54 within-topology radius--metric tests. Its post hoc cross-topology analysis applies Benjamini--Hochberg correction across all 21 topology-pair--radius $QE_B$ contrasts, reported with estimates and confidence intervals in Supplementary Table S15.

### Summary of Manuscript Changes

We now report Benjamini--Hochberg adjusted q-values alongside raw p-values for the dataset-level tests. Each topology contrast uses a separate family of 42 tests (14 datasets across balanced, holdout, and train $QE$), and the adjusted q-values determine dataset-level figure markers and significance counts. Pooled overall tests are reported separately.

## 5. In-memory quality path versus distributed scaling path

### Reviewer Comment

> The quality benchmarks (Figs. 3-9) run on the in-memory batch path, while the scaling benchmarks (Figs. 10-13) run on the distributed path; the joint claim of "better QE and better scalability" is never shown on the same execution path, and this decoupling is not discussed.

### Response

We agree that the submitted version did not make this execution-path distinction prominent enough. The revised Methods now explicitly state that the Optuna quality runs use the standard in-memory batch path, whereas the speed-scaling results use the Ray-orchestrated distributed execution layer. We also revised the deployment comparison language so that Fig. 15 is described as an integrated deployment comparison rather than a pure topology-only attribution.

To test whether the diagnostic conclusions depended on the execution path, we added a matched benchmark comparing local CuPy with Ray streaming under identical datasets, seeds, topologies, and fixed tuned configurations. No comparison reached statistical significance before or after Benjamini-Hochberg correction, and the estimated differences were small (Supplementary Table S14). We report this diagnostic after the tuning and stability results and before the Ray-pathway performance benchmarks.

### Summary of Manuscript Changes

We now identify the execution pathway used by each experiment: the quality and Optuna analyses use the in-memory batch path, whereas the scaling and XPySOM runtime experiments use Ray distributed execution. We added a matched concordance analysis across 14 datasets, 10 seeds, three topologies, and identical tuned configurations; the small Ray-minus-local differences were not significant before or after correction. Section 5.5 and Supplementary Table S14 report the protocol, paired differences, confidence intervals, and adjusted q-values, while Section 7 separately retains the caveat that the deployment comparison combines implementation, tuning, and topology effects.

## 6. HDSSSOM pilot scope

### Reviewer Comment

> HDSSSOM is eliminated from the rest of the study (Section 5.2) using a smaller pilot (10 datasets/5 seeds) than the main protocol (14/10) used everywhere else. This should be validated at full scale, or at minimum flagged as provisional, before being used to justify dropping a whole sampling method from subsequent analysis.

### Response

We agree that the smaller protocol should not be presented as a full-scale elimination study. The revised manuscript frames HDSSSOM as a focused screening pilot and explicitly states that the result should be interpreted under the stated pilot configuration rather than as a comprehensive evaluation of all possible HDSSSOM schedules.

### Summary of Manuscript Changes

We strengthened the qualification in Section 5.2 so that HDSSSOM is described as a focused pilot screen under the tested smaller configuration, rather than a full-scale elimination study or a comprehensive assessment of all possible schedules.

## 7. RNG runtime penalty in the practical recommendation

### Reviewer Comment

> RNG's 27x runtime penalty over hexagonal at large grid sizes (Section 6.3) is reduced to a brief caveat in the Discussion's general recommendation to use RNG (Section 8.5); given the size of the effect, this deserves a clearer statement of where the recommendation breaks down.

### Response

We agree that the RNG recommendation must be explicitly conditional on runtime budget and grid size. The revised Discussion identifies RNG as the default topology for most datasets, while qualifying this recommendation for very large grids. It identifies MST as the preferable graph alternative when RNG's topology overhead is prohibitive and reports the measured hexagonal, MST, and RNG runtimes at the largest tested grid size.

The geometric rationale makes that conditional recommendation more precise: RNG retains more geometry-supported local connections than MST, which may benefit concentrated regions, but the same additional topology work contributes to its higher grid-size cost. Thus MST's greater freedom and lower graph cost make it the practical compromise when RNG's mesh-like local connectivity is not worth the runtime penalty.

### Summary of Manuscript Changes

We made the RNG recommendation conditional on runtime budget and grid size. The Discussion now gives the measured 8-GPU runtimes at grid size 64—32.54 s for hexagonal, 266.45 s for MST, and 880.83 s for RNG—and recommends MST when RNG's graph-construction and path-calculation overhead is prohibitive on large grids. The geometric discussion also connects RNG's additional local connectivity to both its potential quality benefit and its higher cost.

## 8. Discussion structure

### Reviewer Comment

> Sections 5.3-5.4 and especially Discussion Section 8 (8.1-8.5) fragment an otherwise continuous, well-sequenced argument into short subsections of 2-3 sentences each. Consolidating into flowing prose with topic-sentence transitions, particularly in the Discussion, would let the paper's coherent narrative come through more clearly.

### Response

We agree that the Discussion should read as a continuous argument rather than a list of short independent subsections. We consolidated it into a continuous sequence covering sampling, topology, tuning, stability, and systems limits, while removing repeated interpretation already given in the Results.

### Summary of Manuscript Changes

We consolidated the former short Discussion subsections into a continuous sequence of topic-led paragraphs covering sampling, topology, tuning, stability, and systems limitations. Repeated Results interpretation was removed, the execution-path validation was placed in Section 5.5 before the scaling results, and the topology discussion now progresses from fixed-lattice coupling through MST and RNG before turning to the measured runtime trade-offs.

## 9. Appendix redundancy

### Reviewer Comment

> Several supplementary figures (S5-S7, S8-S10) largely re-confirm robustness without adding distinct conclusions, and could be compressed into single multi-panel summaries or a single in-text sentence pointing to a consolidated figure. This would help offset the length added by moving the load-bearing results identified above into the main text.

### Response

We agree that the supplement should prioritize results that materially support the main claims. In revision, we compressed the redundant robustness material into two consolidated supplementary figures: Supplementary Fig. S4 for the topology top-$k$ sensitivity analyses and Supplementary Fig. S5 for the topology-stratified tuned-versus-untuned comparisons. We also moved the direct MST-versus-RNG comparison into the main text as Fig. 8, so the central topology-ordering evidence is no longer relegated to the appendix.

### Summary of Manuscript Changes

We consolidated the redundant supplementary robustness plots into two multi-panel figures: Supplementary Fig. S4 for topology top-$k$ sensitivity and Supplementary Fig. S5 for topology-stratified tuned-versus-untuned comparisons. The direct MST-versus-RNG comparison was promoted to main-text Fig. 8, preserving the supplementary audit trail while removing repeated figure-level interpretation.

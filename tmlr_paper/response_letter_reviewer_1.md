# Response Letter Draft: Reviewer 1

We thank the reviewer for the careful reading and for separating the systems contribution from the quality/topology claims. We agree that the systems evidence is stronger in the submitted version than the topology-quality evidence, and the revision is structured to make that distinction explicit. Where the manuscript already contained the relevant analysis but did not state the point clearly enough, we have clarified the text. Where the reviewer identified a missing control or missing reporting detail, we have added the corresponding amendment below.

## 1. Quality and topology diagnostics

### Reviewer Comment

> The quality claim runs on QE alone. QE measures distance to the best-matching unit, not neighborhood preservation, which is the point of a SOM. No topographic error or trustworthiness, even though Forest et al., 2020 is cited and implements them 

> Report a preservation metric next to QE in every topology comparison. Forest et al., 2020 is already cited and has topographic error and trustworthiness.


### Response

We agree that $QE$ alone does not characterize SOM neighborhood ordering. We carefully considered the reviewer's suggestion to report topographic error. Topographic error records the proportion of samples for which the first and second best-matching units are not immediately adjacent. This binary criterion depends critically on the number of immediate neighbours and therefore does not provide a like-for-like comparison between maps with different topologies. López-Rubio and Díaz Ramos (2014) use MTR to compare alternative SOM grid topologies.

We therefore added Mean Tied Rank (MTR), which groups non-winning units by their shortest-path hop distance from the first BMU and assigns units at the same hop distance their average ordinal rank. This construction accommodates the different adjacency rules of the three topologies: hexagonal adjacency is fixed by the lattice, MST adjacency is a sparse tree derived from the node weights, and RNG adjacency is weight-derived with variable node degree. In every case, MTR uses the map's own graph, counts the units fewer hops away, and accounts for the number of units tied at the second BMU's hop distance. It therefore provides a graded measure of local ordering across the three adjacency structures.

The matched analysis in Table 1. found lower MTR for RNG than for hexagonal maps both with and without tuning. The fixed QE-tuned profiles had lower $QE$ but higher MTR than the untuned profiles, with a much larger MTR difference for hexagonal maps (24.34 tied-rank positions) than for MST (1.43) or RNG (1.11). The main text reports the principal effects, and Supplementary Tables S12-S13 provide the full results.

We also expanded the geometric rationale. A fixed hexagonal lattice predetermines neighborhood coupling independently of the learned data-space geometry, so movement of one prototype can influence lattice neighbors that are not locally related in data space. MST minimizes this coupling and lets prototypes redistribute along irregular or elongated structures, but its tree constraint can omit additional useful local connections in dense regions. RNG is less free because nodes may influence several neighbors, yet those connections are supported by the evolving prototype geometry rather than imposed beforehand. It can therefore remain sparse where appropriate and become more mesh-like in concentrated regions. We present $QE$, MTR, node utilization, and dead-node fraction as evidence consistent with this interpretation, not as proof of a unique causal mechanism.

To illustrate these neighborhood structures beyond a two-dimensional synthetic example, we expanded Fig. 5 to include matched hexagonal, MST, and RNG overlays for KDD Cup 99, trained in standardized 41-dimensional space and displayed using a shared two-dimensional PCA projection. The representative overlays also provide qualitative confirmation that the prototypes are deployed across the observed data structure. In the KDD Cup 99 panels, the hexagonal topology shows several apparently nonlocal connections across separated regions of the projected prototype distribution, whereas the MST and RNG connections more closely follow its local geometry. We interpret this cautiously because the two-dimensional PCA display can distort relationships in the original 41-dimensional space.

### Summary of Manuscript Changes

We added MTR as a graded, topology-aware measure of local neighborhood ordering alongside $QE$, defined how it is calculated for fixed hexagonal and dynamically derived MST/RNG adjacency. Table 1 and Supplementary Tables S12--S13 now report the matched MTR and node-use diagnostics across 14 datasets and 20 seeds. The Results and Discussion now distinguish quantization fidelity from local ordering, report the stronger MTR trade-off observed for tuned hexagonal maps, and identify RNG as the strongest joint $QE$/MTR result while keeping the causal interpretation cautious. We also expanded Fig. 5 with matched KDD Cup 99 topology overlays and added a geometric explanation of fixed-lattice coupling versus graph neighborhoods derived from evolving prototypes.

## 2. Hexagonal neighborhood-radius control

### Reviewer Comment

> Add a control that shrinks the hexagonal neighborhood radius. If a tighter-radius hexagonal map closes most of the QE gap, the graph topologies are winning on looseness, not structure.

### Response

We agree that neighborhood radius is an important potential confound. In addition to the existing Optuna design, in which `initial_radius` was optimized independently for every topology over the same interval and search budget, we performed additional experiment where we iteratively increase the radius while keeping other settings fixed at their optimal value based on the optuna results. The results are illustrated in Figure 9.

Across the entire radius range investigated, MST and RNG consistently attained lower normalised QE than hexagonal, with this being most obvious after $r=1.5$ (Fig. 9A). Regarding each topology's optimal $QE_B$ radius, RNG and MST's optimal occurred at $r=1.5$, which is larger than hexagonal at $r=0.75$. The graph-topology gains therefore cannot be explained by preferential radius treatment or by comparison with a broad default-radius hexagonal map. Even at this optimal point, hexagonal still demonstrated inferior $QE_B$ to both graph topologies, implying that decreased radius alone is not responsible for the observed $QE$ differential. Fig. 9B additionally shows that the sharp hexagonal $QE$--MTR trade-off is substantially attenuated for MST and RNG : RNG maintains lower MTR than MST while its $QE_B$ remains stable. Fig. 9C shows that MST and RNG also maintain high node utilization in this region, which does not support reduced use of map capacity as the explanation for their lower $QE_B$. In both the MTR and Node utilisation, MST and RNG perform consistently better than hexagonal. 

### Summary of Manuscript Changes

We clarified that `initial_radius` was optimized independently for all three topology families over the same 0.5--10.0 interval. We also added a matched seven-radius sensitivity experiment across 14 datasets and 20 seeds, holding the other settings fixed and applying multiplicity-corrected cross-topology tests. The new Fig. 9 summarizes normalized balanced $QE$, MTR, and node utilization; MST and RNG retain lower observed balanced $QE$ than hexagonal maps across the tested range, with adjusted differences from $r=1.027$ onward.

## 3. Related work and external baselines

### Reviewer Comment

> Related work names prior graph and GPU SOMs but does not benchmark against any. aweSOM (Ha et al., JOSS 2025), a GPU Python SOM in the same N > 10^6 range, is not mentioned.

> Add recent GPU and distributed SOM baselines. Somoclu CUDA, GigaSOM, aweSOM. aweSOM is single-node CPU/GPU with ensemble stacking, so it is a quality baseline, not a distributed competitor.

### Response

We agree that the baseline selection required clearer justification. We expanded the related-work discussion and attempted an aweSOM benchmark. All five aweSOM attempts reached the 1800-s timeout before completing 60% of $N$ online updates on the standard speed workload, as referenced in Section 4.2. Matching FloatSOM's 10 full batch iterations would require $10N$ pointwise updates.

We now discuss Somoclu and GigaSOM as important parallel-systems references. Somoclu was not benchmarked against because Mancini et al. (2020) directly compared both Somoclu and SomocluGPU with XPySOM and found them to be more than 10 times slower in the reported benchmark. As we benchmark FloatSOM directly against XPySOM, we retained XPySOM as the controlled executable comparator. This published comparison does not replace a contemporary FloatSOM--Somoclu head-to-head benchmark, and we do not infer a transitive speedup from it. Similarly, GigaSOM is a CPU algorithm coded in Julia. Due to the differences in hardware (with a large CPU cluster, as opposed to our present GPU based hardware) and coding language, a true head-to-head comparison is not possible.

### Summary of Manuscript Changes

We expanded the Introduction and related work to position MiniSom, aweSOM, XPySOM, Somoclu, and GigaSOM according to their training and systems models. Section 4.2 now documents the attempted aweSOM benchmark: all five runs reached the 1,800-second timeout before completing 60% of $N$ online updates, so aweSOM was not included in the timed comparison. We explain why XPySOM remains the closest executable Python/GPU batch comparator and treat Somoclu and GigaSOM as published systems context rather than fair head-to-head baselines. We also added the complete XPySOM calibration protocol and replaced the prior equivalence wording with the narrower finding that no implementation-associated hexagonal $QE$ difference was detected.

## 4. Numbers should be embedded in the paper

### Reviewer Comment

> Quality numbers are not in the paper. Per-dataset effect sizes exist only as external table paths (Tables S4 to S11 are captions pointing at files). They cannot be read off the forest plots either.

> Put the numbers in the paper. Embed the per-dataset effect-size tables instead of external paths.

### Response

We agree. The submitted version included the tables as reproducibility artifacts but did not embed their numerical contents in the manuscript, which makes the paper harder to evaluate independently. We have replaced the path-only supplementary table captions with embedded tables for the XPySOM calibration summaries, topology effect and q-value summary, deployment effect summaries, topology runtime summary, and the new matched-radius cross-topology tests in Supplementary Table S15. The manuscript now contains the numerical values needed to evaluate the corresponding figures.

### Summary of Manuscript Changes

We replaced the path-only supplementary-table captions with embedded numerical tables, including the XPySOM calibration summaries, topology effects and adjusted q-values, deployment effects, topology runtimes, and matched-radius comparisons. The revised supplement therefore contains the values needed to evaluate the corresponding plots directly, including the paired topology comparisons in Table S7 and the deployment summary in Table S8.

## 5. Deployment comparison separates topology, tuning, and implementation

### Reviewer Comment

> Deployment comparison (Section 7, Figure 13) is a different claim. Tuned FloatSOM RNG against untuned hexagonal XPySOM. One number, three changes: topology, tuning, implementation. The 14.5, 9.1, 22.5 percent gains cannot be credited to topology. Sections 5.1, 5.3, and 5.4 already have the pieces to separate them.

> Separate topology from tuning and implementation in the deployment comparison, or tune both systems.

### Response

We agree that Fig. 15 should not be read as attributing the full gain to topology alone. It presents an integrated deployment comparison between an untuned XPySOM run and the recommended tuned FloatSOM RNG configuration. The component effects are evaluated separately in Sections 5.1, 5.3, and 5.4, but the submitted Section 7 did not make this decomposition sufficiently clear.

We revised Section 7 and the Fig. 15 caption so that the 14.5%, 9.1%, and 22.5% improvements are described as an integrated deployment effect rather than a topology-only effect. We also point readers to the tuned-hexagonal and tuned-MST deployment figures in the Supplementary material.

### Summary of Manuscript Changes

We revised Section 7 and the Fig. 15 caption to state explicitly that the deployment comparison combines implementation, tuning, and topology effects; the reported 14.5%, 9.1%, and 22.5% improvements are no longer presented as topology-only gains. The text points readers to Sections 5.1, 5.3, and 5.4 for the component analyses and to Supplementary Figs. S7--S8 for the corresponding tuned hexagonal and MST comparisons.

## 6. MST and RNG novelty claims

### Reviewer Comment

> Drop "novel" for the topologies. Jang et al., 2009, already in the references, put MSTs on SOMs. Frame this as the first version that scales on GPUs.

### Response

We agree that the submitted wording could imply that MSTs had not previously been associated with SOMs. Weno longer claim novelty for the graph object itself. Instead, the contribution is the scalable GPU implementation and large-scale evaluation of dynamically refreshed graph neighborhoods during SOM training. We retain only the narrower, qualified statement that we have not identified prior work using dynamically refreshed RNG neighborhoods for SOM training.

### Summary of Manuscript Changes

We removed the broad claim that graph topologies themselves are novel and now describe the contribution as scalable, GPU-compatible implementation and large-scale evaluation of refreshed graph neighborhoods during SOM training. The related-work section distinguishes training-time MST neighborhoods from post hoc MST visualization while acknowledging prior training-time variants, and the Methods now state explicitly that MST/RNG graphs determine the update neighborhood. We retain only a qualified claim that we found no prior SOM work using dynamically refreshed RNG and MST training neighborhoods.

## 7. Runtime cost of RNG recommendation

### Reviewer Comment

> Measure the runtime cost of the recommended topology. Section 8.5 recommends RNG. At grid size 64, RNG is 27x hexagonal, MST is 8x. The MST distances-to-CPU step for Kruskal is the cheap part. The all-pairs hop-distance step both share plus RNG's blocker test is the expensive part. Put a number on the RNG cost.

### Response

We agree that any recommendation of RNG must be paired with its runtime cost. Fig. 14 reports this cost: at grid size 64, the 8-GPU mean runtime was 32.54 s for hexagonal, 266.45 s for MST, and 880.83 s for RNG, corresponding to 8.19x and 27.07x the hexagonal runtime for MST and RNG, respectively. We now repeat these values at the point where RNG is recommended and make the grid-size limitation explicit.

### Summary of Manuscript Changes

We revised the practical recommendation to make topology cost explicit and grid-size dependent. The Discussion now reports the measured 8-GPU runtimes at grid size 64—32.54 s for hexagonal, 266.45 s for MST, and 880.83 s for RNG—and recommends RNG for the strongest joint $QE$/MTR result only when its overhead is acceptable; MST is identified as the lower-cost graph option when $QE$ is the main objective or large graphs make RNG prohibitive.

## 8. Multiple-comparison correction

### Reviewer Comment

> State whether the paired t-tests were corrected for multiple comparisons. About 42 per topology, 14 datasets by 3 metrics. Seed counts and intervals are already there.

### Response

We agree. The manuscript now reports the multiple-comparison correction method used for the topology comparisons, and the associated figure and table descriptions have been updated accordingly. Only two dataset-level annotations changed from significant to non-significant: `blobs` in Fig. 7B changed from p=0.0358 to q=0.0627, and `iris` changed from p=0.0444 to q=0.0745; the remaining changes affected star levels only.

### Summary of Manuscript Changes

We added a multiple-comparison policy for the topology analyses. Each hexagonal-versus-graph contrast treats the 42 dataset-level tests (14 datasets by three $QE$ metrics) as one Benjamini--Hochberg family, reports q-values alongside raw p-values, and uses q-values for figure markers and significance counts; pooled overall tests remain separate. The affected figure captions and Supplementary Table S7 were updated accordingly.

## 9. Dead-node and node-utilization reporting

### Reviewer Comment

> Add a dead-node or node-utilization count across the three topologies. Cheap, and it bears on the preservation question.

### Response

We agree, and thank the reviewer for pointing this out. Node utilization is a useful confirmatory diagnostic for checking whether the MTR result is accompanied by broadly used map capacity rather than uneven allocation in which some nodes are effectively unused. Considering this, we added node-utilization and dead-node-fraction diagnostics to the matched topology benchmark outputs, using the same matched units as the topology comparison.

The matched tuned-profile result favored RNG rather than indicating poorer map use. Relative to tuned hexagonal maps, tuned RNG increased balanced node utilization by 0.0197 (95% CI 0.0158 to 0.0235; p=1.38e-20) and reduced balanced dead-node fraction by the same amount. We report the split-specific and balanced diagnostics separately and present node utilization as a confirmatory diagnostic alongside MTR.

Quantitative node-utilization and dead-node evidence is shown in Fig. 9 and Supplementary Tables S12-S13. The representative overlays in Fig. 5 additionally provide qualitative confirmation that prototypes are deployed across the observed data structures, although they do not classify individual nodes as active or dead. Together, these results are consistent with the interpretation that geometry-derived neighborhoods avoid some unnecessary fixed-lattice coupling while retaining map capacity; they do not establish that interpretation as a unique causal explanation.

### Summary of Manuscript Changes

We added node utilization and dead-node fraction as post hoc diagnostics on the training and holdout splits, balanced in the same way as $QE$. Table 1 and Supplementary Tables S12--S13 now report these measures with the matched topology diagnostics, allowing the topology comparison to account for whether lower $QE$ is accompanied by reduced map-capacity use.

## 10. HDSSSOM framing

### Reviewer Comment

> Keep the HDSSSOM results framed as the pilot they already are.

### Response

We agree and preserve the current framing. The submitted manuscript already describes HDSSSOM as a focused screening pilot and not as part of the main full-versus-random sampling benchmark. We kept that language and added a sentence to avoid broadening the claim beyond the pilot configuration.

### Summary of Manuscript Changes

We retained HDSSSOM as a focused screening pilot and strengthened that qualification in Section 5.2. The revision now states explicitly that the smaller pilot configuration is an elimination screen under the tested schedule, not a comprehensive evaluation of all possible HDSSSOM schedules.

## 11. Scaling efficiency and extrapolated 1-GPU baselines

### Reviewer Comment

> Some scaling points are extrapolated 1-GPU values, not measured (Section 4.2). Stated in the paper. Still limits the efficiency panels.

### Response

We agree that extrapolated 1-GPU denominators limit the efficiency analysis. The affected 1-GPU jobs did not complete within the fixed benchmark timeout, particularly for workloads entering the disk-backed execution regime, so no measured $T_1$ was available at those points. Because scaling efficiency requires a 1-GPU denominator, we used local extrapolation from the last successful 1-GPU measurement to provide a provisional reference for plotting. These extrapolated values are not measured runtimes and should not be interpreted as validated estimates of single-GPU performance.

We have therefore moderated the interpretation of the efficiency panels. Points with directly measured 1-GPU baselines provide the primary evidence for scaling efficiency. Points using extrapolated baselines are treated as descriptive estimates only, and are not used to support a general claim that efficiency improves across the full workload range. Absolute runtime and directly measured speedup remain the primary scaling results. Efficiencies above 100\% are interpreted as reflecting parallelism together with changes in memory and data-staging regime, rather than superlinear computation.

### Summary of Manuscript Changes

We clarified which 1-GPU scaling denominators were measured and which were locally extrapolated because runs exceeded the benchmark timeout. Extrapolated efficiency values are now presented only as descriptive plotting references and are not used to support general scaling claims. The revised interpretation prioritizes absolute runtimes and directly measured baselines, and notes that efficiencies above 100% may reflect a change in memory or data-staging regime as well as parallel throughput.

# Response Letter Draft: Reviewer 1

We thank the reviewer for the careful reading and for separating the systems contribution from the quality/topology claims. We agree that the systems evidence is stronger in the submitted version than the topology-quality evidence, and the revision is structured to make that distinction explicit. Where the manuscript already contained the relevant analysis but did not state the point clearly enough, we have clarified the text. Where the reviewer identified a missing control or missing reporting detail, we have added the corresponding amendment below.

## 1. Quality and topology diagnostics

### Reviewer Comment

> The quality claim runs on QE alone. QE measures distance to the best-matching unit, not neighborhood preservation, which is the point of a SOM. No topographic error or trustworthiness, even though Forest et al., 2020 is cited and implements them.

### Response

We agree that $QE$ alone does not characterize SOM neighborhood ordering. We added Mean Tied Rank (MTR), which ranks the second BMU by its graph-distance shell around the first, together with node-utilization and dead-node diagnostics. MTR is preferable to raw topographic error here because one-hop adjacency differs across hexagonal, MST, and RNG graphs [@ramosROLELATTICEDIMENSIONALITY2018; @nemeStatisticalPropertiesLattices2005; @machon-gonzalezFLSOMIndividualKernel2010].

The matched analysis found lower MTR for RNG than for hexagonal maps with and without tuning; MST did not clearly lower untuned MTR. Tuning improved $QE$ but increased MTR in every topology, with a much larger increase for hexagonal maps (24.34 tied-rank positions) than for MST (1.43) or RNG (1.11). Node-use results did not indicate reduced map utilization. The main text reports the principal effects, and Supplementary Tables S12-S13 provide the full results.

### Manuscript Amendment

In Section 4.1, we added:

> "We quantify local BMU-neighborhood ordering using Mean Tied Rank (MTR) [@ramosROLELATTICEDIMENSIONALITY2018]. For each sample $x_i$, non-winning units are ranked by graph distance from the first BMU, $b_i^{(1)}$, with tied distance shells assigned their average ordinal rank. If the second BMU, $b_i^{(2)}$, lies in shell $S_d$ and $L_d$ units lie in closer shells, its tied rank is $\tau_i=L_d+(|S_d|+1)/2$, and $MTR=N^{-1}\sum_i \tau_i$. Lower MTR indicates closer local ordering between the first and second BMUs."

> "MTR is used instead of topographic error because one-hop adjacency is not equivalent across hexagonal, MST, and RNG graphs; topographic error also depends on lattice properties and map design [@ramosROLELATTICEDIMENSIONALITY2018; @nemeStatisticalPropertiesLattices2005; @machon-gonzalezFLSOMIndividualKernel2010]. We additionally report node utilization, defined as the fraction of nodes selected as a BMU, and its complement, dead-node fraction. These post hoc diagnostics are computed on training and holdout splits and balanced as for $QE$."

In Section 4.3.1, we clarified the fixed tuned rerun design, including the exact tuned and untuned profiles, 14 datasets, 20 matched seeds, pairing keys, and diagnostic summaries:

> "The final topology diagnostic comprised all 14 benchmark datasets, 20 shared random seeds, full sampling, and each of the hexagonal, MST, and RNG topologies, giving 280 dataset--seed units per paired contrast."

In Section 5.3, we revised the opening sentence to:

> "Topology comparisons use $QE$ as the optimized endpoint and hexagonal batch SOM as the regular-topology baseline; Fig. 5 illustrates the three neighborhood structures. `initial_radius` was optimized over the same interval for every topology."

We also added the tuned and untuned RNG diagnostic result:

> "The fixed-configuration diagnostics provide the corresponding MTR and node-use comparison (Table \ref{tab:matched_topology_diagnostics_summary}). MST and RNG both improved balanced $QE$ relative to hexagonal maps with and without tuning, but only RNG clearly lowered untuned balanced MTR. Under tuning, no balanced-$QE$ difference was detected between MST and RNG, while RNG lowered balanced MTR by 2.96 tied-rank positions. RNG therefore gave the strongest joint $QE$/MTR result, while MST remained a close competitor on $QE$."

In Section 5.4, we added:

> "Tuning improved all three pooled $QE$ metrics. It also increased balanced MTR, by 24.34 tied-rank positions for hexagonal maps compared with 1.43 for MST and 1.11 for RNG. This local-ordering trade-off was therefore much stronger for the fixed lattice and was not accompanied by lower node utilization."

In the Discussion, we added:

> "$QE$ and MTR capture different properties: $QE$ measures vector-quantization fidelity, while MTR measures local ordering between the first and second BMUs. Tuning improves $QE$ across all topologies but increases MTR, with a much larger increase for hexagonal maps than for MST or RNG. Topology and hyperparameter choice should therefore be considered together."

## 2. Hexagonal neighborhood-radius control

### Reviewer Comment

> Add a control that shrinks the hexagonal neighborhood radius. If a tighter-radius hexagonal map closes most of the QE gap, the graph topologies are winning on looseness, not structure.

### Response

We agree that neighborhood radius is an important potential confound. In addition to the existing Optuna design, in which `initial_radius` was optimized independently for every topology over the same interval and search budget, we added the requested radius-only control as Fig. 9. We evaluated seven identical radii for hexagonal, MST, and RNG maps using 20 matched seeds on all 14 datasets, while holding every non-radius training hyperparameter fixed.

MST and RNG retained lower normalized $QE_B$ than hexagonal when compared at the same radii (Fig. 9A). Their lowest observed $QE_B$ occurred at $r=1.5$, which is larger rather than smaller than the hexagonal minimum at $r=0.75$. The graph-topology gains therefore cannot be explained by preferential radius treatment or by comparison with a broad default-radius hexagonal map. Fig. 9B additionally shows that the sharp hexagonal $QE$--MTR trade-off is substantially attenuated for MST and RNG around their useful radius range: RNG maintains lower MTR than MST while its $QE_B$ remains stable around $r=1.027$--1.5. Fig. 9C shows that MST and RNG also maintain high node utilization in this region, excluding reduced use of map capacity as the explanation for their lower $QE_B$.

### Manuscript Amendment

In Section 4.1, we added:

> "The neighborhood-radius search space was shared across topology families. In particular, `initial_radius` was an Optuna-optimized parameter for hexagonal, MST, and RNG runs, with the same search interval of 0.5 to 10.0 in each case. Thus, the hexagonal topology comparisons below use a tuned hexagonal comparator rather than a default-radius hexagonal baseline."

In Section 4.3.3, we added the matched initial-radius sensitivity protocol, including the seven tested radii, 14 datasets, 20 matched seeds, fixed non-radius configuration, normalized-$QE$ calculation, observed-unit MTR and node-utilization summaries, and multiplicity correction.

In Section 5.3, we added:

> "Crucially, MST and RNG retain lower $QE_B$ than hexagonal when evaluated at the same radii with every other training hyperparameter fixed. Their quantization advantage therefore cannot arise from preferential radius treatment or from assigning graph topologies a broader initial neighborhood; if anything, their lowest observed $QE_B$ occurs at a larger radius than the hexagonal minimum."

We also added Fig. 9 to the main manuscript. Panel A reports normalized $QE_B$, panel B reports observed balanced MTR and the topology-specific $QE$--MTR trade-off, and panel C reports balanced node utilization as a map-capacity check.

## 3. Related work and external baselines

### Reviewer Comment

> Related work names prior graph and GPU SOMs but does not benchmark against any. aweSOM (Ha et al., JOSS 2025), a GPU Python SOM in the same N > 10^6 range, is not mentioned.

> Add recent GPU and distributed SOM baselines. Somoclu CUDA, GigaSOM, aweSOM. aweSOM is single-node CPU/GPU with ensemble stacking, so it is a quality baseline, not a distributed competitor.

### Response

We agree that the baseline selection required clearer justification. We expanded the related-work discussion, attempted an aweSOM benchmark, and now explain why XPySOM was retained as the executable comparator. All five aweSOM attempts reached the 1800-s timeout before completing 60% of $N$ online updates on the standard speed workload. Matching FloatSOM's 10 full batch iterations would require $10N$ pointwise updates.

We now discuss Somoclu and GigaSOM as important parallel-systems references. Differences in training regime, language, hardware, and published benchmark design prevent a controlled direct comparison in the present study, and the manuscript states this limitation explicitly.

### Manuscript Amendment

In the Introduction, we added:

> "Recent tools such as aweSOM have improved single-node serial-online SOM execution through CPU/GPU acceleration and ensemble stacking [@haAweSOMCPUGPUaccelerated2025]. However, many current implementations remain constrained to single-device workloads that must fit within video random-access memory (VRAM, GPU memory), with limited support for distributed compute, out-of-core execution, and modern GPU orchestration."

In Section 2.1, we revised the related-work discussion to:

> "Open-source SOM libraries range from lightweight implementations to systems-oriented packages. MiniSom implements classical serial-online training [@vettigliJustGlowingMinisom2018], while aweSOM adds CPU/GPU acceleration and ensemble stacking for large single-node workloads [@haAweSOMCPUGPUaccelerated2025]. XPySOM instead provides GPU-accelerated batch training and is the closest executable comparator to FloatSOM's Python/GPU training pathway [@manciniXPySomHighPerformanceSelfOrganizing2020]."

> "Somoclu and GigaSOM provide additional parallel-systems context [@wittekSomocluEfficientParallel2017; @kratochvilGigaSOMjlHighperformanceClustering2020]. GigaSOM.jl, for example, trained a $32 \times 32$ SOM on 1,167,129,317 cells within a larger Julia analysis completed in under 25 minutes on an 11-node, 256-core CPU cluster [@kratochvilGigaSOMjlHighperformanceClustering2020]. These systems differ from FloatSOM in training regime, execution model, language, or hardware, so we treat their published results as systems context rather than controlled benchmarks."

In Section 4.2, we added the aweSOM benchmark attempt:

> "XPySOM was selected as the executable external baseline because it most closely matches FloatSOM's Python/GPU batch-training regime. We also attempted aweSOM on the standard speed workload, but all five runs reached the 1800-s timeout before completing 60% of $N$ online updates. Matching FloatSOM's 10 batch iterations would require $10N$ pointwise updates, so aweSOM was not included in the timed comparison."

## 4. Numbers should be embedded in the paper

### Reviewer Comment

> Quality numbers are not in the paper. Per-dataset effect sizes exist only as external table paths (Tables S4 to S11 are captions pointing at files). They cannot be read off the forest plots either.

> Put the numbers in the paper. Embed the per-dataset effect-size tables instead of external paths.

### Response

We agree. The submitted version included the tables as reproducibility artifacts but did not embed their numerical contents in the manuscript, which makes the paper harder to evaluate independently. We have replaced the path-only supplementary table captions with embedded tables for the XPySOM calibration summaries, topology effect and q-value summary, deployment effect summaries, and topology runtime summary. The manuscript now contains the numerical values needed to evaluate the corresponding figures.

### Manuscript Amendment

In the Supplementary Tables section, we replaced the path-only captions with embedded tables. For example, the revised captions now read:

> "Supplementary Table S7. Paired topology-comparison effects for hexagonal versus MST and hexagonal versus RNG across Balanced QE, Holdout QE, and Train QE."

For the deployment comparison, the revised caption now reads:

> "Supplementary Table S8. Figure 15 deployment comparison percent summary for tuned FloatSOM RNG versus untuned hexagonal XPySOM across $QE_B$, $QE_H$, and $QE_T$."

## 5. Deployment comparison separates topology, tuning, and implementation

### Reviewer Comment

> Deployment comparison (Section 7, Figure 13) is a different claim. Tuned FloatSOM RNG against untuned hexagonal XPySOM. One number, three changes: topology, tuning, implementation. The 14.5, 9.1, 22.5 percent gains cannot be credited to topology. Sections 5.1, 5.3, and 5.4 already have the pieces to separate them.

> Separate topology from tuning and implementation in the deployment comparison, or tune both systems.

### Response

We agree that Fig. 15 should not be read as attributing the full gain to topology alone. It presents an integrated deployment comparison between an untuned XPySOM run and the recommended tuned FloatSOM RNG configuration. The component effects are evaluated separately in Sections 5.1, 5.3, and 5.4, but the submitted Section 7 did not make this decomposition sufficiently clear.

We revised Section 7 and the Fig. 15 caption so that the 14.5%, 9.1%, and 22.5% improvements are described as an integrated deployment effect rather than a topology-only effect. We also point readers to the tuned-hexagonal and tuned-MST deployment figures in the Supplementary material.

### Manuscript Amendment

In Section 7, we revised the opening paragraph to:

> "Fig. 15 compares untuned hexagonal XPySOM with tuned FloatSOM RNG and therefore combines implementation, tuning, and topology effects [@manciniXPySomHighPerformanceSelfOrganizing2020]. Sections 5.1, 5.3, and 5.4 separate these components; Supplementary Figs. S7-S8 provide the corresponding tuned hexagonal and MST comparisons."

We revised the Fig. 15 caption to:

> "Figure 15. Integrated deployment comparison of untuned hexagonal XPySOM versus tuned FloatSOM RNG, combining implementation, tuning, and topology effects."

## 6. MST and RNG novelty claims

### Reviewer Comment

> Drop "novel" for the topologies. Jang et al., 2009, already in the references, put MSTs on SOMs. Frame this as the first version that scales on GPUs.

### Response

We agree that the submitted wording could imply that MSTs had not previously been associated with SOMs. Jang et al. and FlowSOM use MSTs as post hoc interpretive structures, and earlier SOM variants also discuss MST-defined training neighborhoods. We therefore no longer claim novelty for the graph object itself.

The contribution is the scalable GPU implementation and large-scale evaluation of dynamically refreshed graph neighborhoods during SOM training. We retain only the narrower, qualified statement that we have not identified prior work using dynamically refreshed RNG neighborhoods for SOM training.

### Manuscript Amendment

In the Abstract, we replaced:

> "novel topologies beyond regular lattices"

with:

> "scalable training-time graph topologies beyond regular lattices"

In Section 2.3, we added:

> "The SOM literature has also explored alternatives to fixed lattices, including dynamic maps and graph-structured neighborhoods [@vasighiDirectedBatchGrowing2017; @spanakisAMSOMAdaptiveMoving2016; @kangasVariantsSelforganizingMaps1990; @jangUseMinimalSpanning2009]. MSTs have previously appeared in SOM analyses, but prior uses generally treat the MST as an interpretive structure over an already trained map rather than as the neighborhood relation that drives SOM learning. For example, Jang et al. use MSTs for interpretation, subnode embedding, and map-shape assessment, while FlowSOM overlays an MST on trained SOM codes to visualize relationships among metaclusters [@jangUseMinimalSpanning2009; @vangassenFlowSOMUsingSelforganizing2015]. This is distinct from the training-time role used here: in FloatSOM, the MST is the operative neighborhood graph during SOM updates, is recalculated from the evolving node-weight geometry, and directly changes the update influence matrix used during learning. Thus, current deployed MST uses in SOM workflows generally configure post hoc connections on trained maps, whereas FloatSOM uses MST/RNG graphs as the training neighborhood itself. A post hoc MST overlay would also leave the quantization error of the underlying trained SOM unchanged, because $QE$ is determined by distances between samples and the fixed learned prototypes; the $QE$ improvements reported here therefore require graph-mediated training that changes those prototype locations. Earlier SOM variants also discussed MST-defined neighborhoods during learning [@kangasVariantsSelforganizingMaps1990], but these alternatives have not been assessed on large-scale datasets and do not have implementations that are either publicly available or suitable for distributed GPU computation. We have not identified prior work implementing dynamically refreshed MST or RNG topologies as scalable GPU-compatible SOM training neighborhoods. FloatSOM's topology contribution is therefore a scalable GPU-compatible implementation and large-scale quantification of refreshed graph-based SOM training, rather than a post hoc MST overlay on a conventional trained SOM."

We also added:

> "Relative Neighborhood Graphs (RNGs) [@toussaintRelativeNeighbourhoodGraph1980] are of particular interest here. To our knowledge, dynamically refreshed RNG neighborhoods have not previously been used as a SOM training topology; we return to the full rationale and implementation for RNG in Section 3.2.2."

In Section 3.2.1, we clarified that the MST topology changes training rather than only visualization:

> "The MST topology replaces fixed lattice neighborhood distance with graph hop distance on a minimum spanning tree built from current SOM nodes. The MST is therefore part of the SOM training rule: it determines which nodes receive neighborhood influence during each topology-refresh interval, rather than serving as a visualization or analysis layer after training."

In the Discussion, we replaced broad "novel graph-based topology" wording with:

> "Because these graphs define the training neighborhood, the differences reflect changed SOM updates rather than post hoc overlays."

and:

> "FloatSOM brings these topology and systems choices together in a single large-scale SOM framework."

## 7. Runtime cost of RNG recommendation

### Reviewer Comment

> Measure the runtime cost of the recommended topology. Section 8.5 recommends RNG. At grid size 64, RNG is 27x hexagonal, MST is 8x. The MST distances-to-CPU step for Kruskal is the cheap part. The all-pairs hop-distance step both share plus RNG's blocker test is the expensive part. Put a number on the RNG cost.

### Response

We agree that any recommendation of RNG must be paired with its runtime cost. Fig. 14 reports this cost: at grid size 64, the 8-GPU mean runtime was 32.54 s for hexagonal, 266.45 s for MST, and 880.83 s for RNG, corresponding to 8.19x and 27.07x the hexagonal runtime for MST and RNG, respectively. We now repeat these values at the point where RNG is recommended and make the grid-size limitation explicit.

### Manuscript Amendment

In Section 8, we added:

> "RNG is the preferred topology when $QE$ is the priority and topology overhead is acceptable, but this recommendation depends on grid size. At grid size 64, hexagonal, MST, and RNG required 32.54, 266.45, and 880.83 s on 8 GPUs, respectively. Hexagonal is therefore the throughput-oriented choice for very large grids, while MST provides a compromise when a graph topology is desired."

## 8. Multiple-comparison correction

### Reviewer Comment

> State whether the paired t-tests were corrected for multiple comparisons. About 42 per topology, 14 datasets by 3 metrics. Seed counts and intervals are already there.

### Response

We agree that the statistical reporting should state whether the paired tests were multiplicity-corrected. We revised the manuscript-facing topology table generation so that the topology-comparison q-values are Benjamini-Hochberg adjusted over exactly the family identified by the reviewer: 42 non-global dataset-level tests per topology comparison, corresponding to 14 datasets across $QE_B$, $QE_H$, and $QE_T$. This correction is computed separately for hexagonal-versus-MST and hexagonal-versus-RNG. Supplementary Table S7 reports these q-values alongside raw `p_value` entries retained for audit, and all affected forest-plot significance legends now use q-value notation rather than p-value notation.

We also audited the figure-level significance annotations against the corrected q-value convention. The correction did not change the substantive interpretation of the results: most affected annotations only changed star level while remaining significant. Only two dataset-level points crossed the significance threshold after correction, both in Fig. 7B for the holdout-QE hexagonal versus RNG comparison: `blobs` changed from p=0.0358 (`*`) to q=0.0627 (`ns`), and `iris` changed from p=0.0444 (`*`) to q=0.0745 (`ns`). We have corrected these figure annotations accordingly.

### Manuscript Amendment

In Section 4.4, we added:

> "We report Benjamini-Hochberg q-values alongside raw p-values for related dataset-level tests, and use q-values for figure markers and significance counts. Each topology contrast in Figs. 6-7 forms a separate family of 42 tests (14 datasets across $QE_B$, $QE_H$, and $QE_T$). Pooled overall tests are reported separately with raw p-values."

In the Fig. 4 caption, we added:

> "Figure significance markers use Benjamini-Hochberg adjusted q-values."

In the Fig. 6 and Fig. 7 result text, we added:

> "Supplementary Table S7 lists the corresponding per-dataset and overall hexagonal-comparison effect estimates, 95% confidence intervals, raw p-values, and Benjamini-Hochberg q-values for MST and RNG."

In the Fig. 6 and Fig. 7 captions, we added:

> "Supplementary Table S7 reports the per-dataset effect estimates, 95% confidence intervals, raw p-values, and dataset-level Benjamini-Hochberg adjusted q-values."

In the supplementary topology table caption, we revised:

> "Supplementary Table S7. Paired topology-comparison effects for hexagonal versus MST and hexagonal versus RNG across Balanced QE, Holdout QE, and Train QE."

## 9. Dead-node and node-utilization reporting

### Reviewer Comment

> Add a dead-node or node-utilization count across the three topologies. Cheap, and it bears on the preservation question.

### Response

We agree, and thank the reviewer for pointing this out. Node utilization is a useful confirmatory diagnostic for checking whether the MTR result is accompanied by broadly used map capacity rather than uneven allocation in which some nodes are effectively unused. Considering this, we added node-utilization and dead-node-fraction diagnostics to the matched topology benchmark outputs, using the same matched units as the topology comparison.

The matched tuned-profile result favored RNG rather than indicating poorer map use. Relative to tuned hexagonal maps, tuned RNG increased balanced node utilization by 0.0197 (95% CI 0.0158 to 0.0235; p=1.38e-20) and reduced balanced dead-node fraction by the same amount. We report the split-specific and balanced diagnostics separately and present node utilization as a confirmatory diagnostic alongside MTR.

### Manuscript Amendment

In Section 4.1, we added:

> "We additionally report node utilization, defined as the fraction of nodes selected as a BMU, and its complement, dead-node fraction. These post hoc diagnostics are computed on training and holdout splits and balanced as for $QE$."

In Section 5.3, we added:

> "The fixed-configuration diagnostics provide the corresponding MTR and node-use comparison (Table \ref{tab:matched_topology_diagnostics_summary}). MST and RNG both improved balanced $QE$ relative to hexagonal maps with and without tuning, but only RNG clearly lowered untuned balanced MTR."

We also embedded the paired diagnostic table:

> "Supplementary Table S12. Matched topology diagnostic paired summaries across tuned and untuned profiles."

> "Supplementary Table S13. Full matched topology diagnostic means by profile, dataset, and topology."

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

> "Efficiencies above 100\% should also be interpreted as a combined consequence of parallelism and a changed memory/data-staging regime, not as evidence of superlinear compute scaling."

# Response Letter Draft: Reviewer 1

We thank the reviewer for the careful reading and for separating the systems contribution from the quality/topology claims. We agree that the systems evidence is stronger in the submitted version than the topology-quality evidence, and the revision is structured to make that distinction explicit. Where the manuscript already contained the relevant analysis but did not state the point clearly enough, we have clarified the text. Where the reviewer identified a missing control or missing reporting detail, we have added the corresponding amendment below.

## 1. Quality and topology diagnostics

### Reviewer Comment

> The quality claim runs on QE alone. QE measures distance to the best-matching unit, not neighborhood preservation, which is the point of a SOM. No topographic error or trustworthiness, even though Forest et al., 2020 is cited and implements them.

### Response

The submitted manuscript used $QE$ as the principal deployment metric because the benchmark was designed around large-scale vector quantization and calibrated against XPySOM on that metric. We agree that $QE$ alone does not characterize SOM neighborhood ordering, and we have added Mean Tied Rank (MTR) and node-use diagnostics for the final trained maps.

MTR follows the tied-rank approach proposed by Ramos et al. for comparing SOMs with different topologies [@ramosROLELATTICEDIMENSIONALITY2018]. It ranks the second BMU according to its graph-distance shell around the first BMU, with lower values indicating closer local BMU-neighborhood ordering. We use MTR instead of raw topographic error because one-hop adjacency varies with graph connectivity and degree; prior work has also shown that topographic error depends on lattice properties and map design [@ramosROLELATTICEDIMENSIONALITY2018; @nemeStatisticalPropertiesLattices2005; @machon-gonzalezFLSOMIndividualKernel2010].

The new fixed-configuration analysis evaluates $QE$, MTR, node utilization, and dead-node fraction under matched dataset, seed, topology, and split keys. Under untuned configurations, RNG lowered balanced MTR by 2.44 tied-rank positions relative to hexagonal maps (95% CI 2.20 to 2.68; p=5.16e-57; 269/11 matched pairs favoring RNG), while MST did not clearly improve MTR. Under tuned configurations, RNG improved balanced $QE$ by 0.0515 (95% CI 0.0318 to 0.0712; p=4.94e-07) and MTR by 25.67 tied-rank positions (95% CI 24.77 to 26.56; p=2.17e-154; 280/0 matched pairs favoring RNG) relative to hexagonal maps.

Tuning reduced balanced $QE$ but increased balanced MTR in every topology. The MTR increase was substantially larger for hexagonal maps (24.34 tied-rank positions) than for MST (1.43) or RNG (1.11). Node-use diagnostics did not indicate reduced map utilization; tuned RNG increased balanced node utilization by 0.0197 relative to tuned hexagonal maps (95% CI 0.0158 to 0.0235; p=1.38e-20). The main text now includes a compact diagnostic summary, with full paired and dataset-level results in the supplement.

### Manuscript Amendment

In Section 4.1, we added:

> "We assess cross-topology local BMU-neighborhood ordering using Mean Tied Rank (MTR), following the tied-rank approach proposed for comparing SOMs with different topologies [@ramosROLELATTICEDIMENSIONALITY2018]. For each sample $x_i$, let $b_i^{(1)}$ and $b_i^{(2)}$ denote the first and second best-matching units. Non-winning units are ranked by their graph shortest-path distance from $b_i^{(1)}$, with units in the same graph-distance shell assigned their average ordinal rank. If $b_i^{(2)}$ lies in shell $S_d$ and $L_d$ non-winning units lie in closer shells, its tied rank is $\tau_i=L_d+(|S_d|+1)/2$, and $MTR=N^{-1}\sum_i \tau_i$. Lower MTR indicates that the second-best prototype is topologically closer to the winning prototype."

> "We use MTR instead of raw topographic error for cross-topology comparisons because the latter classifies the second BMU according to one-hop adjacency, which varies with graph connectivity and degree. Topographic error is therefore not directly comparable across the hexagonal, MST, and RNG graphs considered here; prior work has also shown that it depends on lattice properties and map-design choices such as size [@ramosROLELATTICEDIMENSIONALITY2018; @nemeStatisticalPropertiesLattices2005; @machon-gonzalezFLSOMIndividualKernel2010]. MTR is reported in tied-rank positions over SOM nodes rather than feature-space units or graph-edge counts."

In Section 4.3, we clarified the fixed tuned rerun design:

> "The Optuna and fixed-configuration analyses address different evaluation targets. Top-$k$ Optuna summaries estimate attainable performance within the search budget, whereas fixed-configuration reruns estimate the performance of a single deployable setting. Fixed tuned configurations use the mean selected numeric parameters and modal categorical parameters and are evaluated under matched dataset, seed, topology, and split keys."

In Section 5.3, we revised the opening sentence to:

> "Topology comparisons retain $QE$ as the primary optimized endpoint, with the Optuna hexagonal batch setting as the regular-topology baseline. Fig. 5 illustrates the neighborhood structures produced by hexagonal, MST, and RNG maps. MTR, node utilization, and dead-node fraction provide post hoc topology and capacity-use diagnostics for matched fixed tuned and untuned configurations."

We also added the tuned and untuned RNG diagnostic result:

> "Figs. 7-8 estimate attainable $QE$ under a matched Optuna budget. The fixed-configuration analysis in Table \ref{tab:matched_topology_diagnostics_summary} complements these comparisons by evaluating $QE$, MTR, and node use for deployable tuned and untuned configurations under matched dataset, seed, topology, and split keys."

> "Under untuned configurations, MST and RNG both improved balanced $QE$ relative to hexagonal maps, while only RNG clearly lowered balanced MTR (Table \ref{tab:matched_topology_diagnostics_summary}). Under tuned configurations, MST and RNG again improved balanced $QE$ relative to hexagonal maps. No significant balanced-$QE$ difference was detected between RNG and MST, while RNG lowered balanced MTR by 2.96 tied-rank positions relative to MST. RNG therefore provided the strongest joint $QE$/MTR result, while MST remained a close competitor on $QE$."

In Section 5.4, we added:

> "Tuning improved balanced $QE$ for hexagonal, MST, and RNG configurations but increased balanced MTR in each case. The increase was substantially larger for hexagonal (24.34 tied-rank positions) than for MST (1.43) or RNG (1.11), indicating a stronger local BMU-neighborhood ordering trade-off for the fixed lattice. Node-use diagnostics show that these MTR differences were not accompanied by lower map utilization."

In the Discussion, we added:

> "$QE$ and MTR characterize complementary properties of the fitted maps: $QE$ measures vector-quantization fidelity, whereas MTR measures the graph-rank separation between the first and second BMUs. RNG produced lower MTR than hexagonal maps under both untuned and tuned configurations. Tuning reduced balanced $QE$ but increased balanced MTR in all three topology families, with a substantially larger MTR increase for hexagonal maps than for MST or RNG. The $QE$-oriented tuning procedure therefore entails a stronger local-ordering trade-off for the fixed lattice than for the graph-based topologies."

## 2. Hexagonal neighborhood-radius control

### Reviewer Comment

> Add a control that shrinks the hexagonal neighborhood radius. If a tighter-radius hexagonal map closes most of the QE gap, the graph topologies are winning on looseness, not structure.

### Response

We agree that neighborhood radius is an important potential confound. The existing Optuna design partially addresses this concern because `initial_radius` was optimized independently for every topology over the same interval and search budget. The selected hexagonal radius was smaller than the selected MST and RNG radii: 1.03, 1.46, and 1.41 under full sampling, respectively, and 1.17, 1.82, and 1.77 under random sampling.

The reported MST/RNG $QE$ gains therefore do not result from comparison with a broad default-radius hexagonal map. This is not a radius-only ablation, however, and it does not isolate radius from the other optimized parameters. We now state both points explicitly in Sections 4.1 and 5.3.

### Manuscript Amendment

In Section 4.1, we added:

> "The neighborhood-radius search space was shared across topology families. In particular, `initial_radius` was an Optuna-optimized parameter for hexagonal, MST, and RNG runs, with the same search interval of 0.5 to 10.0 in each case. Thus, the hexagonal topology comparisons below use a tuned hexagonal comparator rather than a default-radius hexagonal baseline."

In Section 5.3, we added:

> "Because `initial_radius` was optimized for every topology over the same interval, each Optuna comparison incorporates a topology-specific radius selected under the same search budget. The distilled deployable configurations selected full-sampling `initial_radius` values of 1.03 for hexagonal, 1.46 for MST, and 1.41 for RNG; under random sampling, the corresponding values were 1.17, 1.82, and 1.77. The MST/RNG $QE$ gains therefore do not arise from comparison with a broader default-radius hexagonal map, although the joint optimization does not isolate radius from the other tuned hyperparameters."

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

> "Open-source SOM libraries range from lightweight implementations to systems-oriented packages. MiniSom is a compact Python implementation of the classical serial-online regime, in which the map is updated after each sampled point [@vettigliJustGlowingMinisom2018]. aweSOM accelerates serial-online training on CPUs and GPUs, supports ensemble stacking, and reports performance on single-node workloads containing up to approximately $10^8$ points [@haAweSOMCPUGPUaccelerated2025]."

> "XPySOM instead implements GPU-accelerated batch SOM training and is the closest executable comparator to FloatSOM's Python/GPU batch-training pathway [@manciniXPySomHighPerformanceSelfOrganizing2020]. Its original evaluation compared XPySOM with MiniSom, Somoclu, and TensorFlow SOM and reported order-of-magnitude speedups in that benchmark setting [@manciniXPySomHighPerformanceSelfOrganizing2020]. Somoclu and GigaSOM provide additional parallel-systems context [@wittekSomocluEfficientParallel2017; @kratochvilGigaSOMjlHighperformanceClustering2020], although differences in execution model, language, hardware, and published benchmark design prevent a controlled comparison with the Python/CUDA/Ray workflow evaluated here."

> "The training regime also limits direct comparison with aweSOM: serial-online cost scales with the number of pointwise updates, whereas batch SOM training aggregates assignments over each iteration. We attempted to run aweSOM on the standard speed workload ($10^7$ samples, 50 dimensions, and a $32 \times 32$ map), but all five runs reached the 1800-s timeout before completing 60% of $N$ online updates. Matching the 10 full batch iterations used for FloatSOM would require $10N$ pointwise updates. We therefore retain XPySOM as the executable external baseline and treat aweSOM, Somoclu, and GigaSOM as related systems context."

## 4. Numbers should be embedded in the paper

### Reviewer Comment

> Quality numbers are not in the paper. Per-dataset effect sizes exist only as external table paths (Tables S4 to S11 are captions pointing at files). They cannot be read off the forest plots either.

> Put the numbers in the paper. Embed the per-dataset effect-size tables instead of external paths.

### Response

We agree. The submitted version included the tables as reproducibility artifacts but did not embed their numerical contents in the manuscript, which makes the paper harder to evaluate independently. We have replaced the path-only supplementary table captions with embedded tables for the XPySOM calibration summaries, topology effect and q-value summary, deployment effect summaries, and topology runtime summary. The manuscript now contains the numerical values needed to evaluate the corresponding figures.

### Manuscript Amendment

In the Supplementary Tables section, we replaced the path-only captions with embedded tables. For example, the revised captions now read:

> "Supplementary Table S7. Paired topology-comparison effects for hexagonal versus MST and hexagonal versus RNG across balanced QE, holdout QE, and train QE. Rows list metric/dataset entries, including the OVERALL row. Effect columns report the paired mean percent improvement of hexagonal over the comparator topology, using hexagonal QE as the reference denominator; positive values favor hexagonal and negative values favor MST or RNG. The CI columns give the corresponding 95% paired $t$-test confidence intervals. Raw p-values are retained for audit, and q-values report Benjamini-Hochberg adjustment across the 42 dataset-level tests in each topology-comparison family (14 datasets by three QE endpoints, separately for MST and RNG); OVERALL rows are pooled summaries and retain q=NA."

For the deployment comparison, the revised caption now reads:

> "Supplementary Table S8. Figure 14 deployment comparison percent summary for tuned FloatSOM RNG versus untuned hexagonal XPySOM across $QE_B$, $QE_H$, and $QE_T$. Rows list per-dataset and `GLOBAL_OVERALL` entries with the plotted median percent change and 95% confidence interval."

## 5. Deployment comparison separates topology, tuning, and implementation

### Reviewer Comment

> Deployment comparison (Section 7, Figure 13) is a different claim. Tuned FloatSOM RNG against untuned hexagonal XPySOM. One number, three changes: topology, tuning, implementation. The 14.5, 9.1, 22.5 percent gains cannot be credited to topology. Sections 5.1, 5.3, and 5.4 already have the pieces to separate them.

> Separate topology from tuning and implementation in the deployment comparison, or tune both systems.

### Response

We agree that Fig. 14 should not be read as attributing the full gain to topology alone. It presents an integrated deployment comparison between an untuned XPySOM run and the recommended tuned FloatSOM RNG configuration. The component effects are evaluated separately in Sections 5.1, 5.3, and 5.4, but the submitted Section 7 did not make this decomposition sufficiently clear.

We revised Section 7 and the Fig. 14 caption so that the 14.5%, 9.1%, and 22.5% improvements are described as an integrated deployment effect rather than a topology-only effect. We also point readers to the tuned-hexagonal and tuned-MST deployment figures in the Supplementary material.

### Manuscript Amendment

In Section 7, we revised the opening paragraph to:

> "Fig. 14 is an integrated deployment comparison rather than a topology-only attribution. It compares the untuned hexagonal XPySOM workflow against the recommended tuned FloatSOM RNG workflow, so the reported difference includes implementation, hyperparameter tuning, and topology choice [@manciniXPySomHighPerformanceSelfOrganizing2020]. The components are separated in the preceding analyses: Section 5.1 calibrates FloatSOM and XPySOM under matched hexagonal settings, Section 5.3 compares hexagonal, MST, and RNG inside FloatSOM under the same Optuna budget, and Section 5.4 evaluates tuned configurations against the untuned reference. Supplementary Figures S7-S8 provide the corresponding tuned hexagonal and tuned MST deployment comparisons against untuned hexagonal XPySOM."

We revised the Fig. 14 caption to:

> "Figure 14. Integrated deployment comparison of untuned hexagonal XPySOM versus tuned FloatSOM RNG. The comparison intentionally combines implementation, hyperparameter tuning, and topology choice and should not be interpreted as attributing the full difference to topology alone."

## 6. MST and RNG novelty claims

### Reviewer Comment

> Drop "novel" for the topologies. Jang et al., 2009, already in the references, put MSTs on SOMs. Frame this as the first version that scales on GPUs.

### Response

We thank the reviewer for pointing out the ambiguity in the submitted novelty wording. We agree that the paper should not imply that MSTs have never been used in connection with SOMs. Instead, the novelty is the role of the graph in the training algorithm. Prior MST-on-SOM uses, including Jang et al. and FlowSOM, place or overlay an MST on an already trained SOM for interpretation, visualization, metacluster relationships, subnode embedding, or map-shape assessment. Thus, current deployed MST uses in SOM workflows generally configure post hoc connections on trained maps, whereas FloatSOM uses MST/RNG graphs as the training neighborhood itself. A post hoc MST overlay would also leave the quantization error of the underlying trained SOM unchanged, because $QE$ is determined by distances between samples and the fixed learned prototypes. The $QE$ improvements reported in our benchmarks therefore require graph-mediated training that changes those prototype locations. That use is not equivalent to the FloatSOM topology path: in our implementation, the MST is the operative neighborhood graph during training, is recalculated from the evolving node-weight geometry, and directly changes the SOM update influence matrix.

We revised the manuscript to make this distinction explicit. Earlier SOM variants discussed MST-defined neighborhoods during learning, so we do not claim that the graph object alone is new. The contribution is the scalable GPU-compatible refreshed MST training topology and its large-scale quantitative evaluation. We have not identified prior work implementing dynamically refreshed MST or RNG topologies as scalable GPU-compatible SOM training neighborhoods. For RNG, we retain a narrower and qualified novelty claim: we have not identified prior work applying dynamically refreshed Relative Neighborhood Graphs as the neighborhood topology in SOM training.

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

> "Because MST and RNG are used as training-time neighborhoods, these differences reflect changes to the SOM update dynamics rather than post hoc graph summaries placed over an unchanged trained map."

and:

> "FloatSOM therefore provides a unified large-scale SOM framework that combines scalable training-time graph topology support with sampling options, optimized hyperparameters, and distributed out-of-memory GPU execution."

## 7. Runtime cost of RNG recommendation

### Reviewer Comment

> Measure the runtime cost of the recommended topology. Section 8.5 recommends RNG. At grid size 64, RNG is 27x hexagonal, MST is 8x. The MST distances-to-CPU step for Kruskal is the cheap part. The all-pairs hop-distance step both share plus RNG's blocker test is the expensive part. Put a number on the RNG cost.

### Response

We agree that any recommendation of RNG must be paired with its runtime cost. Fig. 13 reports this cost: at grid size 64, the 8-GPU mean runtime was 32.54 s for hexagonal, 266.45 s for MST, and 880.83 s for RNG, corresponding to 8.19x and 27.07x the hexagonal runtime for MST and RNG, respectively. We now repeat these values at the point where RNG is recommended and make the grid-size limitation explicit.

### Manuscript Amendment

In Section 8, we added:

> "Overall, these results support a practical deployment strategy that uses RNG with topology-aware tuned configurations when $QE$ is the priority and topology-construction overhead is acceptable. That recommendation is conditional on grid size. In the grid-size scaling benchmark, the largest tested grid size (64) required 32.54 s for hexagonal, 266.45 s for MST, and 880.83 s for RNG on 8 GPUs, making MST 8.19x and RNG 27.07x slower than hexagonal at that point. For workloads dominated by very large grids, hexagonal remains the appropriate throughput-oriented default. MST is a practical compromise when graph-based topology is desired but RNG's blocker-test cost is too high."

## 8. Multiple-comparison correction

### Reviewer Comment

> State whether the paired t-tests were corrected for multiple comparisons. About 42 per topology, 14 datasets by 3 metrics. Seed counts and intervals are already there.

### Response

We agree that the statistical reporting should state whether the paired tests were multiplicity-corrected. We revised the manuscript-facing topology table generation so that the topology-comparison q-values are Benjamini-Hochberg adjusted over exactly the family identified by the reviewer: 42 non-global dataset-level tests per topology comparison, corresponding to 14 datasets across $QE_B$, $QE_H$, and $QE_T$. This correction is computed separately for hexagonal-versus-MST and hexagonal-versus-RNG. Supplementary Table S7 reports these q-values alongside raw `p_value` entries retained for audit, and all affected forest-plot significance legends now use q-value notation rather than p-value notation.

We also audited the figure-level significance annotations against the corrected q-value convention. The correction did not change the substantive interpretation of the results: most affected annotations only changed star level while remaining significant. Only two dataset-level points crossed the significance threshold after correction, both in Fig. 7B for the holdout-QE hexagonal versus RNG comparison: `blobs` changed from p=0.0358 (`*`) to q=0.0627 (`ns`), and `iris` changed from p=0.0444 (`*`) to q=0.0745 (`ns`). We have corrected these figure annotations accordingly.

### Manuscript Amendment

In Section 4.4, we added:

> "For dataset-level families of related paired tests, we compute Benjamini-Hochberg adjusted q-values in addition to raw paired $t$-test p-values. Dataset-level figure significance markers and significance counts use these adjusted q-values. For each topology contrast in Figs. 6-7, adjustment is applied across 42 dataset-level tests: 14 datasets evaluated on $QE_B$, $QE_H$, and $QE_T$. Hexagonal-versus-MST and hexagonal-versus-RNG constitute separate adjustment families. Global pooled tests are reported as overall summaries and are not included in these families; they therefore retain raw p-values only."

In the Fig. 4 caption, we added:

> "Figure significance markers use Benjamini-Hochberg adjusted q-values."

In the Fig. 6 and Fig. 7 result text, we added:

> "Supplementary Table S7 lists the corresponding per-dataset and overall hexagonal-comparison effect estimates, 95% confidence intervals, raw p-values, and Benjamini-Hochberg q-values for MST and RNG."

In the Fig. 6 and Fig. 7 captions, we added:

> "Supplementary Table S7 reports the per-dataset effect estimates, 95% confidence intervals, raw p-values, and dataset-level Benjamini-Hochberg adjusted q-values."

In the supplementary topology table caption, we revised:

> "Supplementary Table S7. Paired topology-comparison effects for hexagonal versus MST and hexagonal versus RNG across balanced QE, holdout QE, and train QE. Rows list metric/dataset entries, including the OVERALL row. Effect columns report the paired mean percent improvement of hexagonal over the comparator topology, using hexagonal QE as the reference denominator; positive values favor hexagonal and negative values favor MST or RNG. The CI columns give the corresponding 95% paired $t$-test confidence intervals. Raw p-values are retained for audit, and q-values report Benjamini-Hochberg adjustment across the 42 dataset-level tests in each topology-comparison family (14 datasets by three QE endpoints, separately for MST and RNG); OVERALL rows are pooled summaries and retain q=NA."

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

> "Topology comparisons retain $QE$ as the primary optimized endpoint, with the Optuna hexagonal batch setting as the regular-topology baseline. Fig. 5 illustrates the neighborhood structures produced by hexagonal, MST, and RNG maps. MTR, node utilization, and dead-node fraction provide post hoc topology and capacity-use diagnostics for matched fixed tuned and untuned configurations."

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

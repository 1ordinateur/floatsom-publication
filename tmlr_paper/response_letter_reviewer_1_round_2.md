# Response Letter: Reviewer 1, Round 2

We thank the reviewer for the careful follow-up and for identifying where the revised manuscript still points to supporting analyses without stating their combined implication directly. We address the requested clarification below.

## 1. Explicit decomposition of the deployment QE gains

### Reviewer Comment

> In section 5; You have all three numbers on global balanced QE. Implementation alone (Table S6) gives ~0 with p=0.0923. Tuning alone (Table S9) gives 13.04%. Full stack (Table S8) gives 14.49%. Topology accounts for 1.4 of those 14.5 points. Train QE adds 2.3, holdout 0.9, which matches the direct within-FloatSOM contrast in S7 (RNG vs hexagonal at −1.297%).
>
> Pointing to three sections that contain these numbers isn't the same as reporting the split. One sentence in Section 7 will do.

### Response

We now report the split directly in Section 7. The matched untuned hexagonal implementation comparison showed no detectable global balanced-$QE$ gain, with a point estimate effectively equal to zero (median improvement 0.000016%, $p=0.0923$; Supplementary Table S6). On the common untuned-XPySOM reference scale, tuning accounted for 13.04 percentage points of the full tuned-RNG stack's 14.49% gain, and topology accounted for the remaining 1.44 points. The topology contribution was 0.88 points for holdout $QE$ and 2.34 points for train $QE$ (Supplementary Tables S8--S9), consistent with the direct 1.297% tuned RNG-over-hexagonal balanced-$QE$ advantage in Supplementary Table S7.

### Summary of Manuscript Changes

We added one sentence to Section 7 that reports the implementation, tuning, and topology split explicitly. It states the near-zero untuned implementation result, the 13.04-point tuning contribution, the 14.49% full tuned-RNG gain, and the resulting 1.44-point balanced-$QE$ topology contribution together with its 0.88-point holdout and 2.34-point train components.

## 2. Comparison at each topology's minimum QE radius

### Reviewer Comment

> Section 2 - Table S15 compares topologies within each radius. I asked whether hexagonal at its own best radius closes the gap against RNG at its own best radius. That is not what S15 shows.
>
> Your comment mentions optima at r=1.5 for graph topologies and r=1.027 for hexagonal, but that number isn't in the manuscript and I can't reconstruct it from Fig. 9A plus S15.
>
> What S15 does show: no detectable topology difference at the two smallest radii. MST vs hexagonal comes in at p=0.480 and p=0.875, RNG vs hexagonal at p=0.119 and p=0.396. Separation appears above r=1.027. Your main-text sentence is true for point estimates but does not say anything about this.
>
> Add per-topology minimum QE_B across the sweep with a paired test between those minima, and state that separation vanishes at r≤0.75.

### Response

We added Supplementary Table S15, which reports the exact dataset-balanced normalized $QE_B$ value and 95% confidence interval for hexagonal, MST, and RNG at each of the seven tested radii. These are the numerical values underlying Fig. 9A, so the per-topology minima can now be reconstructed directly. The table shows that the fixed-sweep minimum occurred at $r=0.75$ for hexagonal and $r=1.5$ for MST and RNG. We also clarified that $r=1.0266$ is the independently Optuna-selected hexagonal anchor, not the hexagonal minimum within this fixed sweep.

Supplementary Table S16 reports the cross-topology comparisons at each shared radius and shows that separation vanished at $r\leq0.75$, with no significant MST--hexagonal or RNG--hexagonal difference at either of the two smallest radii. Supplementary Table S17 reports the requested paired comparisons between configurations at their topology-specific minima. MST had 0.98% lower $QE_B$ than hexagonal (ratio 0.9902, 95% CI [0.9747, 1.0060], $p=0.2028$, $q=0.3042$), and RNG had 1.35% lower $QE_B$ than hexagonal (ratio 0.9865, 95% CI [0.9689, 1.0043], $p=0.1252$, $q=0.3042$); neither difference was significant. RNG and MST also did not differ significantly (ratio 0.9962, 95% CI [0.9834, 1.0092], $p=q=0.5377$).

### Summary of Manuscript Changes

We added the exact normalized $QE_B$ radius-response values to Supplementary Table S15, moved the shared-radius contrasts to Supplementary Table S16, and moved the topology-specific minimum comparisons to Supplementary Table S17. We also revised the Methods, Section 5.3.4, and the Fig. 9 caption to distinguish the selected hexagonal anchor from the fixed-sweep minimum and to identify where each numerical result is reported.

## 3. MTR permutation-null analysis and claim scope

### Reviewer Comment

> Your argument against topographic error is correct. But MTR can't fairly compare a fixed lattice to a graph rebuilt from node weights. MST and RNG edges come from pairwise distances between weight vectors, so the second BMU is already determined by the graph structure. That alignment is free in the graph arms but must be trained in hexagonal, which is exactly what's being measured. The decaying refresh schedule tries to attempt it but slightly. It doesn't fix it.
>
> S13 shows why: untuned to tuned, hexagonal MTR_H jumps 3.51→27.48 (blobs) and 3.56→31.04 (circles). MST moves 5.23→6.78 and 7.21→9.95. RNG barely changes. A metric that's essentially flat under training in two of three arms isn't discriminating among them.
>
> Untuned, pooled MST vs hexagonal contrast is −0.20, favoring hexagonal.
>
> There's also a floor from shell size. Hexagonal has ~6 nodes at hop 1 versus MST's ~2, so perfect-ordering floors are ~3.5 and ~1.5 respectively. Converting back, tuned hexagonal at 27 sits about 3 hops out, and tuned MST at 6.8 sits between 3–4. MTR and hop count order these differently, and no null is reported.
>
> RNG vs MST does survive: RNG has more edges (higher floor) yet scores lower (4.91 vs 7.87). What doesn't survive is the claim that graphs preserve local ordering better than hexagonal. Hence, earlier weakness that pointed at this is not answered yet. that lower QE could mean the map is drifting toward a loose vector quantizer. MTR as currently constructed can't rule that out.
>
> This matters because tuned RNG and MST show no QE_B difference (p=0.126, point estimate slightly favoring MST). So your recommendation now rests mainly on MTR.
>
> Possible ways to fix (1) Report MTR against a per-map null: permute which weight vector sits on each graph node, give the observed-to-null ratio. (2) Or demote MTR to descriptive only, keep it for RNG vs MST comparison, and restate the recommendation on QE and stability. Then Discussion sentence claiming RNG beats "both MST and hexagonal" on MTR should change accordingly.

### Response

We implemented the first suggested analysis. For every trained map and evaluation split, we held the final adjacency and observed first--second BMU identity pairs fixed, uniformly permuted prototype identities among the 100 graph nodes 1,000 times, and recomputed MTR. The MST and RNG graphs were not rebuilt after permutation. For each map, we calculated the observed-to-null ratio as the observed MTR divided by the mean permutation-null MTR.

The null MTR is nearly exactly 50 as once the first BMU is fixed, random reassignment makes the second BMU equally likely to occupy any of the other 99 nodes. MTR's shell tied ranks, counted with their shell multiplicities, partition ordinal ranks 1 through 99. Their mean is therefore $(1+99)/2=50$, irrespective of node degree or shell sizes. The empirical profile--topology null means ranged from 49.999 to 50.012, confirming this result.

The mean observed-to-null ratios were 0.123, 0.127, and 0.075 for untuned hexagonal, MST, and RNG, respectively, and 0.614, 0.160, and 0.098 for the tuned profiles. Within the two weight-derived graph families, RNG had a lower null-relative MTR ratio than MST by 0.0520 in the untuned profile (95% CI [0.0468, 0.0573], $p=1.33\times10^{-54}$) and by 0.0614 in the tuned profile (95% CI [0.0561, 0.0668], $p=1.63\times10^{-65}$). Supplementary Table S19 reports all six ratios and all matched topology contrasts.

Because the null values were effectively identical across topology families, the ratio did not provide a topology-specific normalization. We therefore report graph--hexagonal MTR differences descriptively and use MTR inferentially only to compare MST and RNG.

Within this comparison, RNG showed significantly better local topological ordering than MST, as measured by null-relative MTR, in both tuned and untuned profiles. We removed the claim that RNG beats “both MST and hexagonal” in topology preservation. For moderate map sizes, the practical recommendation of RNG as the starting topology is now based primarily on its overall $QE$, stability, and node-utilization results, with the RNG--MST MTR comparison providing additional support.

### Summary of Manuscript Changes

We added a description of the permutation-null analysis to the main Methods and placed the complete per-map fixed-adjacency permutation procedure and exact $P/2=50$ derivation in Supplementary Methods S2. We reported the six observed-to-null ratios and matched contrasts in Supplementary Table S19, treated graph--hexagonal MTR differences as descriptive, and used MTR inferentially only for the RNG--MST comparison. The practical recommendation now reflects the combined $QE$, stability, and node-utilization evidence, with RNG's significantly better MTR-defined local ordering than MST providing additional support.

## 4. aweSOM quality baseline

### Reviewer Comment

> aweSOM - A serial-online implementation was never going to finish $10^7$ samples in 1800s. I asked for aweSOM as a quality baseline. Your datasets max out at 581k, and most are under 2k. Iris, wine, digits, breast_cancer, and olivetti_faces would run in minutes each. These could be run. Ensemble stacking remains untested. The Somoclu and GigaSOM justifications stand.

### Response

We now evaluate aweSOM as a quality baseline alongside the distributed-systems comparators. As recommended, we ran aweSOM on Iris, Wine, Digits, Breast Cancer, and Olivetti Faces and compared it with untuned hexagonal FloatSOM using 10 shared seeds, matched preprocessing and train--holdout partitions, 100-node maps, and 50 training passes. Across the five datasets, FloatSOM reduced balanced, holdout, and train quantization error by an average of 36.77%, 28.52%, and 45.11%, respectively. The complete dataset-level effects, confidence intervals, and multiplicity-adjusted tests are reported in Supplementary Table S18.

Given the large difference between aweSOM's serial-online training and the batch training used by FloatSOM and XPySOM, together with the large-workload timeout documented in the speed-benchmark protocol (Section 4.2), we used XPySOM as the external implementation comparator in subsequent analyses.

### Summary of Manuscript Changes

We added the aweSOM quality protocol to Section 4.1.4, the average quality results and subsequent-comparator rationale to Section 5.1.2, and the complete dataset-level comparison to Supplementary Table S18. We also clarified that aweSOM's large serial-online speed workload was separate from its evaluation as a quality baseline.

## 5. Fixed node count across topology comparisons

### Reviewer Comment

> On the Map size, you state it for XPySOM calibration and the speed benchmark (both 32×32), but not for the Optuna campaign or the 280-unit diagnostic. QE falls monotonically with node count. If map size was in your search space and landed differently per topology, you have the same confound I raised for radius. From S13, tuned iris Util_T = 0.7235 on 105 training samples implies P ≈ 144, not 1024. State map size everywhere. If it varied by topology, control for it.

### Response

We agree that the node count should have been stated explicitly. Map size was fixed rather than optimized and did not vary among topology families. All Optuna quality runs used $P=100$ nodes, implemented as a $10\times10$ hexagonal lattice and as 100-node MST and RNG graphs. The subsequent 280 dataset--seed topology diagnostic retained the same $P=100$ node count for every topology and for both tuned and untuned profiles. Consequently, node count was controlled across the topology comparisons.

The Iris value $Util_T=0.7235$ is the fraction of the 100 nodes selected as BMUs, averaged across seeds; it corresponds to an average of 72.35 occupied nodes. It does not assume that each of the 105 training observations selects a distinct node.

### Summary of Manuscript Changes

We now state in Sections 4.1 and 4.3.1 that the Optuna campaign and fixed-configuration topology diagnostic used a fixed node count of $P=100$ across all topology families and that map size was not part of the Optuna search space.

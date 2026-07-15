# Response Letter Draft: Reviewer 3

We sincerely thank the reviewer for the careful, constructive, and encouraging assessment of our work. We especially appreciate the reviewer highlighting the engineering contribution, the rigor of the empirical evaluation, and the practical relevance of modernizing SOM training for billion-scale workloads. The requested changes were thoughtful and helped us improve both the explanatory depth and the presentation of the manuscript. In response, we have expanded the systems accounting, strengthened the geometric explanation of the topology choices, added a real high-dimensional example to the representative-topology figure, and improved the legibility and auditability of the quantitative results.

## 1. Geometric intuition and expanded Figure 5

### Reviewer Comment

> First, I suggest expanding the discussion section to include a deeper geometric intuition regarding why the Relative Neighborhood Graph topology outperforms fixed grids on real-world datasets. Providing visual examples of how these dynamic graphs adapt to complex low-dimensional manifolds would be highly beneficial.

### Response

We agree that the topology motivation benefits from a more concrete geometric account and from an example beyond a two-dimensional synthetic dataset.

A fixed hexagonal lattice assigns neighborhood relationships before learning and independently of the evolving prototype geometry. Movement of one prototype can therefore influence lattice neighbors that are not locally related in learned data space, potentially displacing otherwise useful prototypes and increasing dead nodes. MST minimizes this coupling and gives prototypes greater freedom to redistribute along irregular or elongated structures. That freedom is also MST's limitation: a tree cannot retain several locally appropriate connections where a dense region is better described by a mesh-like neighborhood.

RNG occupies an intermediate position. It is less free than MST because nodes may influence several neighbors, but those additional connections are supported by the evolving prototype geometry rather than uniformly imposed beforehand. RNG can remain sparse where appropriate and form multiple connections in concentrated regions, influencing nodes that should be influenced without the fixed lattice's uniform coupling. We present the observed $QE$, MTR, node-utilization, and dead-node results as evidence consistent with this interpretation, rather than as proof of a unique causal mechanism.

We expanded Fig. 5 from one row to a 2x3 layout. Panels A--C show circles in native 2D, and panels D--F show KDD Cup 99 trained on standardized 41-dimensional data and displayed using a shared two-dimensional PCA projection. In the KDD Cup 99 display, the hexagonal topology contains several apparently nonlocal connections spanning separated regions of the projected prototype distribution, whereas the MST and RNG connections more closely follow its local geometry. Because the projection can distort graph geometry in the original feature space, we treat this pattern as an illustration rather than direct evidence of original-space nonlocality. Quantitative node-utilization and dead-node evidence remains in Fig. 9 and the matched diagnostic tables.

### Manuscript Amendment

In Section 3.2, we added:

> "These alternatives change which prototypes are coupled during learning. A fixed hexagonal lattice assigns the same predetermined neighborhood pattern independently of the learned data-space geometry. Moving one prototype can therefore influence lattice neighbors that are not locally related in data space, potentially displacing otherwise useful prototypes. An MST minimizes this coupling and gives prototypes greater freedom to redistribute along irregular or elongated structures, but its tree constraint can omit additional locally appropriate connections in dense regions. RNG is less free than MST because a node can influence several neighbors; unlike the fixed lattice, however, those additional connections are supported by the evolving prototype geometry. It can therefore remain sparse where appropriate and form a more mesh-like neighborhood in concentrated regions."

In the topology Results, we expanded Fig. 5 and added:

> "Figure 5. Representative node and connection overlays for matched 100-node hexagonal, MST, and RNG SOMs. A--C: circles in native 2D. D--F: KDD Cup 99, trained on standardized 41-dimensional observations and displayed using one shared 2D PCA projection fitted to the observations and applied to all three sets of prototypes. Training used seed 42, full sampling, random initialization, 50 iterations, and the untuned XPySOM-like settings used at this point in the manuscript: initial learning rate 0.5, initial radius 5 with exponential decay, momentum disabled, and XPySOM-compatible normalization. Grey observation clouds show a deterministic maximum of 30,000 points; black lines are topology connections and red markers are SOM nodes. PCA can distort graph geometry in the original 41-dimensional space."

> "In the KDD Cup 99 projection (Fig. 5D--F), the hexagonal topology contains several apparently nonlocal connections spanning separated regions of the projected prototype distribution, whereas the MST and RNG connections more closely follow its local geometry. This pattern illustrates how predetermined lattice neighbors can couple the updates of prototypes that are not locally adjacent in the displayed data structure, restricting how independently the nodes can redistribute. Because the display is a two-dimensional PCA projection of the 41-dimensional training space, this visual pattern is illustrative rather than direct evidence of nonlocality in the original space."

In the Discussion, we added:

> "A geometric interpretation consistent with these results is that a fixed lattice imposes uniform, predetermined coupling: when a prototype moves, it can unnecessarily influence lattice neighbors that are unrelated in the learned data space, potentially displacing useful prototypes and increasing dead nodes. The apparently nonlocal hexagonal connections in the KDD Cup 99 projection (Fig. 5D) provide a visual illustration of this coupling and the associated restriction on independent prototype redistribution, subject to the distortions inherent in the two-dimensional PCA display. MST minimizes such coupling, allowing prototypes to redistribute more freely along irregular or elongated data structures. This freedom is also its limitation, because a tree cannot retain multiple locally appropriate connections where a concentrated region is better represented by a mesh. RNG occupies an intermediate position. Its nodes can influence several neighbors, but those connections arise from the evolving prototype geometry rather than being imposed before training; RNG can remain sparse where appropriate and form multiple connections in dense regions. The node-utilization and dead-node results in Fig. 9, together with $QE$ and MTR, are consistent with this account, but they do not identify a unique causal mechanism [@kohonenEssentialsSelforganizingMap2013; @kangasVariantsSelforganizingMaps1990; @toussaintRelativeNeighbourhoodGraph1980]."

## 2. Distributed memory, communication, and large-grid cost

### Reviewer Comment

> Second, the authors should provide more quantitative details regarding the memory and communication overhead required to synchronize complex topology states across multiple GPUs, especially for large grid sizes. This would help clarify the sharp runtime increases observed in the grid-size scaling experiments.

### Response

We thank the reviewer for identifying where the original systems explanation needed more quantitative detail. FloatSOM does not communicate observations between workers during an iteration. Each worker processes its own shard in bounded chunks and accumulates a node-by-feature update numerator and a node-wise normalization denominator. Those two arrays are synchronized once per iteration by NCCL all-reduce; the updated prototypes then remain resident and identical on every worker.

For $P$ nodes and $d$ features stored as float32, the two accumulators contain $Pd+P$ values. Their logical all-reduce payload is therefore $4P(d+1)$ bytes per worker per iteration, plus a 4-byte sample-count reduction. For a ring all-reduce over $G$ workers, each worker sends and receives approximately $2(G-1)/G$ times that payload. This communication term is independent of the number of observations $N$: increasing $N$ increases local BMU/update work and data staging, whereas increasing $P$ increases both the synchronized update and the graph-topology work.

The memory explanation now separates observation and topology terms. With chunks of at most $C$ rows, the dominant bounded worker arrays scale as $O(Cd+Pd)$ rather than requiring the full $O(Nd)$ observations in GPU memory. Graph distances and influence structures add $P^2$ terms; FloatSOM tiles or spills these structures when required. The measured grid-size results show the practical consequence. At grid size 64 on 8 GPUs, hexagonal, MST, and RNG required 32.54, 266.45, and 880.83 s, respectively. Thus, the distributed pathway scales much more favorably in sample count than in node count, especially for RNG.

### Manuscript Amendment

In Section 3.3.1, after the distributed-update equations, we added:

> "For $P$ nodes and $d$ features in float32, the two update accumulators contain $P d+P$ values, so their logical all-reduce payload is $4P(d+1)$ bytes per worker per iteration, plus one 4-byte sample count. A ring all-reduce sends and receives approximately $2(G-1)/G$ times that payload per worker for $G$ workers. This communication is independent of the number of observations $N$; increasing $N$ instead increases worker-local computation and data staging. With observation chunks of at most $C$ rows, the dominant bounded worker arrays scale as $O(Cd+Pd)$ rather than requiring the full $O(Nd)$ dataset in GPU memory. Graph-topology distance and influence structures add terms that scale with $P^2$ and can be tiled or spilled when necessary, which explains why increasing node count is a less favorable scaling regime than increasing sample count."

In Section 6.2, we clarified the distinction between sample and grid-size scaling:

> "Section 6.2 primarily concerns sample scaling. In that setting, the topologies exhibit similar scaling characteristics: as sample count increases, the GPU-count response and efficiency curves have similar qualitative shapes for RNG (Fig. 13B,E) and for the corresponding hexagonal and MST outputs (Fig. S6). Conversely, when the number of SOM nodes is increased (grid-size scaling), the topologies differ substantially. That grid-size regime is analyzed in Section 6.3, where MST and RNG take 8.19x and 27.07x the hexagonal runtime, respectively, at the largest tested grid size."

In Section 6.3, we added the absolute grid-size runtimes:

> "However, when the grid itself is enlarged in Fig. 14C, topology-dependent runtime differences become readily evident. At the largest tested grid size (grid size 64), the 8-GPU mean runtimes are 32.54 s (0.54 min) for hexagonal, 266.45 s (4.44 min) for MST, and 880.83 s (14.68 min) for RNG, corresponding to 8-GPU MST and RNG runtimes that are 8.19x and 27.07x the hexagonal runtime, respectively."

## 3. Forest-plot legibility

### Reviewer Comment

> Finally, improving the legibility of the forest plots is necessary, as the data points and error bars are currently difficult to read without zooming in significantly.

### Response

We thank the reviewer for drawing our attention to this presentation issue. We increased the confidence-interval line widths and point-marker sizes by 200% across all forest plots to improve legibility at manuscript scale.

We also made the quantitative results easier to inspect without relying on the plotted markers alone. The numerical supplementary tables that were previously referenced only by external paths are now embedded in the manuscript. They report effect estimates, confidence intervals, raw p-values, adjusted q-values where applicable, sample counts, and directional summaries. Dataset-level significance annotations now use Benjamini--Hochberg adjusted q-values, and the direct MST--RNG comparison has been moved into the main text as Fig. 8.

### Manuscript Amendment

We revised all forest-plot assets by increasing their confidence-interval line widths and point-marker sizes by 200%. We also revised the statistical reporting in Section 4.4:

> "We report Benjamini-Hochberg q-values alongside raw p-values for related dataset-level tests, and use q-values for figure markers and significance counts. Each topology contrast in Figs. 6-7 forms a separate family of 42 tests (14 datasets across $QE_B$, $QE_H$, and $QE_T$). Pooled overall tests are reported separately with raw p-values."

The forest-plot captions now identify the intervals and embedded numerical results; for example, the revised Fig. 6 caption states:

> "Figure 6. Hexagonal versus MST topology on $QE$ metrics under full sampling only. Panels A-C report paired full sampling only $QE$ effects for $QE_B$, $QE_H$, and $QE_T$ across the available full sampling datasets. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect. Supplementary Table S7 reports the per-dataset effect estimates, 95% confidence intervals, raw p-values, and dataset-level Benjamini-Hochberg adjusted q-values."

We also added the direct graph-topology comparison to the main Results:

> "Fig. 8 directly compares MST and RNG under full sampling. The two graph topologies are close on $QE$, and neither dominates across all three endpoints."

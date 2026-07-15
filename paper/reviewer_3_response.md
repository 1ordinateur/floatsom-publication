# Response Letter Draft: Reviewer 3

We thank the reviewer for the constructive suggestions. We have expanded the systems accounting, strengthened the geometric explanation of the topology choices, enlarged the qualitative topology figure with a real high-dimensional dataset, and made the quantitative forest-plot results easier to evaluate directly from the manuscript.

## 1. Distributed memory, communication, and large-grid cost

### Reviewer Comment

> **Paraphrased:** The reviewer requested a quantitative account of the data communicated between workers, the memory scaling of the distributed implementation, and the practical cost of graph topologies as the SOM grid becomes large.

### Response

We clarified that FloatSOM does not communicate observations between workers during an iteration. Each worker processes its own shard in bounded chunks and accumulates a node-by-feature update numerator and a node-wise normalization denominator. Those two arrays are synchronized once per iteration by NCCL all-reduce; the updated prototypes then remain resident and identical on every worker.

For $P$ nodes and $d$ features stored as float32, the two accumulators contain $Pd+P$ values. Their logical all-reduce payload is therefore $4P(d+1)$ bytes per worker per iteration, plus a 4-byte sample-count reduction. For a ring all-reduce over $G$ workers, each worker sends and receives approximately $2(G-1)/G$ times that payload. The communication term is independent of the number of observations $N$: increasing $N$ increases local BMU/update work and data staging, whereas increasing $P$ increases both the synchronized update and the graph-topology work.

The memory explanation now separates observation and topology terms. With chunks of at most $C$ rows, the dominant bounded worker arrays scale as $O(Cd+Pd)$ rather than requiring the full $O(Nd)$ observations in GPU memory. Graph distances and influence structures add $P^2$ terms; FloatSOM tiles or spills these structures when required. The measured grid-size results show the practical consequence. At grid size 64 on 8 GPUs, hexagonal, MST, and RNG required 32.54, 266.45, and 880.83 s, respectively. Thus the distributed pathway scales much more favorably in sample count than in node count, especially for RNG.

### Manuscript Amendment

In Section 3.3.1, after the distributed-update equations, we added the explicit float32 payload and bounded-memory expressions. Sections 6.2--6.3 and the Discussion retain the grid-64 absolute runtimes and the 8.19x MST and 27.07x RNG penalties relative to hexagonal.

## 2. Geometric intuition and expanded Figure 5

### Reviewer Comment

> **Paraphrased:** The reviewer requested clearer geometric intuition for the hexagonal, MST, and RNG training topologies and a larger representative-topology figure that includes a real, high-dimensional dataset rather than only a two-dimensional synthetic example.

### Response

We agree that the topology motivation benefits from a concrete geometric account and from an example beyond a two-dimensional synthetic dataset.

A fixed hexagonal lattice assigns neighborhood relationships before learning and independently of the evolving prototype geometry. Movement of one prototype can therefore influence lattice neighbors that are not locally related in learned data space, potentially displacing otherwise useful prototypes and increasing dead nodes. MST minimizes this coupling and gives prototypes greater freedom to redistribute along irregular or elongated structures. That freedom is also MST's limitation: a tree cannot retain several locally appropriate connections where a dense region is better described by a mesh-like neighborhood.

RNG occupies an intermediate position. It is less free than MST because nodes may influence several neighbors, but those additional connections are supported by the evolving prototype geometry rather than uniformly imposed beforehand. RNG can remain sparse where appropriate and form multiple connections in concentrated regions, influencing nodes that should be influenced without the fixed lattice's uniform coupling. Quantitative utilization and dead-node evidence is reported in Fig. 9 and the matched diagnostic tables; Fig. 5 remains a qualitative illustration of prototype deployment and neighborhood geometry.

We expanded Fig. 5 from one row to a 2x3 layout. Panels A--C show circles in native 2D, and panels D--F show KDD Cup 99 trained on standardized 41-dimensional data and displayed using a shared two-dimensional PCA projection. In the KDD Cup 99 display, the hexagonal topology contains several apparently nonlocal connections spanning separated regions of the projected prototype distribution, whereas the MST and RNG connections more closely follow its local geometry. The caption states that this projection can distort graph geometry in the original feature space, so we present the pattern as illustrative rather than direct evidence of original-space nonlocality.

The visual encoding remains deliberately qualitative: grey observations, black topology connections, and uniformly red SOM nodes, without active/dead styling or utilization annotations.

### Manuscript Amendment

We added the fixed-lattice/MST/RNG coupling explanation to Section 3.2 and the Discussion. In the topology Results, we now describe the apparently nonlocal hexagonal connections in the KDD Cup 99 projection and explain how they visually illustrate predetermined lattice coupling and restricted independent prototype redistribution, subject to the PCA caveat. The Discussion explicitly connects Fig. 5D to this geometric interpretation while retaining Fig. 9 and the matched tables as the quantitative utilization and dead-node evidence. We replaced Fig. 5 and synchronized its caption with the revised manuscript.

## 3. Forest plots, numerical tables, and MST--RNG evidence

### Reviewer Comment

> **Paraphrased:** The reviewer requested larger and more legible forest-plot elements, numerical results that can be inspected directly in the manuscript, corrected significance reporting, and a direct main-text comparison of MST and RNG.

### Response

We increased the confidence-interval line widths and point-marker sizes by 200% across all forest plots to improve legibility at manuscript scale. We also embedded the numerical supplementary tables that were previously referenced only by external paths. These tables report the effect estimates, confidence intervals, raw p-values, adjusted q-values where applicable, sample counts, and directional summaries needed to audit the plotted comparisons.

The dataset-level figure annotations now use Benjamini--Hochberg adjusted q-values rather than raw p-values. The correction changed the significance threshold for two holdout-$QE$ hexagonal--RNG dataset points in Fig. 7B: `blobs` changed from p=0.0358 to q=0.0627, and `iris` from p=0.0444 to q=0.0745. Their annotations are now `ns`; other affected points retain the same substantive interpretation even where the displayed star level changed.

Finally, we moved the direct MST--RNG comparison into the main text as Fig. 8 rather than asking the reader to infer that contrast from two separate hexagonal comparisons. The result is deliberately qualified: MST and RNG are close on $QE$, while the matched MTR evidence favors RNG as the stronger joint $QE$/MTR option. Fig. 9, rather than Fig. 5, supplies the quantitative node-utilization and dead-node context.

### Manuscript Amendment

The revised forest plots use larger typography and graphical elements, the supplementary numerical tables are embedded in the manuscript, q-value annotations are corrected, and the main topology-results sequence now includes the direct MST--RNG comparison before the radius sensitivity and utilization diagnostics.

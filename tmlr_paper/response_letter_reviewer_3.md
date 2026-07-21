# Response Letter Draft: Reviewer 3

We sincerely thank the reviewer for the careful, constructive, and encouraging assessment of our work. We especially appreciate the reviewer highlighting the engineering contribution, the rigor of the empirical evaluation, and the practical relevance of modernizing SOM training for billion-scale workloads. The requested changes were thoughtful and helped us improve both the explanatory depth and the presentation of the manuscript. In response, we have expanded the systems accounting, strengthened the geometric explanation of the topology choices, added a real high-dimensional example to the representative-topology figure, and improved the legibility and auditability of the quantitative results.

## 1. Geometric intuition and expanded Figure 5

### Reviewer Comment

> First, I suggest expanding the discussion section to include a deeper geometric intuition regarding why the Relative Neighborhood Graph topology outperforms fixed grids on real-world datasets. Providing visual examples of how these dynamic graphs adapt to complex low-dimensional manifolds would be highly beneficial.

### Response

We agree that the topology motivation benefits from a more concrete geometric account and from an example beyond a two-dimensional synthetic dataset.

A fixed hexagonal lattice assigns neighborhood relationships before learning and independently of the evolving prototype geometry. Movement of one prototype can therefore influence lattice neighbors that are not locally related in learned data space, potentially displacing otherwise useful prototypes and increasing dead nodes. MST minimizes this coupling and gives prototypes greater freedom to redistribute along irregular or elongated structures. That freedom is also MST's limitation: a tree cannot retain several locally appropriate connections where a dense region is better described by a mesh-like neighborhood.

RNG occupies an intermediate position. It is less free than MST because nodes may influence several neighbors, but those additional connections are supported by the evolving prototype geometry rather than uniformly imposed beforehand. RNG can remain sparse where appropriate and form multiple connections in concentrated regions, influencing nodes that should be influenced without the fixed lattice's uniform coupling.

We expanded Fig. 5 from one row to a 2x3 layout. Panels A--C show circles in native 2D, and panels D--F show KDD Cup 99 trained on standardized 41-dimensional data and displayed using a shared two-dimensional PCA projection. In the KDD Cup 99 display, the hexagonal topology contains several apparently nonlocal connections spanning separated regions of the projected prototype distribution, whereas the MST and RNG connections more closely follow its local geometry. Quantitative node-utilization and dead-node evidence remains in Fig. 9 and the matched diagnostic tables.

### Summary of Manuscript Changes

We expanded Fig. 5 to a 2x3 comparison of matched hexagonal, MST, and RNG maps on both the native two-dimensional circles dataset and standardized 41-dimensional KDD Cup 99 data shown through a shared PCA projection. The Results now describe the apparently nonlocal fixed-lattice connections visible in the KDD projection, with an explicit warning that PCA can distort the original geometry. The Methods distinguish fixed regular-lattice adjacency from graph neighborhoods refreshed from evolving prototypes, and the Discussion develops the corresponding geometric interpretation: MST reduces predetermined coupling but is restricted to a tree, whereas RNG can retain several geometry-supported local connections. 

## 2. Distributed memory, communication, and large-grid cost

### Reviewer Comment

> Second, the authors should provide more quantitative details regarding the memory and communication overhead required to synchronize complex topology states across multiple GPUs, especially for large grid sizes. This would help clarify the sharp runtime increases observed in the grid-size scaling experiments.

### Response

We thank the reviewer for identifying where the original systems explanation needed more quantitative detail. FloatSOM does not communicate observations between workers during an iteration. Each worker processes its own shard in bounded chunks and accumulates a node-by-feature update numerator and a node-wise normalization denominator. Those two arrays are synchronized once per iteration by NCCL all-reduce; the updated prototypes then remain resident and identical on every worker.

For $P$ nodes and $d$ features stored as float32, the two accumulators contain $Pd+P$ values. Their logical all-reduce payload is therefore $4P(d+1)$ bytes per worker per iteration, plus a 4-byte sample-count reduction. For a ring all-reduce over $G$ workers, each worker sends and receives approximately $2(G-1)/G$ times that payload. This communication term is independent of the number of observations $N$: increasing $N$ increases local BMU/update work and data staging, whereas increasing $P$ increases both the synchronized update and the graph-topology work.

The memory explanation now separates the source of the cost from the mechanism used to manage it. With chunks of at most $C$ rows, the dominant bounded worker arrays scale as $O(Cd+Pd)$ rather than requiring the full $O(Nd)$ observations in GPU memory. Graph-distance and influence structures add both computation and storage terms that scale with $P^2$; for MST and especially RNG, topology construction, all-pairs path calculation, and influence-cache updates therefore become more expensive as node count increases. FloatSOM tiles these structures and can spill them from VRAM to system RAM when necessary. Spilling is a memory-management response that can add transfer overhead, but it is not the sole explanation for the runtime increase. The measured grid-size results show the combined practical consequence. At grid size 64 on 8 GPUs, hexagonal, MST, and RNG required 32.54, 266.45, and 880.83 s, respectively. Thus, the distributed pathway scales much more favorably in sample count than in node count, especially for RNG.

To further examine the RNG runtime spike in grid-size scaling, we compared the topology-construction operations. For each of the $\binom{P}{2}$ candidate node pairs, RNG checks every possible third node in its relative-neighborhood blocker test, giving up to $\binom{P}{2}(P-2)=O(P^3)$ pair--blocker comparisons per topology refresh. At $P=4096$, this is approximately 34.3 billion unordered pair--blocker checks. MST instead constructs and sorts approximately $\binom{4096}{2}=8.39$ million candidate edges using Kruskal's algorithm, an $O(P^2\log P)$ edge-construction step after pairwise distances are calculated. Both topologies then perform the shared graph-distance and influence calculations. The additional $O(P^3)$ RNG blocker-test work therefore provides a quantitative algorithmic explanation for the sharp RNG cost increase at large grid sizes.

### Summary of Manuscript Changes

We added communication and memory accounting to Section 3.3.1. For $P$ nodes, $d$ float32 features, and $G$ workers, the manuscript now gives the per-iteration logical all-reduce payload as $4P(d+1)$ bytes per worker plus the sample-count reduction, and the ring all-reduce traffic factor as approximately $2(G-1)/G$. It also distinguishes bounded worker storage, $O(Cd+Pd)$ for chunk size $C$, from the $P^2$ graph-distance and influence structures and explains tiling and RAM spill behavior.

Section 6.3 now reports the grid-size-64 runtimes on 8 GPUs—32.54 s for hexagonal, 266.45 s for MST, and 880.83 s for RNG—and quantifies the construction cost: at $P=4096$, RNG may perform about 34.3 billion pair--blocker checks, compared with approximately 8.39 million candidate MST edges before the shared graph-distance calculations. The Discussion uses these controlled internal measurements to qualify the topology recommendation and treats Somoclu and GigaSOM only as non-comparable published systems context.

## 3. Forest-plot legibility

### Reviewer Comment

> Finally, improving the legibility of the forest plots is necessary, as the data points and error bars are currently difficult to read without zooming in significantly.

### Response

We thank the reviewer for drawing our attention to this presentation issue. We increased the confidence-interval line widths and point-marker sizes by 200% across all forest plots to improve legibility at manuscript scale.

### Summary of Manuscript Changes

We regenerated all forest-plot assets with confidence-interval lines and point markers increased by 200%.

# Floatsom Paper

> **Deprecated (archival only):** This manuscript is retained for reference. The current non-colours manuscript is `manuscript.md`.

## Title

_TBD_

## Authors

_TBD_

## Abstract

_TBD_

## 1. Introduction

Self-Organizing Maps (SOMs) remain attractive for topology-preserving representation learning, but practical deployments now require scales and data heterogeneity that classical implementations do not target. In particular, most high-performance GPU SOM pipelines are optimized for regular lattice neighborhoods (most commonly hexagonal or rectangular grids), while non-lattice topologies are either unsupported or confined to small-scale CPU-oriented studies. This creates a gap between theoretical topology flexibility and real-world throughput requirements.

Our objective in this work is to close that gap by building a highly scalable non-hexagonal SOM architecture on GPUs and validating it at practically relevant workload sizes. Prior systems have generally been constrained in one of two ways: they support large GPU workloads but only fixed lattice neighborhoods, or they support graph-structured neighborhoods but at limited scale and without robust distributed execution. We target both dimensions simultaneously: topology flexibility and systems scalability.

The topology motivation is grounded in established SOM history. Minimum spanning tree (MST)-based neighborhood structure was already suggested in early SOM work by Kohonen [@kangasVariantsSelforganizingMaps1989]. We therefore implement an MST topology in a modern GPU/distributed stack and evaluate it directly against hexagonal baselines. At the same time, the field has increasingly considered Relative Neighborhood Graphs (RNGs) for geometric learning [@toussaintRelativeNeighbourhoodGraph1980]. In the standard RNG construction, two prototypes are connected when no third prototype lies closer to both endpoints than the endpoints are to each other under the blocker criterion. Our working hypothesis is that this direct proximity-graph definition can preserve a more regular local node distribution than MST while retaining the geometric freedom that fixed grids lack on non-regular data manifolds.

These motivations define the three innovations studied in this manuscript: a conflict-aware colours update algorithm for improved optimization quality, new graph topologies (MST and RNG) for non-hexagonal neighborhood structure, and a multi-GPU plus OOM-capable execution pipeline that scales these methods to large datasets and grid sizes.

Our experimental program is organized into two comparison tracks. In the algorithm-comparison track (focus of Sections 4.3, 5.1, and 6), we compare colours against batch under a fixed execution protocol and report both Quantization Error (QE) and distortion where available. In the topology-comparison track reported in this draft (Section 5.2), we compare MST against hexagonal; because distortion is unavailable for MST in this setup, topology comparisons are reported with QE-only objectives.

For algorithm benchmarking, the batch comparator is treated as a family rather than a single fixed regime: `batch_mode` is optimized internally over `{full_batch, minibatch}`. This choice targets the practically relevant quantity for finite-budget optimization, namely the strongest batch-family configuration per dataset, while preserving branch-disaggregated diagnostics for robustness interpretation.

## 2. Related Work

### 2.1 Python SOM Implementations

Classical Python SOM usage has been strongly influenced by MiniSom-style interfaces [@vettigliJustGlowingMinisom2026]. Within this ecosystem, XPySOM is a prominent high-performance Python implementation and a useful contextual reference for batch-SOM behavior. XPySOM is explicitly described as a batch SOM derived from MiniSom with NumPy/CuPy backends, providing a direct GPU acceleration path for single-process execution [@manciniXPySomHighPerformanceSelfOrganizing2020].

### 2.2 Beyond Fixed Lattices

Most practical high-throughput SOM implementations focus on regular lattice neighborhoods (rectangular/hexagonal), which simplify neighborhood indexing and vectorized updates. However, graph-structured neighborhoods have long been discussed in SOM literature. In particular, MST-style neighborhood structure appears in early SOM discussions by Kohonen [@kangasVariantsSelforganizingMaps1989]. More recent geometric-learning practice has also increased interest in proximity graphs such as Relative Neighborhood Graphs (RNGs) for balancing local regularity with geometric flexibility [@toussaintRelativeNeighbourhoodGraph1980].

### 2.3 Gap Between Topology Flexibility and Scalable Systems

The current gap is not only algorithmic but systems-level: methods that expose non-regular topologies are rarely implemented with robust large-scale GPU execution, while GPU-oriented implementations are often restricted to regular grids. This limits evaluation at real-world data sizes and makes it difficult to test whether non-hexagonal topologies remain practical under modern throughput requirements.

### 2.4 XPySOM Context Versus FloatSOM Batch Pipeline

We use XPySOM as an external calibration baseline in the Optuna quality track only, not as the primary baseline for topology or multi-GPU/OOM systems claims [@manciniXPySomHighPerformanceSelfOrganizing2020]. Both systems implement batch-style SOM updates, but they target different operating regimes.

The XPySOM implementation (`xpysom/xpySOM.py`) follows a single-process NumPy/CuPy design with local chunk-level parallelism controlled by `n_parallel`. Its batch update forms per-sample neighborhood weights and applies normalized updates with `nan_to_num` protection for zero-denominator cases. By design, it does not include distributed collectives, multi-GPU actor orchestration, or out-of-core data loading. In contrast, our batch pipeline is embedded in a Ray+NCCL distributed runtime with explicit multi-GPU synchronization, topology-aware extensions (MST/RNG), and bounded-memory chunked execution controls.

Accordingly, the contribution evaluated in this manuscript is not merely kernel-level acceleration, but a systems-level extension of batch SOM to larger deployment envelopes: multi-GPU worker groups, disk-backed loading, bounded-memory chunking, and recovery-oriented execution paths. In this draft, XPySOM is used for quality calibration in the Optuna track, while topology and systems scaling results remain within-FloatSOM comparisons.

## 3. Methods

We present the methods in the same order as the results: update-rule innovation, topology innovation, and systems acceleration. Specifically, Section 3.1 introduces the custom colours scheduler, Sections 3.2-3.3 define the two graph topologies (MST and RNG), and Section 3.4 describes the multi-GPU and OOM-capable execution stack used to run these algorithms at scale. Standard SOM primitives are assumed from prior literature and are not re-derived here.

### 3.1 Custom Colours Algorithm

The colours methodology is implemented as a conflict-constrained scheduling layer on top of the established SOM update rule. The central design objective is to maximize independent parallel updates while avoiding simultaneous updates among neurons that have non-negligible mutual neighborhood influence.

At iteration $t$, the topology module returns a precomputed influence matrix $I^{(t)} \in \mathbb{R}^{N \times N}$. The induced conflict graph is:

$$
(i,j)\in E_c^{(t)} \iff I_{ij}^{(t)} > \tau.
\tag{1}
$$

In the current implementation, thresholding is algorithm-specific: the systematic general routine uses $\tau=0.1$, whereas greedy-balanced routines for irregular graphs use $\tau=0.01$. The resulting color partition $\{S_1,\dots,S_C\}$ is computed with topology-dependent fast paths: regular layouts (`grid`, `hexagonal`) use direct vectorized tiling formulas, while irregular structures use vectorized MIS/greedy-balanced assignment. The regular-layout fast path is a surrogate for explicit conflict-graph coloring: it is designed to enforce broad independence patterns without constructing all pairwise conflict edges each iteration. To avoid redundant recomputation, color sets are cached by the deduplicated radius key returned by topology precomputation.

```text
Algorithm 1: Colors Partition Construction
Input: influence matrix I_t, topology type T, threshold policy τ(T), optional target color count C*
Output: color sets S = {S_1, ..., S_C}
1: Build conflict graph G_c^t where (i, j) is an edge iff I_t[i, j] > τ(T)
2: if T is regular (grid or hexagonal) then
3:     Use direct vectorized color assignment formulas
4: else
5:     Use vectorized MIS/greedy-balanced graph coloring
6: return S
```

#### 3.1.1 Single-round execution path (primary setting)

For this manuscript, the primary protocol is the one-round configuration (`max_rounds = 1`). Under this regime, each outer training iteration executes a single pass over the color sets. The algorithm first acquires the radius-specific influence map and corresponding cached color partition, constructs the sample order (`random` or `strided`), and computes BMUs against current weights. It then traverses color sets in randomized order; for each color set, only the samples whose BMUs belong to that set are selected. The selected subset contributes accumulated update and accumulated influence tensors, and the normalized/momentum-adjusted update is applied in place.

For color set $S_c$ with selected sample subset $B_c$, local accumulation is given in Eq. (2):

$$
\begin{aligned}
U_j^{(c)} &= \sum_{x \in B_c} \eta_t\, h_{j,b(x)}^{(t)}\,(x - w_j^{(t)}), \\
H_j^{(c)} &= \sum_{x \in B_c} h_{j,b(x)}^{(t)}.
\end{aligned}
\tag{2}
$$

`compute_weight_updates(...)` performs this in chunked GPU kernels using adaptive chunk size `min(len(samples_for_color_set), config.chunk_size)`.

```text
Algorithm 2: Single-Round Colors Update
Input: weights W_t, samples X_t, radius r_t, learning rate η_t, momentum m_t, topology T
Output: updated weights W_{t+1}
1: I_t ← T.get_precomputed_influence_matrix(r_t, gaussian)
2: S ← ColorPartition(I_t, T)  // Algorithm 1, cached by deduplicated radius
3: π ← sample ordering (random permutation or strided order)
4: b(x) ← BMU of each x ∈ X_t[π] under current W_t
5: for each color set S_c in randomized order do
6:     B_c ← {x ∈ X_t[π] : b(x) ∈ S_c}
7:     Compute (U^(c), H^(c)) with chunked GPU accumulation
8:     Normalize(U^(c), H^(c)) and apply momentum m_t
9:     W_t ← W_t + ΔW^(c)
10: end for
11: return W_t
```

#### 3.1.2 Multi-round extension (implemented)

The implementation also supports $R>1$ rounds per outer iteration. Chunk indices are partitioned into $R$ round-groups (`np.array_split`) after optional randomization. At each round, the corresponding sample subset is gathered and BMUs are recomputed against the latest intermediate weights; thus, each round consumes a progressively updated model state. This yields the inner map in Eq. (3). The one-round case is the $R=1$ specialization.

$$
w^{(t,r+1)} = \Phi_r(w^{(t,r)}), \quad r=0,\dots,R-1.
\tag{3}
$$

### 3.2 MST Topology Implementation

Having specified the update scheduler, we next define the first topology contribution: MST neighborhoods computed from prototype geometry rather than fixed lattice adjacency.

MST topology replaces fixed lattice neighborhood distance with graph shortest-path distance on a minimum spanning tree built from current prototypes. The pairwise prototype matrix is formed with the GPU Gram-identity kernel in Eq. (4), which avoids 3D broadcast tensors and preserves $O(N^2d)$ dense linear-algebra structure.

$$
D_{ij}^2 = \lVert w_i \rVert_2^2 + \lVert w_j \rVert_2^2 - 2 w_i^\top w_j.
\tag{4}
$$

After distance construction, MST edges are extracted by CPU Kruskal, adjacency is built, and all-pairs graph distances are computed via chunked GPU Floyd-Warshall. The row chunk size is resolved from memory-budget controls (`_resolve_fw_row_chunk_size(...)`) to bound temporary allocations. Learning-time neighborhood influence is then evaluated on graph distances using Eq. (5).

$$
h_{ij}(r_t) = \exp\!\left(-\frac{g_{ij}^2}{2\,(r_t/2)^2}\right).
\tag{5}
$$

To amortize repeated topology queries, radii are deduplicated using a 10% threshold and influence matrices are cached by `(radius, influence_function)`. Topology refresh is controlled by fixed or dynamic update frequency. In dynamic mode, with progress variable $p_t=\min(t/T,1)$, the implemented schedule is:

$$
f_t = \mathrm{clip}_{[f_0,f_T]}\!\left(\mathrm{round}\!\left(f_0 + (f_T - f_0)\,\gamma(p_t)\right)\right).
\tag{6}
$$

with $\gamma$ selected from the implemented family:
$$
\gamma(p)=
\begin{cases}
\dfrac{e^{5p}-1}{e^{5}-1}, & \texttt{exponential},\\[6pt]
p, & \texttt{linear},\\[4pt]
\dfrac{1}{1+e^{-15(p-0.5)}}, & \texttt{sigmoid},\\[8pt]
\dfrac{1-\exp\!\left(-\dfrac{(p-0.5)^2}{2(0.3)^2}\right)}
{1-\exp\!\left(-\dfrac{0.25}{2(0.3)^2}\right)}, & \texttt{gaussian},\\[12pt]
\dfrac{5p}{1+5p}, & \texttt{asymptotic}.
\end{cases}
\tag{7}
$$

The topology update trigger follows:
$$
u_t=\mathbb{1}\!\left[(t=0)\ \lor\ ((t+1)\bmod f_t=0)\right].
\tag{8}
$$

```text
Algorithm 3: Dynamic MST Topology Update
Input: prototypes W_t, iteration t, total iterations T, schedule parameters (f_0, f_T, γ)
Output: topology state (E_t, g_t, cached influences)
1: Compute refresh trigger u_t using Eq. (8)
2: if u_t = 0 then
3:     return previous topology state
4: end if
5: Compute pairwise squared distances D_t^2 on GPU using Eq. (4)
6: Transfer D_t^2 to CPU and run Kruskal to obtain MST edges E_t
7: Build adjacency list from E_t
8: Compute all-pairs hop distances g_t with chunked GPU Floyd-Warshall
9: Build/update influence-cache entries h(g_t, r) for deduplicated radii
10: Commit E_t, g_t, and cache state
11: return topology state
```

### 3.3 RNG Topology Implementation

We then introduce RNG as the second topology contribution. The implementation is intentionally aligned with MST downstream so that differences in behavior arise primarily from edge construction rather than subsequent caching and shortest-path machinery.

RNG topology constructs a Relative Neighborhood Graph over current prototype distances and then reuses the MST infrastructure for shortest-path precomputation, radius-deduplicated influence caching, and dynamic update scheduling.

The edge criterion is given in Eq. (9):

$$
(i,j) \in E_{\mathrm{RNG}}
\iff
\nexists k \neq i,j \text{ such that } \max(D_{ik},D_{jk}) < D_{ij}.
\tag{9}
$$

Candidate elimination is GPU-chunked along the blocker-node axis to control memory pressure while preserving the direct strict blocker test. No post-hoc connectivity repair is applied after edge extraction. In implementation terms, `RNGTopology` subclasses `MSTTopology` and overrides edge construction only, so downstream graph-distance and influence-cache behavior is intentionally shared.

```text
Algorithm 4: RNG Topology Construction
Input: pairwise distances D_t
Output: RNG edge set E_t
1: Initialize upper-triangular finite candidate set C
2: for blocker-node chunks K do
3:     Identify blocked pairs (i, j) for which ∃k ∈ K:
4:         max(D_t[i, k], D_t[j, k]) < D_t[i, j]
5:     Remove blocked pairs from C
6: end for
7: E_t ← remaining pairs in C
8: return E_t
```

### 3.4 Multi-GPU + OOM Methodology and Implementation

After defining the algorithmic components, we describe the execution system used in experiments. This section covers how we distribute computation across GPUs, how data are streamed for large workloads, and how memory safeguards preserve progress under high-pressure regimes.

#### 3.4.1 General Multi-GPU Logic

Distributed execution uses Ray actors with one GPU per worker and NCCL collectives for synchronous aggregation. Each worker computes local update/influence accumulators, followed by global summation.

For worker $g \in \{1,\dots,G\}$, local accumulators are:

$$
\begin{aligned}
U_j^{(g)} &= \sum_{x \in X_g} \eta_t\,h_{j,b(x)}^{(t)}(x-w_j), \\
H_j^{(g)} &= \sum_{x \in X_g} h_{j,b(x)}^{(t)}.
\end{aligned}
\tag{10}
$$

Global synchronized accumulators are:

$$
\begin{aligned}
U_j &= \sum_{g=1}^{G} U_j^{(g)}, \\
H_j &= \sum_{g=1}^{G} H_j^{(g)}.
\end{aligned}
\tag{11}
$$

In colors mode, sample counts are all-reduced per color-flush step:

$$
S_{c,f} = \sum_{g=1}^{G} \left|X_{g,c,f}\right|.
\tag{12}
$$

Normalization and momentum are then applied with globally consistent denominators (implemented as $\max(S_{c,f},1)$ for numerical safety). Weights remain resident on worker GPUs across iterations, and the driver exchanges lightweight metadata rather than full weight tensors except when an explicit topology refresh fetch is required.

#### 3.4.2 Multi-GPU Implementation Details

The implementation centers on a custom CPU-to-GPU loader pipeline and a buffered compute/transfer schedule. `RayWorkerManager` orchestrates actor lifecycle, data staging, and collective group creation. Worker-local storage is served through `FastArrayStore`, and each worker instantiates `CPUGPUFastLoader`, which provides memory-mapped or RAM-direct chunk access with pinned host staging buffers.

For large workloads, the loader operates in disk-backed chunk mode by default: chunks are read from `FastArrayStore` using memory-mapped views and transferred in bounded batch units (`chunk_size`) rather than materializing full datasets in GPU memory. The loader maintains an adaptive host prefetch cache whose depth is constrained by available RAM and workers-per-node, and falls back to direct mmap reads when a chunk is not already resident in host cache. This design allows training to proceed on datasets that exceed aggregate device memory capacity.

For large neuron grids in graph topologies (MST/RNG), topology-side distance and influence structures are also memory-aware: when node count exceeds the CPU-offload threshold, graph-distance matrices and cached influence maps are stored on CPU and transferred back to GPU on demand. Consequently, both data ingestion and topology-state management are executed in chunked/disk-backed fashion under high memory pressure.

The loader exposes direct transfer into preallocated GPU buffers (`transfer_to_gpu_buffer`) and supports multiple pinned host buffers indexed by transfer slot (`pinned_buffer_index`). This is used by the worker multi-buffering subsystem, which allocates a configurable number of GPU buffers and synchronizes transfer and compute streams with CUDA events. The default configuration in this codebase uses three buffers (`DEFAULT_MULTI_BUFFERING_PRELOAD_BUFFERS = 3`), yielding a triple-buffer schedule: one buffer under compute, one buffer receiving the next transfer, and one buffer prefetched ahead. Buffer reuse is guarded by ready/done event dependencies to prevent overwrite-before-consume hazards.

NCCL strategy is synchronous SUM all-reduce over update and influence tensors for each iteration; the colors path additionally all-reduces sample counts to enforce an identical normalization denominator across workers. Collective signatures are kept identical across ranks by explicit dtype and contiguity normalization in the color worker path. Optional collective warmup and optional barriers are available for stability instrumentation; payload minimization is enforced by sending topology objects only when required.

Color synchronization is executed in lock-step at color-flush granularity. For each color index and flush slot, every worker first contributes a binary availability indicator (`has_batch`) through all-reduce. If the global sum is zero, all workers execute an empty-update path to preserve collective call parity. Otherwise, workers with local color data execute update accumulation while workers without data enter the same collective sequence with zero tensors. In both cases, all workers participate in identical all-reduce calls for updates, influence, and sample counts, thereby guaranteeing globally consistent color-step updates and eliminating rank divergence.

```text
Algorithm 5: Multi-GPU Iteration with Loader + Triple Buffer + NCCL
Input: distributed workers {1..G}, persistent GPU weights W_t on each worker
Output: synchronized updated weights W_{t+1}
1: On each worker, read chunk stream via CPUGPUFastLoader (FastArrayStore backend)
2: Allocate/ensure B transfer slots (default B=3): pinned host buffers + GPU buffers
3: For each local step:
4:     Schedule H2D transfer of current chunk to buffer b on transfer stream
5:     Prefetch up to B-1 future chunks into remaining slots
6:     Wait on ready-event(b), run BMU/update compute on compute stream
7:     Record done-event(b) so transfer stream can safely reuse slot b
8: Aggregate local accumulators U^(g), H^(g) (and S^(g) in colors mode)
9: NCCL all-reduce SUM: U ← Σ_g U^(g), H ← Σ_g H^(g), and S ← Σ_g S^(g) when needed
10: Normalize and apply momentum using global denominators
11: Update persistent GPU weights in place and keep W_{t+1} resident
12: return lightweight iteration metadata to driver
```

```text
Algorithm 6: Cross-GPU Color Synchronization (per color, per flush)
Input: local color batches on each worker g, collective group C
Output: synchronized color-step weight update
1: has_batch_g ← 1 if worker g has pending data for this color/flush, else 0
2: has_batch_global ← AllReduceSUM_C(has_batch_g)
3: if has_batch_global = 0 then
4:     Execute empty-update path on all workers (collective parity preservation)
5: else
6:     On workers with data: compute local (U^(g), H^(g), S^(g))
7:     On workers without data: set (U^(g), H^(g), S^(g)) to zeros
8:     U ← AllReduceSUM_C(U^(g)); H ← AllReduceSUM_C(H^(g)); S ← AllReduceSUM_C(S^(g))
9:     Normalize with global S and apply momentum identically on all workers
10:    Apply synchronized color-step update to resident weights
11: end if
12: return
```

#### 3.4.3 OOM-Capable Strategy

OOM robustness is implemented through layered memory controls rather than an exceptional fallback path. At compute time, BMU search is node-chunked, update accumulation is sample-chunked, Floyd-Warshall is row-tiled by byte-budget policy, and RNG blocker evaluation is chunked over the third axis. At transfer time, pinned host buffers and bounded prefetch depth limit host and device staging pressure. In addition, processing configuration enforces stability guardrails (including colors chunk-size capping).

Operationally, workers perform explicit stream synchronization and memory-pool cleanup at controlled boundaries to reduce allocator fragmentation during long runs. For distributed robustness, the colors processor uses a progress watchdog with default fallback timeout (300 s when unset), straggler detection based on peer-partition medians, and hard reset/retry from last-known-good CPU checkpoint governed by `iteration_timeout_max_retries`.

#### 3.4.4 Implementation Fidelity and Deliberate Deviations

The equations above match the implemented core update and topology logic, with three deliberate implementation safeguards:

1. MST edge construction uses squared Euclidean distances $D^2$ (Eq. (4)) without square root. This is not numerically identical to $D$, but it is order-equivalent for Kruskal, so the resulting MST is unchanged under exact arithmetic.
2. Distributed colors normalization uses denominator clamp $\max(S_{c,f},1)$, and lock-step empty-update paths when no worker has data for a flush. This deviates from the unconstrained analytic form only on degenerate zero-sample flushes, and is required to avoid undefined normalization and collective divergence.
3. For regular (`grid`, `hexagonal`) topologies, colours assignment uses a direct vectorized tiling surrogate instead of explicit per-iteration conflict-graph construction. This is a throughput- and memory-oriented approximation; explicit conflict-graph routines are retained for irregular topologies.

#### 3.4.5 Relation to the XPySOM Batch Update

To make baseline claims precise, we map our batch equations to the XPySOM source implementation. For a local batch $B_t$, XPySOM forms per-node weights
$q_{xj}=\eta_t\,h_{j,b(x)}^{(t)}$, then computes:
$$
\Delta w_j^{\mathrm{XPY}}
=
\frac{\sum_{x\in B_t} q_{xj}\,(x-w_j^{(t)})}
{\sum_{x\in B_t} q_{xj}},
\quad
w_j^{(t+1)}=w_j^{(t)}+\mathrm{nan\_to\_num}\!\left(\Delta w_j^{\mathrm{XPY}}\right).
\tag{13}
$$

Our accumulated numerator is the same structural term prior to normalization:
$$
U_j^{\mathrm{FS}}=\sum_{x\in B_t}\eta_t\,h_{j,b(x)}^{(t)}(x-w_j^{(t)}).
\tag{14}
$$

FloatSOM is now aligned with XPySOM for this batch-update normalization path, i.e., the same denominator structure is used:
$$
\Delta w_j^{\mathrm{FS}}
=
\frac{U_j^{\mathrm{FS}}}{\sum_{x\in B_t} q_{xj}}.
\tag{15}
$$

Consequently, FloatSOM and XPySOM are aligned both at accumulation and normalization for this pathway, and therefore match the same core batch-update form.

These update-equivalence statements are orthogonal to standard SOM primitives, which remain aligned with established practice in both codebases: Euclidean BMU selection via squared-distance order preservation, standard Gaussian/mexican-hat/bubble neighborhood kernels, and conventional learning-rate/radius decay families. We therefore treat these primitives as shared background and focus novelty claims on topology design and distributed/OOM systems behavior.

## 4. Experimental Setup

### 4.1 Datasets and preprocessing

We use two dataset protocols: Optuna quality benchmarking and speed scaling benchmarking.

Optuna protocol uses the `run_optuna --mode full --config full` scenario set, which covers ten sklearn datasets: `swiss_roll`, `moons`, `circles`, `blobs`, `s_curve`, `breast_cancer`, `wine`, `iris`, `digits`, and `olivetti_faces`. In this configuration, dataset difficulty is fixed to `hard` for all scenarios. For synthetic datasets, this concretely means: `swiss_roll` (`n_samples=30000`, `noise=0.2`, `hole=True`), `moons` (`n_samples=30000`, `noise=0.2`), `circles` (`n_samples=30000`, `noise=0.2`, `factor=0.2`), `blobs` (`n_samples=30000`, `centers=8`, `cluster_std=2.0`, default `n_features=2` unless overridden), and `s_curve` (`n_samples=30000`, `noise=0.2`). Real-world datasets are loaded from sklearn with their native sample/feature layouts.

For Optuna preprocessing, `generate_sklearn_dataset(...)` applies `StandardScaler` normalization when `normalize=True`, then converts arrays to `cupy.float32`. Each trial uses a deterministic seeded permutation and a fixed 70/30 train-holdout split. Model fitting is performed on the training split, while objective metrics are computed on holdout (and optionally train, depending on metric type/split configuration).

Speed-benchmark protocol generates synthetic random matrices with uniform values in $[0,1]$, stores them as chunked `float32` Zarr arrays, and reuses cached datasets keyed by `(samples, input_dim, seed, zarr_chunk_size)` hash. Benchmark training reads these arrays through `FileDataSource` and the fast CPU-GPU loader path, enabling repeated scaling runs without regenerating data.

Sampling modes supported by the training stack are `full`, `random`, and `hdsssom`; for non-`full` modes, `samples_per_epoch` is derived from dataset size and configured sampling fraction. Let $X=\{x_i\}_{i=1}^{N}$ and let $m$ denote the per-iteration sample budget. The three regimes are:

$$
<<<<<<< main
\mathcal{I}_t^{\mathrm{full}}=\{1,\dots,N\}, \qquad
\mathcal{I}_t^{\mathrm{random}} \sim \mathrm{Unif}\!\left(\left\{I\subseteq\{1,\dots,N\}:|I|=m\right\}\right), \qquad
\mathcal{I}_t^{\mathrm{hdsssom}}=\operatorname{HDSSSOM}\!\left(\mathbf{d}_{t-1},\mathbf{a}_{t-1};m\right),
\tag{16}
$$

=======
\mathcal{I}_t^{\mathrm{full}}=\{1,\dots,N\},
$$

$$
\mathcal{I}_t^{\mathrm{random}} \sim \mathrm{Unif}\!\left(\left\{I\subseteq\{1,\dots,N\}:|I|=m\right\}\right),
$$

$$
\mathcal{I}_t^{\mathrm{hdsssom}}=\operatorname{HDSSSOM}\!\left(\mathbf{d}_{t-1},\mathbf{a}_{t-1};m\right).
\tag{16}
$$

>>>>>>> local
with selected training batch $X_t^{(s)}=\{x_i: i\in \mathcal{I}_t^{(s)}\}$. Here, `random` is the naive baseline (uniform random subset each iteration), while HDSSSOM is treated as a non-naive subsampling regime following [@wetmoreSpeedingSelfOrganizingFeature2005].

For random subsampling, optimization remains aligned to the whole-dataset objective in expectation:
$$
\mathbb{E}_{\mathcal{I}_t^{\mathrm{random}}}\!\left[\frac{1}{m}\sum_{i\in\mathcal{I}_t^{\mathrm{random}}}\ell(x_i;W)\right]
=
\frac{1}{N}\sum_{i=1}^{N}\ell(x_i;W).
\tag{17}
$$

This is why smarter subsampling policies can improve practical quality-time tradeoffs: they process fewer samples per iteration while still targeting the full dataset objective over training. In the speed-scaling PBS series used in this paper, sampling is fixed to `full`.

### 4.2 Metrics

Primary quality metric is Quantization Error (QE), implemented with GPU distance kernels as in Eq. (18):

$$
\mathrm{QE}(X,W)=\frac{1}{|X|}\sum_{x\in X}\min_j \|x-w_j\|_2.
\tag{18}
$$

The objective/benchmark stack also supports `topographic_error`, `trustworthiness`, `neighborhood_preservation`, `distortion_measure`, and `topographic_function`.

For the analyses reported in this manuscript, metric usage is stratified by comparison target: algorithm comparison (colours versus batch) uses QE plus distortion where available, whereas topology comparison (MST versus hexagonal in the current draft) uses QE-only objectives because distortion is not defined for MST in this setup.

To keep optimization numerically stable, invalid metric values are sanitized to finite penalties:
1. minimization objectives map invalid values to `+1e10`,
2. maximization objectives (`trustworthiness`, `neighborhood_preservation`) map invalid values to `-1e10`.

Topology-only metrics are treated specially in split handling and are not optimized on train-only objective configurations.

### 4.3 Optuna protocol (algorithm-comparison campaign)

The algorithm-comparison Optuna campaign is orchestrated by `floatsom/benchmarks/optuna/run_multiple_optuna_jobs.sh`. The launcher defaults to `NUM_SEEDS=${1:-5}`, samples seeds from shell `$RANDOM`, fixes `EVALUATION_SPLITS=("both")`, and submits one PBS job per `(seed, split)` pair. Each generated job requests queue `gpuvolta` resources (`ngpus=4`, `ncpus=48`, `mem=128GB`, `walltime=47:00:00`, `jobfs=400GB`) and loads `rapids/25.06`.

All production Optuna and speed benchmarks reported in this manuscript were executed on NCI Gadi (National Computational Infrastructure, Australia) `gpuvolta` nodes [@nciGadiHpcSystems2026]. The documented node profile is 4 `NVIDIA V100` GPUs (32 GB HBM2 each), dual 24-core Intel Cascade Lake CPUs, and 382 GB host RAM, on a 200 Gb/s HDR InfiniBand Dragonfly+ fabric [@nciGadiHpcSystems2026]. Storage-backed data movement used Gadi shared filesystems, notably Lustre-backed `/scratch` (20 PB, 7,200 disks, up to 980 GB/s) and `/g/data` project storage (>90 PB aggregate) [@nciStorageSystems2026].

For the production run analyzed in this manuscript, each job executed `python3 -m floatsom.benchmarks.optuna.run_optuna --mode full --config full --topology hexagonal --objectives quantization_error distortion_measure --trials 200 --seed <seed> --evaluation-split both`, with sampling scenarios covering all three methods (`sampling_method \in \{full, random, hdsssom\}`). Results are written under `/g/data/eu59/SIFEAN/sfa/noninf_optuna_benchmarks/both/seed_<seed>/`, with scheduler logs in `gadi_optunalogs/`.

For QE-comparison fairness, training-iteration count is fixed across trials in the Optuna execution path used for this manuscript. In the objective construction, `iterations` is not part of the Optuna parameter configuration (`floatsom/benchmarks/optuna/config/parameters.py`), and `FloatSOMParams` is instantiated with `total_iterations=params.get('iterations', 100)` and `min_iterations=params.get('min_iterations', 100)` (`floatsom/benchmarks/optuna/core/objective.py`). Because no trial-specific `iterations` override is provided in the executed campaign, each QE trial runs with the same iteration count, so `full` versus `random` QE comparisons are iteration-matched.

This control also clarifies the `colors` versus batch-family interpretation: the main training loop executes the same outer iteration budget for all methods (`for iteration in range(total_iterations)` in `floatsom/base/floatsom.py`), and the `batch_mode=minibatch` branch applies repeated intra-iteration updates over minibatches (`for i in range(0, n_samples, self.chunk_size)` in `floatsom/processing/batch_processor.py`). In minibatch mode, this is approximately $\lceil n_{\mathrm{train}}/\texttt{chunk\_size}\rceil$ weight-update applications per outer iteration (one per minibatch), which can be substantially greater than one. Thus, minibatch acts as a high-update comparator under the same outer iteration budget, so observed `colors` advantages are not attributable to simply granting `colors` extra update opportunities.

With `--evaluation-split both`, objectives are expanded into split-aware targets. For QE, this produces two optimization targets (`quantization_error_holdout`, `quantization_error_train`), which are the two QE objectives used in the split-wise algorithm comparisons reported here. Distortion-based comparisons are restricted to algorithmic comparisons where the metric is available.

Inside `run_optuna --mode full`, the `full` preset uses algorithm set `{'colors','batch'}` and 10 datasets (`swiss_roll`, `moons`, `circles`, `blobs`, `s_curve`, `breast_cancer`, `wine`, `iris`, `digits`, `olivetti_faces`). Under the executed restrictions (`topology=hexagonal`, `sampling_method \in \{full, random, hdsssom\}`), this yields $60$ scenarios per seed ($10$ datasets $\times$ $2$ algorithms $\times$ $3$ sampling methods $\times$ $1$ topology). With `--trials 200`, the trial budget is $12{,}000$ trials per seed ($60{,}000$ for the default 5-seed launcher invocation).

Within each batch scenario, `batch_mode` is optimized as an internal categorical hyperparameter over `{full_batch, minibatch}`, and minibatch-specific controls are activated conditionally (for example, `chunk_size` is active only when `batch_mode=minibatch`). This hierarchical parameterization is aligned with TPE's tree-structured search: Optuna can model conditional parameter dependencies and can concentrate proposal density within the active branch, while branch-specific parameters remain inactive outside their branch. This enables joint optimization of full-batch and minibatch subspaces under shared objectives and permits per-dataset branch selection without running separate external campaigns.

Under the default launcher invocation (`NUM_SEEDS=5`, split fixed to `both`), the executed campaign consists of $5 \times 60 \times 200 = 60{,}000$ trials. Per seed, the batch arm contributes $30$ scenarios ($6{,}000$ trials), and the colours arm contributes $30$ scenarios ($6{,}000$ trials). A fully segregated equal-budget branch campaign that externally split the batch arm into fixed `full_batch` and fixed `minibatch` scenario sets would require $90$ scenarios per seed ($18{,}000$ trials per seed; $90{,}000$ total trials over 5 seeds), i.e., $+30{,}000$ additional trials relative to the executed campaign ($+50\%$ trial volume). This additional workload exceeded the available benchmark compute budget.

At the trial level, the objective pipeline is: deterministic dataset generation and split, merge of sampled parameters with forced categorical scenario parameters, conditional-parameter activation (`get_conditional_parameters()`), FloatSOM training, metric evaluation on split-aware targets, and objective return as scalar or vector depending on objective count. Normalization-specific fallback parameters (`norm_alpha`, `norm_clamp_factor`, `norm_percentile`) are populated when required by the selected normalization mode. Persisted trial artifacts include split-aware metric dictionaries (`metrics_holdout`, `metrics_train`), per-objective values, training metadata (`train_time`, `iterations_completed`, split sizes), and sampled/forced parameter assignments.

#### 4.3.1 External calibration baseline (XPySOM)

To anchor against an accepted prior Python SOM implementation, we include XPySOM as an external calibration baseline in the Optuna quality track. This baseline is intentionally restricted to the quality-comparison envelope where protocol matching is feasible.

Fairness controls are:
1. same dataset list and deterministic train/holdout splits used in the FloatSOM Optuna analysis,
2. same seed set at scenario level,
3. fixed and iteration-matched training budget (same total iterations per trial),
4. matched map size/topology-compatible setting (hexagonal/lattice-compatible comparisons only),
5. same primary quality endpoint reporting (QE split-wise and Balanced QE).

Because XPySOM is single-process and does not implement our distributed OOM/topology extensions, it is not used as a comparator in Section 5.2 (MST/RNG topology claims) or Section 5.4 (multi-GPU/OOM speed scaling claims). Its role is calibration in the Optuna quality results only.

### 4.4 Speed benchmark protocol

The scaling results are produced by the PBS series `run_gpu_benchmarking_{1,2,3,4,8}gpu.pbs.sh`, each invoking `floatsom.benchmarks.speed_benchmarks.run_gpu_scaling_benchmark` in `--mode sample_scaling`. Across this series, the benchmark payload is fixed to `--processing_methods minibatch`, `--topologies hexagonal mst`, `--minibatch_chunk_size 5000`, `--merge_existing`, `--resume`, and per-run `--resume_state` files, with a shared cache directory at `/g/data/eu59/SIFEAN/sfa/8gpu_scaling_cache`.

Because these scripts do not override the remaining CLI defaults, each run uses `repeats=3`, `sample_sizes=[1000000, 5000000, 10000000, 50000000, 100000000, 500000000]`, `fixed_dimension=50`, `grid_size=32`, `total_iterations=10`, `initial_learning_rate=2.0`, `sampling_method=full`, `run_timeout_minutes=30`, and base seed `42` (repeat seeds `42,43,44` via `seed + repeat`). Per GPU-count script, this yields $2$ topologies $\times$ $1$ method $\times$ $6$ sample sizes $\times$ $3$ repeats $=36$ benchmark runs.

Scheduler resources are scaled with GPU count $G \in \{1,2,3,4,8\}$: `ncpus={12,24,36,48,96}`, `mem={90,180,270,360,720}GB`, `ngpus={1,2,3,4,8}`, `walltime=24:00:00`, queue `gpuvolta`, and module `rapids/25.06`. Job filesystem is `400GB` for $G \le 4$ and `800GB` for $G=8$. In all scripts, `--gpu_counts G` and `--ray_gpu_count G` are matched, and `--ray_local_storage_path "$PBS_JOBFS"` is set. The 8-GPU script additionally performs multi-node environment initialization before benchmark launch.

At runtime, each configuration executes in a spawned subprocess with timeout enforcement, Ray cleanup between runs, and incremental result persistence for resume/merge recovery. Scaling statistics are then computed from per-configuration train times over repeats.

### 4.5 Statistical analysis

All Optuna comparison analyses are paired. For each comparison family, rows are matched by `dataset`, `pair_sampling`, `pair_seed`, and `pair_split`, with `pair_batch_mode` included where relevant. Within each paired unit, trials are rank-ordered by the target metric (lower is better), the top-$k$ trials are retained (`top_k=5` in the main publication figures), and each side is summarized by the median of retained trials before constructing pairwise differences.

For each dataset, statistical significance is computed from paired deltas using a two-sided Wilcoxon signed-rank test on non-zero differences (`zero_method='wilcox'`, `correction=False`, `mode='auto'`). We report raw $p$-values and apply Benjamini-Hochberg correction across dataset-level tests (excluding the GLOBAL aggregate row) to obtain $q$-values. Significance threshold is $\alpha=0.05$, and figure significance flags use $q<\alpha$ when $q$-values are available (falling back to $p<\alpha$ otherwise).

Effect summaries include wins/ties/losses and a signed win-balance effect proxy $(\text{wins}_A-\text{wins}_B)/(\text{wins}_A+\text{wins}_B)$. Uncertainty is reported as a 95% Wilcoxon-compatible confidence interval for the location shift (Hodges-Lehmann style, via inversion of the same two-sided signed-rank test used for $p$-values). In this manuscript, wording such as "no detected difference" means failure to reject at the stated threshold under this paired-testing pipeline; it is not a formal equivalence claim.

## 5. Results

### 5.1 Custom Colours Results

This section reports algorithm-level comparisons (`colors` versus `batch`) on the raw Optuna outputs from the executed three-sampling, single-topology campaign (`sampling_method \in \{full, random, hdsssom\}`, `topology=hexagonal`).

#### XPySOM calibration baseline (Optuna track only)

As an accepted external baseline, XPySOM is used here as a calibration reference for quality interpretation under the matched Optuna-style protocol described in Section 4.3.1. This calibration is limited to hexagonal/lattice-compatible quality comparisons with fixed iteration budgets and matched splits/seeds.

In this manuscript, XPySOM calibration outputs are reported as Supplementary Figure S6 (`assets_manual/figures/supp_fig_xpysom_calibration_qe_hexagonal.svg`) and Supplementary Table S1 (`assets/tables/supp_xpysom_calibration_qe_hexagonal.tsv`). Main deployment conclusions continue to rely on the within-FloatSOM paired analyses because those are matched to the topology and systems capabilities evaluated in Sections 5.2 and 5.4.

**Supplementary Figure S6. XPySOM calibration on QE under matched hexagonal Optuna-style settings.**


**Supplementary Table S1. XPySOM calibration QE summary (dataset-wise and global paired outcomes).** See `assets/tables/supp_xpysom_calibration_qe_hexagonal.tsv`.

In comparisons of `full`, `random`, and `hdsssom` under the hexagonal-topology setting, `hdsssom` performed worse than both `full` and `random`. Figure 1 shows that `random` sampling was not inferior to `full` under the executed conditions, with some cases outperforming `full`.


Therefore, going forward in the Optuna dataset analyses in this manuscript, `random` and `full` sampling runs are pooled together unless explicitly reported separately.

In the supplementary sampling-comparison outputs (Supplementary Figure S1A-S1B), there is no case where `full` is statistically superior to `random`. In the executed configuration, the only detected statistical difference is one case where `random` is statistically better than `full`.

Combined with the runtime-ratio evidence in Figure 6A-C, this indicates that under the executed, iteration-matched setup random sampling provided speed gains while showing no detected QE-performance loss versus full sampling.

**Supplementary Figure S1. Disaggregated sampling comparison for `full`, `random`, and `hdsssom` (Hexagonal + MST).** Panel S1A shows the stacked panel layout; panel S1B shows the no-stack 4x3 layout.



For interpretation, pooled batch-family comparisons are treated as the primary deployment-oriented endpoint because they estimate the strongest batch-family configuration attainable under the executed budget. Branch-disaggregated results (`full_batch`-only and `minibatch`-only) are treated as conditional robustness analyses within the pooled TPE optimization process.

#### 5.1.1 Primary endpoint: Balanced QE

The primary endpoint is Balanced QE (mean of train and holdout QE). Figure 2 reports the algorithm comparison as a six-panel composite:

1. Panels A-C: dataset-wise paired-median forest plots for pooled batch modes, full-batch-only, and minibatch-only, respectively.
2. Panels D-F: top-$k$ global-sensitivity curves for the corresponding A-C comparisons.


Let $\Delta_k$ denote the global paired median percent improvement after retaining the top-$k$ trials per paired unit. Under a tail-only optimization-artifact hypothesis in a homogeneous comparator distribution, one expects attenuation of $\Delta_k$ as $k$ increases because progressively less-extreme order statistics enter the retained set. In pooled mixture settings, however, attenuation is not guaranteed: top-$k$ ordering is performed on the union of full-batch and minibatch outcomes, and pooled rank composition can remain stable or vary non-monotonically as branch contributions interleave.

Accordingly, interpretation uses both pooled and disaggregated curves. In Figure 2D-F, the pooled-branch sensitivity curve remains positive across the tested $k$ range and does not collapse as $k$ increases, which is inconsistent with a gain that exists only in an extreme-tail subset. In branch-disaggregated runs, the same directionality is maintained, supporting a robust colours advantage under the executed budget. Complementary per-metric sensitivity panels for holdout/train/distortion are provided in Supplementary Figure S2.

**Supplementary Figure S2. Per-metric sensitivity matrix (holdout/train/distortion; pooled/full/minibatch).**


#### 5.1.2 Split-wise QE reporting

`quantization_error_holdout` and `quantization_error_train` are reported in the supplementary split-wise outcome panels (Supplementary Figure S3A-S3B) rather than the main figure. In broad terms, train QE mirrors the Balanced-QE pattern, while holdout QE does not show an overall significant difference between colours and batch.

**Supplementary Figure S3. Split-wise algorithm outcomes under pooled `full+random` sampling.** Panel S3A reports holdout/train/distortion outcomes, panel S3B reports split-wise QE, and panel S3C reports distortion-focused comparisons.



Full-batch-only reporting is methodologically and operationally important. In many production workflows, users default to full batch for throughput predictability and implementation simplicity, rather than tuning minibatch size/chunking. Full-batch isolation therefore tests a realistic deployment choice. In these disaggregated comparisons, when full batch underperforms minibatch, the observed deficit remains approximately within $1\%$. This margin is practically important: it indicates that full batch remains close to branch-optimal performance, reduces the likelihood that pooled conclusions are driven by severe full-batch failures, and supports the interpretation that pooled batch-family estimates reflect complementary branch strengths rather than cross-branch contamination.

#### 5.1.3 Distortion context

We report distortion as a secondary endpoint for algorithm comparison only. In this study, the legacy distortion term is used to penalize parameter configurations that achieve low QE by fitting only the winning unit while leaving neighboring units less consistent with the local data manifold. We treat this as a neighborhood-coherence complement to QE rather than a replacement objective.

The distortion formulation is adapted for our evaluation pipeline from prior SOM-metrics literature [@forestSurveyImplementationPerformance2020a], with implementation details aligned to our benchmark setting.

At the global level, colours is not significantly worse on distortion. However, at per-dataset resolution across the ten sklearn datasets, colours is worse in $5/10$ datasets and better in $0/10$ (see Supplementary Figure S3A). Considering the QE gains on non-synthetic sklearn datasets (for example, `breast_cancer`, `wine`, and `digits`), users may reasonably judge that the QE advantage offsets this relative topological-distortion weakness for many deployment contexts.


### 5.2 MST Results

Topology comparisons are reported with QE-only endpoints because distortion is unavailable for MST in this setup.

#### 5.2.1 MST versus Hexagonal on Balanced QE

We use dataset-wise paired improvement summaries (MST over Hexagonal) with the same reporting logic as Section 5.1, centered on Balanced QE.

In the executed pooled-`full+random` topology comparison, MST is consistently better than matched hexagonal on QE endpoints for both `colors` and batch-family strata (Figure 3A-3C, with paired-effect directionality preserved across the corresponding sensitivity panels in Figure 3D-3F).

#### 5.2.2 Split-wise QE reporting

We report `quantization_error_holdout` and `quantization_error_train` separately for MST versus Hexagonal to expose train/holdout trade-offs. Figure 3 reports the topology composite as a 2x3 matrix: columns are Balanced QE / Holdout QE / Train QE; the top row shows paired-effect forest panels with pooled-batch and colours strata, and the bottom row shows matching top-$k$ sensitivity panels.


### 5.3 RNG Results

RNG is significantly better than both MST and hexagonal topologies on QE endpoints in the pooled-`full+random` topology comparisons, including Balanced QE and split-wise QE analyses.
<<<<<<< main
=======

Hexagonal-versus-RNG paired QE comparisons are reported in Figure 7, and MST-versus-RNG paired QE comparisons are reported in Figure 8.
>>>>>>> local

This gain is explained by RNG's ability to support both topology regimes within one model: mesh-like local organization where manifold structure is regular, and freer non-mesh connectivity where the manifold is irregular or partially disconnected. In contrast, hexagonal neighborhoods enforce a fixed mesh and MST enforces a strictly tree-constrained structure.

Because the RNG graph is recomputed over moving prototypes during topology refresh, the neighborhood structure can adaptively shift between these two regimes over training rather than remaining fixed. This adaptive behavior yields stronger quality than either single-regime alternative (MST or hexagonal) in the new results.
<<<<<<< main
=======

**Supplementary Figure S7. Representative topology overlays on the circles dataset under default benchmark settings.** Panels A-C show node-connection overlays for `hexagonal`, `mst`, and `rng`, respectively (`assets_manual/figures/supp_fig_topology_circles_representative_default.svg`).
>>>>>>> local

### 5.4 Multi-GPU + OOM Results

Speed results are presented from harmonized multi-run artifacts (merged across 1/2/3/4/8 GPU jobs) so that all scaling figures are generated from a single consistent result table per mode/topology/method. Figure 4A-C reports algorithm-first speedup scaling and Figure 4D-F reports the corresponding efficiency scaling; Figure 5A-C reports MST-to-Hex runtime ratios and Figure 5D-F reports RNG-to-Hex runtime ratios; Figure 6A-C reports random-vs-full sampling runtime ratios.

Speed benchmarking is reported on `full` sampling unless stated otherwise, in order to maximally test speed performance of the algorithms. One `random`-sampling speed benchmark is included to show the speed gains possible from random sampling (Figure 6A-C).

RNG runtime-ratio panels are interpreted jointly with the quality outcomes in Section 5.3.

#### 5.4.1 Figure 4: Algorithm-first GPU scaling (harmonized)

Primary purpose: quantify how processing algorithms scale with GPU count under matched topology settings (Figure 4A-F).

Figure structure (single figure, 2x3 panel grid):
1. Top row (A-C): speedup versus GPU count for `dimension_scaling`, `sample_scaling`, and `grid_size_scaling`.
2. Bottom row (D-F): scaling efficiency versus GPU count for the same three scaling modes, in the same order.

Construction scaffold:
1. Fix topology to `hexagonal` for this figure so algorithm effects are not conflated with topology effects.
2. Compare methods `batch`, `colors`, and `minibatch` using harmonized GPU counts $G \in \{1,2,3,4,8\}$.
3. Use speedup $S(G) = T(1)/T(G)$ and efficiency $E(G)=S(G)/G$.
4. For each scaling mode, aggregate across axis values (sample sizes, dimensions, or grid sizes) using a robust central tendency (median or geometric mean) and show uncertainty (IQR or bootstrap CI).

Reporting focus:
1. Relative scaling quality of `colors` vs `batch` vs `minibatch` (Figure 4A-C).
2. Whether scaling behavior is stable across the three workload axes (Figure 4A-C) and whether efficiency trends remain consistent across those axes (Figure 4D-F).
3. Any efficiency collapse at high GPU counts and its relation to loader/communication overhead (Figure 4D-F).

Platform variance caveat: on Gadi, `gpuvolta` nodes provide 4 GPUs per node [@nciGadiHpcSystems2026], so `G=1` and `G=2` measurements are sub-node allocations. In these runs, the primary source of Disk/CPU data-movement instability is same-node co-tenant traffic (other jobs sharing CPU, memory, and node-local I/O paths); shared-filesystem load can add secondary variance [@nciStorageSystems2026].

Caption:
**Figure 4. Harmonized multi-GPU scaling by processing algorithm.** Results are aggregated from harmonized runs over $G\in\{1,2,3,4,8\}$ GPUs with topology fixed to hexagonal. Panels 4A-4C report speedup $S(G)=T(1)/T(G)$ for dimension-scaling, sample-scaling, and grid-size-scaling workloads, respectively. Panels 4D-4F report the corresponding scaling efficiency $E(G)=S(G)/G$ in the same mode order. Curves compare `batch`, `colors`, and `minibatch`; central tendency and uncertainty summarize performance across workload-axis values within each mode. This figure isolates algorithmic scaling behavior from topology effects (Figure 4A-F).


**Supplementary Figure S4. Algorithm-first multi-GPU scaling with minibatch-inclusive harmonized panels.** This supplementary panel set mirrors Figure 4 but explicitly includes minibatch-inclusive harmonized diagnostics across dimension-, sample-, and grid-size-scaling modes (Supplementary Figure S4A-S4C), with matching efficiency panels (Supplementary Figure S4D-S4F).


#### 5.4.2 Figure 5: Topology speed comparison (harmonized)

Primary purpose: compare runtime behavior of `hexagonal`, `mst`, and `rng` across GPU counts and workload scales (Figure 5A-F).

Figure structure (single figure, 2x3 panel grid):
1. Row 1 (A-C): MST relative runtime to Hex, $R_{\mathrm{MST}} = T_{\mathrm{MST}} / T_{\mathrm{HEX}}$, for `sample_scaling`, `dimension_scaling`, and `grid_size_scaling`.
2. Row 2 (D-F): RNG relative runtime to Hex, $R_{\mathrm{RNG}} = T_{\mathrm{RNG}} / T_{\mathrm{HEX}}$, for the same three scaling modes.

Construction scaffold:
1. Fix algorithm to `minibatch` for this figure so topology effects are isolated from algorithmic update-policy effects.
2. Use harmonized GPU counts $G \in \{1,2,3,4,8\}$.
3. Plot each panel as a workload-axis by GPU-count map (or equivalent faceted line representation) of runtime ratios.
4. Interpret $R<1$ as faster than hexagonal, $R=1$ as parity, and $R>1$ as slower than hexagonal.

Reporting focus:
1. Whether MST and RNG maintain speed competitiveness with hexagonal as scale increases (Figure 5A-F).
2. Whether topology speed ranking changes with GPU count for MST (Figure 5A-C) and RNG (Figure 5D-F).
3. Whether topology differences are mode-dependent (sample vs dimension vs grid scaling) within MST-vs-Hex (Figure 5A-C) and RNG-vs-Hex (Figure 5D-F).

In the harmonized runtime-ratio panels, RNG exhibits relatively poorer scaling at large grid sizes (Figure 5F), consistent with added neighborhood-checking requirements in graph-topology processing.

Caption:
**Figure 5. Harmonized topology-speed comparison under minibatch processing.** Runtime is compared across `hexagonal`, `mst`, and `rng` using harmonized runs over $G\in\{1,2,3,4,8\}$ GPUs with algorithm fixed to `minibatch`. Panels 5A-5C show MST-to-Hex relative runtime $R_{\mathrm{MST}}=T_{\mathrm{MST}}/T_{\mathrm{HEX}}$ for sample-scaling, dimension-scaling, and grid-size-scaling workloads; panels 5D-5F show RNG-to-Hex $R_{\mathrm{RNG}}=T_{\mathrm{RNG}}/T_{\mathrm{HEX}}$ for the same modes. Values below 1 indicate a topology faster than hexagonal, 1 indicates parity, and values above 1 indicate slower runtime (Figure 5A-F).


**Supplementary Figure S5A. MST-vs-Hex performance with minibatch-inclusive harmonized panels.** This supplementary figure expands the topology-speed comparison with minibatch-inclusive harmonized panel construction (Supplementary Figure S5A, panels A-C).


**Supplementary Figure S5B. Per-topology GPU-scaling performance (MST top row, Hexagonal bottom row).** This supplementary figure reports topology-specific scaling performance by workload mode for MST (Supplementary Figure S5B, panels A-C) and Hexagonal (Supplementary Figure S5B, panels D-F).


#### 5.4.3 Figure 6: Sampling-speed comparison (Random/Full)

Primary purpose: isolate the runtime effect of sampling choice (`random` versus `full`) independent of quality metrics (Figure 6A-C).

Figure structure (single figure, 1x3 panel grid):
1. Panel A: sample-scaling runtime ratio $R_{\mathrm{random/full}}=T_{\mathrm{random}}/T_{\mathrm{full}}$ across GPU counts.
2. Panel B: dimension-scaling runtime ratio $R_{\mathrm{random/full}}$ across GPU counts.
3. Panel C: grid-size-scaling runtime ratio $R_{\mathrm{random/full}}$ across GPU counts.

Interpretation:
1. $R_{\mathrm{random/full}}<1$ indicates random sampling is faster than full sampling (Figure 6A-C).
2. $R_{\mathrm{random/full}}=1$ indicates runtime parity (Figure 6A-C).
3. $R_{\mathrm{random/full}}>1$ indicates random sampling is slower than full sampling (Figure 6A-C).

In the executed harmonized comparison, random sampling is observed to be faster overall than full sampling (Figure 6A-C).

Caption:
**Figure 6. Harmonized sampling-speed comparison (Random/Full runtime ratio).** Panels 6A-6C report runtime ratio $R_{\mathrm{random/full}}=T_{\mathrm{random}}/T_{\mathrm{full}}$ across sample-scaling, dimension-scaling, and grid-size-scaling workloads over harmonized GPU counts $G\in\{1,2,3,4,8\}$. Values below 1 indicate faster runtime under random sampling.


#### 5.4.4 Supplementary systems diagnostics (text/table support)

Alongside the supplementary scaling figures above, we use log-derived diagnostics as supporting evidence (not additional main figures):
1. staging mode/strategy and worker staging throughput summaries,
2. per-iteration timing breakdown (submit/get/collective components),
3. stability notes for OOM-avoidance behavior under largest workloads.

## 6. Discussion

### 6.1 Practical framing

This manuscript reports deployment-oriented conclusions under fixed compute budgets. The discussion follows the same claim order used in the text: `C1` sampling (`random` vs `full`), `C2` algorithm (`colors` vs batch-family), `C3` topology (MST vs hexagonal), and `C4` systems scaling. The algorithm comparator remains family-level: `colors` is compared to the strongest batch-family outcome found by pooled TPE search, where `full_batch` versus `minibatch` branch choice is optimized internally.

The external XPySOM calibration (Supplementary Figure S6; Supplementary Table S1) serves as a directionality sanity check, while the main inferential claim is still anchored to the paired within-FloatSOM analyses.

### 6.2 Sampling tradeoff (`random` versus `full`)

The `random` versus `full` result is operationally important and was initially unexpected: `random` is faster in harmonized timing comparisons (Figure 6A-6C) while showing no detected QE loss in the iteration-matched Optuna setup (Section 4.3; Figure 1; Supplementary Figure S1A-S1B). Because trial-level training iterations are fixed in the executed QE pipeline, this comparison is on equal iteration budgets rather than unequal training lengths. One plausible explanation is that repeated random draws still expose most of the dataset over training while reducing per-iteration cost; this differs from more aggressive random-subsampling schemes in prior work [@liuRobustPhenotypingHighly2023]. A second possible contributor is mild stochastic regularization from subset turnover, though the exact mechanism is not resolved here.

### 6.3 Algorithm tradeoff (`colors` versus batch-family)

The main tradeoff is quality versus runtime. `colors` is slower than batch-family baselines in harmonized runtime scaling (Figure 4A-4C; Supplementary Figure S4A-S4C). This runtime penalty is more pronounced after entering disk-backed mode: compared with batch, colours requires two disk-read passes, so post-transition timing behavior is not expected to remain linearly matched to batch. However, QE endpoints in the executed Optuna campaign show stronger quality behavior for `colors` on non-synthetic sklearn datasets where effects are present (notably `breast_cancer`, `wine`, and `digits`; Figure 2A-2C, with supporting split/outcome panels in Supplementary Figure S2 and Supplementary Figure S3A-S3B). Under QE-prioritized deployment settings, that quality gain can justify the runtime penalty.

### 6.4 Topology tradeoff (MST and RNG)

MST remains a favorable quality-time trade relative to hexagonal in this study: on QE, MST is consistently better than matched hexagonal across both `colors` and batch-family strata (Figure 3A-3C), with the same directionality retained in sensitivity panels (Figure 3D-3F). The cost is runtime: MST is slightly slower than hexagonal in topology-speed comparisons (Figure 5A-5C; Supplementary Figure S5A, panels A-C).

<<<<<<< main
RNG now provides the strongest quality outcome, with significant QE improvements over both MST and hexagonal. The central reason is representational flexibility: RNG can express mesh-like neighborhoods in locally regular regions while also supporting freer non-mesh connectivity in irregular or partially disconnected regions, and it can adaptively move between these regimes as prototype geometry changes during training.
=======
RNG now provides the strongest quality outcome, with significant QE improvements over both MST and hexagonal (Figure 7 and Figure 8). The central reason is representational flexibility: RNG can express mesh-like neighborhoods in locally regular regions while also supporting freer non-mesh connectivity in irregular or partially disconnected regions, and it can adaptively move between these regimes as prototype geometry changes during training.

This flexibility comes with a scaling cost in edge construction. For RNG, each candidate pair $(i,j)$ can require checking all potential blocker nodes $k\neq i,j$, yielding worst-case blocker checks:
$$
\binom{N}{2}(N-2)=\frac{N(N-1)(N-2)}{2}=\Theta(N^3).
$$
By comparison, MST edge ranking on the complete graph starts from $\binom{N}{2}=\Theta(N^2)$ pair distances before tree extraction, while hexagonal topology uses fixed local degree (approximately $|E|\approx 3N$ in planar 2D grids), i.e. linear-scale adjacency structure. Practically, RNG is therefore a strong quality option at moderate node counts, but it is not the preferred topology when extremely large node counts are required.
>>>>>>> local

### 6.5 Systems implications and limits

Systems trends across Figure 4A-4F, Figure 5A-5F, and Figure 6A-6C indicate practical scaling effects beyond simple theoretical GPU speedup. At larger workloads, additional GPUs increase aggregate memory capacity, allowing larger working sets to remain resident and reducing memory-pressure penalties (Figure 4D-4F). For disk-backed regimes, runtime is more I/O-bound, so spreading data over more nodes can reduce per-node I/O contention and produce better-than-expected end-to-end gains when storage parallelism is available.
This is particularly relevant for the 1- and 2-GPU points: on Gadi 4-GPU nodes, these are sub-node calls and therefore the observed transfer deviations are primarily attributable to other traffic on the same node, with additional variance possible from shared-filesystem load.

Branch-disaggregated and top-$k$ analyses refine, rather than replace, the pooled conclusion. The pattern that minibatch often dominates on real-dataset settings while full batch can dominate topology-focused settings indicates complementary branch strengths, and the practical full-batch penalty remains small (approximately within $1\%$ where full batch is worse). The non-collapsing disaggregated sensitivity curves with increasing $k$ (Figure 2D-2F) support a method-level `colors` advantage under the executed $60{,}000$-trial campaign, while still motivating future equal-budget, branch-fixed confirmation.

## 7. Conclusion

_TBD_

## References

[@forestSurveyImplementationPerformance2020a]

## Figures (End Matter)

![Figure 1: Main sampling comparison on Balanced QE (Hexagonal / pooled batch): full vs random, full vs HDSSOM, and random vs HDSSOM.](assets_manual/figures/fig_sampling_hex_batch_balanced_qe_main.svg)

![Figure 2: Primary six-panel algorithm comparison on Balanced QE (pooled/full/minibatch outcomes + sensitivity).](assets_manual/figures/fig_sampling_hex_batch_balanced_qe_pooled_full_random.svg)

![Figure 3: MST-vs-Hex topology comparison across QE metrics (full+random pooled sampling).](assets_manual/figures/fig_topology_mst_hex_metrics_pooled_full_random.svg)

![Figure 4: Algorithm-first multi-GPU scaling (harmonized, hexagonal topology).](assets_manual/figures/fig_algorithm_first_gpu_scaling_harmonized.svg)

![Figure 5: Topology runtime ratios (MST/Hex and RNG/Hex) under minibatch processing.](assets_manual/figures/fig_topology_runtime_ratios_harmonized.svg)

![Figure 6: Sampling-speed comparison (Random/Full runtime ratio) across sample, dimension, and grid-size scaling.](assets_manual/figures/fig_sampling_speed_random_full_harmonized.svg)

<<<<<<< main
=======
![Figure 7: Hexagonal-vs-RNG topology comparison across QE metrics (full+random pooled sampling).](assets_manual/figures/fig_topology_hex_rng_metrics_pooled_full_random.svg)

![Figure 8: MST-vs-RNG topology comparison across QE metrics (full+random pooled sampling).](assets_manual/figures/fig_topology_mst_rng_metrics_pooled_full_random.svg)

>>>>>>> local
![Supplementary Figure S1A: Disaggregated sampling comparison (stacked layout).](assets_manual/figures/supp_fig_sampling_mode_disaggregated_hex_mst_balanced_qe_raw.svg)

![Supplementary Figure S1B: Disaggregated sampling comparison (no-stack 4x3 layout).](assets_manual/figures/supp_fig_sampling_mode_disaggregated_no_stack_hex_mst_balanced_qe_raw.svg)

![Supplementary Figure S2: Per-metric sensitivity matrix (holdout/train/distortion; pooled/full/minibatch).](assets_manual/figures/supp_fig_sampling_hex_batch_sensitivity_pooled_full_random.svg)

![Supplementary Figure S3A: Outcomes matrix for holdout/train/distortion (pooled/full/minibatch).](assets_manual/figures/supp_fig_sampling_hex_batch_outcomes_pooled_full_random.svg)

![Supplementary Figure S3B: Split-wise QE matrix for pooled full+random sampling.](assets_manual/figures/supp_fig_sampling_hex_batch_splitwise_qe_pooled_full_random.svg)

![Supplementary Figure S3C: Distortion-focused comparison across pooled/full/minibatch settings.](assets_manual/figures/supp_fig_sampling_hex_batch_distortion_pooled_full_random.svg)

![Supplementary Figure S4: Algorithm-first multi-GPU scaling with minibatch-inclusive harmonized panels.](assets_manual/figures/supp_fig_algorithm_first_gpu_scaling_harmonized_with_minibatch.svg)

![Supplementary Figure S5A: MST-vs-Hex performance with minibatch-inclusive harmonized panels.](assets_manual/figures/supp_fig_topology_mst_hex_performance_harmonized_with_minibatch.svg)

![Supplementary Figure S5B: Per-topology GPU-scaling performance (MST top, Hexagonal bottom).](assets_manual/figures/supp_fig_topology_mst_hex_gpu_scaling_harmonized.svg)

![Supplementary Figure S6: XPySOM calibration on matched QE settings (hexagonal only).](assets_manual/figures/supp_fig_xpysom_calibration_qe_hexagonal.svg)
<<<<<<< main
=======

![Supplementary Figure S7: Representative circles overlays at default settings for Hexagonal (A), MST (B), and RNG (C).](assets_manual/figures/supp_fig_topology_circles_representative_default.svg)
>>>>>>> local

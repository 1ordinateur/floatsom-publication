# Floatsom Paper

## Title

FloatSOM: Topology-Flexible, Out-of-Memory (OOM)-Capable Self-Organizing Maps for Multiple Graphics Processing Units (GPUs)

## Authors

Anonymous Authors

## Abstract

GPU-accelerated Self-Organizing Map (SOM) implementations are among the most competitive options for large-scale SOM analysis, but growing dataset sizes increasingly challenge their practical use because workloads no longer fit cleanly within device-memory limits. We introduce FloatSOM, a GPU-oriented SOM framework for scalable training and deployment that supports multi-GPU execution, out-of-memory disk-backed streaming, and the implementation of novel topologies beyond regular lattices. We evaluate FloatSOM on 14 synthetic and real benchmark datasets together with controlled speed-scaling benchmarks, and show that these improved topologies, combined with topology-aware hyperparameter fine-tuning, yield lower quantization error than current state-of-the-art SOM baselines. FloatSOM also sustains this performance at large scale with high-throughput distributed execution; in the largest benchmark, it trains a 1024-node SOM network on 1,000,000,000 samples with 50 features in 6.16 minutes on 8 GPUs across two separate high-performance-computing nodes.

## 1. Introduction

Self-Organizing Maps (SOMs), originally introduced by Kohonen [@kohonenSelforganizingMap1990], are an unsupervised machine-learning method that uses competitive learning to organize nodes such that they capture the topology of the data. In practice, this topology-preserving representation means SOMs are commonly used to map dataset topology, produce dimensionality-reduced visualizations, and conduct clustering at scale [@kangasVariantsSelforganizingMaps1990]. As dataset size and heterogeneity increase, the computational requirements of SOMs grow in both time and memory. Many current implementations, however, remain constrained to single-device workloads that must fit within video random-access memory (VRAM), with limited support for distributed compute, out-of-core execution, and modern GPU orchestration.

SOM topology presents a second limitation. Classical SOMs are traditionally trained on regular rectangular or hexagonal lattices because fixed grids make neighborhood definition, visualization, and optimization straightforward. However, these regular lattices also impose a strong geometric prior on the learned representation. Accordingly, prior work on dynamic, growing, and graph-structured SOM variants reflects a long-standing recognition that fixed lattices are not always the best match for irregular data geometry [@alahakoonDynamicSelforganizingMaps2000; @vasighiDirectedBatchGrowing2017; @kangasVariantsSelforganizingMaps1990]. However, these alternatives have generally not been developed or evaluated in the high-throughput, large-sample regime targeted by modern GPU-enabled applications and are broadly impractical for deployment at scale.

FloatSOM is designed to address these combined systems and topology limitations. Within FloatSOM, we implement distributed multi-GPU execution, out-of-memory disk-backed streaming, and scalable topology-flexible training beyond standard fixed lattices. Specifically, we implement distributed-compute-compatible minimum-spanning-tree (MST) and relative-neighborhood-graph (RNG) topologies. We also evaluate multiple sampling strategies as a route to further computational acceleration and derive fine-tuned hyperparameter configurations across diverse datasets and recommended operating regimes. FloatSOM is suitable for both high-performance-computing (HPC) operation and consumer-grade desktop GPUs.

## 2. Related Work

Related work on practical SOM deployment spans software implementations, sampling-efficient training, topology design, and model selection.

### 2.1 Open-Source SOM Implementations and Systems

Open SOM implementations range from lightweight libraries to more performance-oriented systems. MiniSom is a compact Python implementation of the classical online regime [@vettigliJustGlowingMinisom2018], whereas XPySOM is a Python-based batch SOM implementation designed for efficient GPU-backed execution [@manciniXPySomHighPerformanceSelfOrganizing2020]. At larger scale, Somoclu and GigaSOM provide mature parallel SOM systems for large workloads [@wittekSomocluEfficientParallel2017; @kratochvilGigaSOMjlHighperformanceClustering2020]. However, an important systems gap remains: XPySOM is limited to a single GPU and requires the full dataset to fit in VRAM, while GigaSOM emphasizes distributed CPU execution rather than distributed GPU training within a broadly used Python workflow. FloatSOM is intended to address that gap.

### 2.2 Sampling Methodologies for SOM Training

Classical online and batch SOM training schemes traditionally sample every data-point in every training iteration [@kohonenSelfOrganizingMaps2001; @liuRobustPhenotypingHighly2023]. Consequently, one obvious route to improving SOM speed and scalability is subsampling, which reduces the amount of data used to train the final SOM. Note that this sampling question is separate from the online-versus-batch distinction: a SOM can use full or sampled data under either update regime, whereas online and batch refer to how updates are accumulated and applied.

To date, numerous subsampling strategies have been proposed. Beyond naive random sampling, guided methods include hierarchical dynamic subset selection SOM (HDSSSOM), which concentrates computation on difficult or stale regions of the data [@wetmoreSpeedingSelfOrganizingFeature2005], and adaptive sampling strategies for design-space exploration [@itoDesignSpaceExploration2016]. However, systematically benchmarked and openly maintained implementations that compare full-data, random, and guided sampling within a modern high-performance SOM workflow remain limited.

### 2.3 SOM Lattice and Adaptive Topologies

Most practical SOM implementations retain regular rectangular or hexagonal lattices because they simplify neighborhood indexing, visualization, and vectorized updates [@kohonenSelforganizingMap1990; @kohonenEssentialsSelforganizingMap2013]. In comparison to rectangular lattices, hexagonal lattices are often preferred in the literature because their neighborhood geometry is more isotropic, reduces directional bias, and produces more accurate results [@whiteTopologyMattersNetwork2008; @kohonenEssentialsSelforganizingMap2013; @forestSurveyImplementationPerformance2020]. Nevertheless, fixed lattice topologies still assume that the data are best represented by a fixed mesh structure [@whiteTopologyMattersNetwork2008].

The SOM literature has explored alternatives to fixed lattices, noting that realistic data distributions often do not map cleanly onto fixed meshes. These alternatives include dynamic maps that change their configuration, node number, or connections throughout the training cycle, such as DBGSOM and AMSOM [@vasighiDirectedBatchGrowing2017; @spanakisAMSOMAdaptiveMoving2016]. Graph-structured neighborhoods have also been proposed, including minimum spanning tree formulations in early SOM work [@kangasVariantsSelforganizingMaps1990] and later smaller-scale MST-based analyses [@jangUseMinimalSpanning2009]. However, these modified SOM topologies have not been assessed on large-scale datasets and do not have implementations that are either publicly available or suitable for distributed GPU computation. Indeed, none of the distributed or current GPU SOM implementations support these non-lattice-based topologies.

Relative Neighborhood Graphs (RNGs) [@toussaintRelativeNeighbourhoodGraph1980] are of particular interest in this work; we return to their full rationale and implementation details in Section 3.2.2.

### 2.4 Hyperparameter Optimization and Fair Comparison

SOM performance depends strongly on choices such as map size, initialization, learning-rate schedule, and neighborhood schedule. Prior work shows that these choices can substantially affect observed performance [@akindukoSOMStochasticInitialization2016; @forestSurveyImplementationPerformance2020]. This makes comparisons based only on untuned defaults difficult to interpret.

More generally, modern machine-learning workflows increasingly rely on automated hyperparameter optimization rather than manual tuning alone. Frameworks such as Optuna provide bounded search over large hyperparameter spaces and can support multi-objective optimization, allowing parameter settings to be selected with respect to several benchmark criteria simultaneously rather than collapsed into a single score [@akibaOptunaNextgenerationHyperparameter2019].

## 3. Methods

FloatSOM combines four methodological components: sample selection, topology definition, batch SOM updates, and the compute framework used to execute them. The framework supports `random`, `full`, and `HDSSSOM` sampling; rectangular, hexagonal, MST, and RNG topologies; and execution modes ranging from local GPU training to single-node and multi-node multi-GPU operation. Hyperparameter optimization is treated as a separate layer applied across these configurations. Figure 1 summarizes this design.

![Figure 1](assets_manual/figures/fig_1.svg){.half-width width=50%}

*Figure 1. Schematic overview of the FloatSOM methods framing used in this manuscript. Sample selection and topology definition have configurable components that feed into the standard batch SOM training step, while the compute framework determines how that same training procedure is executed in practice. The options shown here summarize the FloatSOM configurations discussed in the following subsections.*

### 3.1 Sampling Selector Mathematics

This section formalizes the sampling policies evaluated in this work.

Let the full dataset be $X=\{x_i\}_{i=1}^{N}$. Let $m$ denote the number of samples presented to the SOM in a given training iteration, or the 'sampling budget'. This budget is either fixed directly or determined as a proportion $\rho$ of the dataset:

$$
m=
\begin{cases}
m_0, & \text{if a fixed budget is specified},\\[4pt]
\max\!\left(1,\left\lfloor N\rho \right\rfloor\right), & \text{if a dataset proportion is used}.
\end{cases}
\tag{1}
$$

For index sets $\mathcal{I}_t^{(s)}$ at iteration $t$, the implemented full and random selectors are:

$$
\mathcal{I}_t^{\mathrm{full}}=\{1,\dots,N\},
\qquad
\mathcal{I}_t^{\mathrm{random}} \sim \mathrm{Unif}\!\left(\left\{I\subseteq\{1,\dots,N\}:|I|=m\right\}\right),\ m<N.
\tag{2}
$$

If $m\ge N$, random returns the full dataset (no subsampling). The selected training batch is $X_t^{(s)}=\{x_i:i\in\mathcal{I}_t^{(s)}\}$.

For random subsampling, the per-iteration objective is aligned in expectation with full-data risk:
$$
\mathbb{E}_{\mathcal{I}_t^{\mathrm{random}}}\!\left[\frac{1}{m}\sum_{i\in\mathcal{I}_t^{\mathrm{random}}}\ell(x_i;W)\right]=\frac{1}{N}\sum_{i=1}^{N}\ell(x_i;W).
\tag{3}
$$

For hierarchical dynamic subset selection SOM (HDSSSOM) [@wetmoreSpeedingSelfOrganizingFeature2005], we retain the core published algorithm and re-implement it in a multi-GPU-compatible form as an adaptive sampler that preferentially revisits difficult, under-trained, or stale samples.

### 3.2 Topology Definition

We next define how neighborhood structure is assigned in FloatSOM across regular-lattice and graph-based configurations.

After initialization, neighborhood relations are defined by the selected topology. For regular-lattice baselines, we support both grid and hexagonal layouts, with hexagonal as the standard topology reference in this manuscript based on prior SOM guidance. We implement the hexagonal lattice topology in accordance with [@vettigliJustGlowingMinisom2018]. For MST and RNG, neighborhood structure is derived from the current prototype geometry using the methodologies below.

Note that regardless of the selected topology, FloatSOM uses the same prototype initialization options. Initialization determines only the starting prototype values; neighborhood relations are applied afterward according to the selected topology to establish node connections, keeping the initial state comparable across regular-lattice and graph-based runs.

#### 3.2.1 MST Topology Implementation

Considering the lack of suitable MST implementations, we have developed our own. The MST topology replaces fixed lattice neighborhood distance with graph shortest-path distance on a minimum spanning tree built from current prototypes. For $P$ prototype nodes in feature dimension $d$, the pairwise prototype matrix is formed with the standard squared-distance Gram identity, which avoids 3D broadcast tensors and preserves $O(P^2 d)$ dense linear-algebra structure.

After distance construction, we build a minimum spanning tree over the prototypes and use shortest-path distances on that tree to evaluate the Gaussian neighborhood influence during learning [@kruskalShortestSpanningSubtree1956]. Topology-derived influence matrices are cached and refreshed at fixed or progress-adaptive intervals.

If the current iteration does not trigger recomputation, the previous graph state and cached influences are reused. Otherwise, pairwise prototype distances are rebuilt on GPU, the MST is recomputed on CPU, graph distances are refreshed in chunked GPU fashion, and the influence cache is rebuilt for the deduplicated active radii.

```text
Input: prototypes W_t, iteration t, topology-refresh policy
Output: topology state (E_t, g_t, cached influences)
1: Query the refresh policy for the current iteration
2: if no topology refresh is due then
3:     return previous topology state
4: end if
5: Compute pairwise squared prototype distances on GPU
6: Transfer distances to CPU and run Kruskal to obtain MST edges E_t
7: Build adjacency from E_t
8: Compute all-pairs graph distances g_t with chunked GPU Floyd-Warshall
9: Deduplicate active radii and rebuild/update cached influence maps
10: Commit E_t, g_t, and cache state
11: return topology state
```
*Algorithm 3. Dynamic MST topology update with refresh-triggered recomputation and cached influence reuse.*

#### 3.2.2 RNG Topology Implementation

RNG is our second topology contribution. Our RNG topology constructs a Relative Neighborhood Graph over current prototype distances using the standard RNG criterion [@toussaintRelativeNeighbourhoodGraph1980], and then reuses the MST infrastructure for shortest-path precomputation, radius-deduplicated influence caching, and dynamic update scheduling.

Relative Neighborhood Graphs are less constrained than MSTs because they are not restricted to a single spanning-tree backbone with exactly one route between connected prototypes. Instead, when local geometric evidence supports multiple neighborhood relations, RNG can retain those connections rather than forcing the structure through only one edge choice per region. Consequently, we hypothesize that this added flexibility will permit more faithful recovery of real data-local connections and, as a consequence, a superior topology relative to MST.

We implement the RNG topology by evaluating candidate elimination in chunks to control memory pressure while preserving the direct strict blocker test. No post-hoc connectivity repair is applied after edge extraction; the topology is defined entirely by the canonical RNG criterion.

Under this refresh policy, the RNG path follows the same overall structure as MST, but replaces tree construction with chunked relative-neighborhood edge extraction. Like the MST, if the current iteration does not trigger recomputation, the previous graph state and cached influences are reused; otherwise we recompute the RNG and influence cache.

```text
Input: prototypes W_t, iteration t, topology-refresh policy
Output: topology state (E_t, g_t, cached influences)
1: Query the refresh policy for the current iteration
2: if no topology refresh is due then
3:     return previous topology state
4: end if
5: Compute pairwise squared prototype distances on GPU
6: Evaluate RNG candidate elimination in chunks using the blocker test
7: Retain surviving RNG edges E_t and build adjacency
8: Compute all-pairs graph distances g_t with chunked GPU Floyd-Warshall
9: Deduplicate active radii and rebuild/update cached influence maps
10: Commit E_t, g_t, and cache state
11: return topology state
```
*Algorithm 4. Dynamic RNG topology update with refresh-triggered recomputation and cached influence reuse.*

### 3.3 Multi-GPU + OOM Methodology and Implementation

This section covers how we distribute computation across GPUs, how data are streamed for large workloads, and how memory safeguards preserve progress under high-pressure regimes.

![Figure 2](assets_manual/figures/fig_2.svg){.half-width width=50%}

*Figure 2. Multi-GPU data-loader and NVIDIA Collective Communications Library (NCCL) synchronization schematic. Disk-backed path: data are distributed from shared storage to worker-local shards on node-local storage, read in worker-local chunks (`n_chunks`) into pinned host memory, transferred to GPU while loading overlaps with compute, processed as chunked local BMU/update steps, synchronized by NCCL all-reduce, and normalized into one weight update per iteration. RAM path: random-access memory (RAM) mode, where data are sharded directly into each worker's GPU-local CPU RAM and follow the same pinned-memory-to-GPU path, without requiring disk reads; this path is typically faster when data are already memory-resident because disk-read overhead is removed.*

#### 3.3.1 General Multi-GPU Logic

Distributed execution uses Ray actors with one GPU per worker and NCCL collectives for synchronous aggregation [@moritzRayDistributedFramework2018]. At each iteration, every worker processes its assigned shard locally in `n_chunks`, accumulates worker-local statistics across those chunks, and synchronizes once per iteration.

For worker $g \in \{1,\dots,G\}$, let $X_{t,g}^{(s)}$ denote the worker-local shard of the selected iteration batch $X_t^{(s)}$. Let $j$ index prototype nodes, let $w_j^{(t)}$ denote the prototype vector of node $j$ at iteration $t$, let $b(x)$ denote the best-matching unit (BMU) of sample $x$ under the current weights, let $h_{j,b(x)}^{(t)}$ denote the iteration-$t$ neighborhood influence between node $j$ and the BMU of $x$, and let $\eta_t$ denote the learning rate at iteration $t$. The local accumulators are:

$$
\begin{aligned}
 U_j^{(g)} &= \sum_{x \in X_{t,g}^{(s)}} \eta_t\,h_{j,b(x)}^{(t)}(x-w_j^{(t)}), \\
 H_j^{(g)} &= \sum_{x \in X_{t,g}^{(s)}} h_{j,b(x)}^{(t)}.
\end{aligned}
\tag{4}
$$

Global synchronized accumulators are:

$$
\begin{aligned}
U_j &= \sum_{g=1}^{G} U_j^{(g)}, \\
H_j &= \sum_{g=1}^{G} H_j^{(g)}.
\end{aligned}
\tag{5}
$$

In implementation, $U_j^{(g)}$ and $H_j^{(g)}$ are accumulated across the worker's `n_chunks` and synchronized once per iteration. Normalization and momentum are then applied with globally consistent denominators.

#### 3.3.2 Multi-GPU Implementation Details

As shown in Fig. 2, each worker uses a chunked loading path from CPU memory to GPU memory. In streaming mode, data are distributed to worker-local disk shards and then read chunk-by-chunk into pinned host memory before transfer to GPU; in RAM mode, data are pre-sharded into worker-local CPU RAM and follow the same pinned-memory path without disk reads. FloatSOM overlaps data transfer and compute across CUDA streams, uses JIT-compiled kernels for the core BMU and update steps, and synchronizes worker-local accumulators once per iteration with NCCL all-reduce. Weights therefore remain resident on worker GPUs across iterations.

#### 3.3.3 OOM-Capable Topology Updates

Additional larger-than-memory support for topology updates is provided through topological chunking. Topological chunking applies the same idea to topology-side computations within each worker. When graph-distance or influence structures would otherwise exceed a worker's VRAM, those computations are tiled and evaluated in bounded pieces rather than materialized at once. This topology-side chunking is not depicted in the Fig. 2 data path, but it follows the same per-worker bounded-memory execution rule.

### 3.4 Multi-Objective Hyperparameter Optimization

We use Optuna as an automated multi-objective hyperparameter optimization framework to derive near-optimal performance and corresponding hyperparameters for each FloatSOM configuration under a given sampling, topology, and processing combination [@akibaOptunaNextgenerationHyperparameter2019]. In this setting, multi-objective optimization means searching for parameter configurations that jointly balance the selected benchmark objectives rather than optimizing a single scalar criterion.

## 4. Experimental Setup

We use two benchmark protocols: an Optuna quality benchmark and a speed-scaling benchmark. The first evaluates algorithmic quality and tuned attainable performance, and the second evaluates runtime and distributed scaling behavior. All production Optuna and speed benchmarks reported in this manuscript were executed on Gadi at the National Computational Infrastructure (NCI), Australia, on gpuvolta nodes [@HPCSystemsNCI], using a consistent multi-GPU environment across runs.

### 4.1 Optuna benchmark protocol

We use Optuna-based multi-objective optimization to determine the best attainable performance and corresponding hyperparameters for each sampling (full and random) and topology (hexagonal, MST, and RNG) combination, optimizing jointly for train and holdout quantization error ($QE_T$ and $QE_H$). Each dataset-topology-sampling configuration is optimized for 200 trials across 10 seeds under variant-appropriate search constraints. These runs use the standard in-memory batch path rather than the Ray-distributed execution stack. A separate focused HDSSSOM pilot is reported in Section 5.2 and summarized in Supplementary Table S2. This corresponds to:

$$
14_{\text{Datasets}} \times 10_{\text{Seeds}} \times 3_{\text{Topologies}} \times 2_{\text{SamplingMethods}} \times 200_{\text{Trials}}= 168{,}000 \text{ Runs}
$$

Sampling comparisons in Section 5.2 compare full-vs-random paired analyses pooled across all topologies, with the HDSSSOM pilot also including the HDSSSOM sampling methodology. Topology comparisons in Sections 5.3-5.4 use full-sampling runs across hexagonal, MST, and RNG. Unless explicitly stated otherwise, the remaining analyses reported in this manuscript use full sampling with full-batch training.

#### 4.1.1 Optuna benchmark datasets and preprocessing

The Optuna benchmark uses 14 synthetic and real scikit-learn datasets spanning low- and high-dimensional regimes [@pedregosaScikitlearnMachineLearning2011]; dataset metadata are listed in Supplementary Table S1. Synthetic datasets are generated according to the random seed, whereas real datasets are loaded directly from sklearn. Across the Optuna protocol, inputs are standardized feature-wise to zero mean and unit variance using `StandardScaler`, then deterministically permuted and split into fixed 70/30 train-holdout partitions. We retain both partitions to evaluate both training-set representation and transfer to unseen samples.

#### 4.1.2 Optuna quality metrics

Our primary quality metric is Quantization Error ($QE$) [@kohonenSelfOrganizingMaps2001]. We report both train and holdout $QE$, denoted $QE_T$ and $QE_H$, respectively. Here, $QE_T$ captures use cases where the full observed population is available and the map is intended to represent that same population, while $QE_H$ captures generalization settings where the trained SOM is projected onto previously unseen samples.

Balanced $QE$, denoted $QE_B$, is defined as the mean of $QE_T$ and $QE_H$. $QE_B$ is therefore a composite endpoint that weights representation fidelity (train) and transfer-to-unseen-data fidelity (holdout) equally. Note that in the Optuna runs, $QE_T$ and $QE_H$ are optimized jointly as a two-objective vector, with $QE_B$ only calculated *post hoc*. 

### 4.2 Speed-scaling benchmark protocol

The speed-scaling benchmark evaluates runtime and distributed scaling behavior in different compute and algorithm configurations. Speed scaling is evaluated with runs across $G\in\{1,2,4,8\}$ GPUs under a series of fixed scaling protocols. Runtime summaries are computed from repeated executions per configuration and reported as both absolute training time and efficiency ratios relative to a 1-GPU comparison. These scaling and multi-GPU/OOM-capable runs use the Ray-orchestrated distributed execution layer built on top of the standard FloatSOM training path [@moritzRayDistributedFramework2018]. Accordingly, the scaling figures in Sections 6.1-6.2 and the runtime/scaling comparison reported later against XPySOM should be interpreted as distributed-execution results rather than the in-memory Optuna path.

For scaling-efficiency calculations, when a single-GPU baseline was missing at a given axis value due to timeout, we estimated that baseline by local linear extrapolation from the last available 1-GPU point on the same curve. That is, runtime was assumed to scale proportionally with the axis variable for the extrapolation step (for example, doubling sample count or doubling dimensionality doubles the estimated 1-GPU runtime).

Topology-speed comparisons include hexagonal, MST, and RNG, with harmonized workload settings so ratios isolate topology-associated runtime effects. A dedicated $G\in\{1,2,4\}$ batch-mode random-versus-full comparison is additionally included to isolate sampling-specific runtime effects independently of the multi-GPU scaling runs. 

#### 4.2.1 Scaling benchmark datasets

The speed benchmark uses synthetic random matrices with uniform values in $[0,1]$. No train-holdout split was used here because the objective is runtime rather than generalization, so all generated data are used for training. We use a shared default scaling dataset and then vary one workload axis at a time. The shared default configuration is $10^7$ samples, 50 dimensions, a $32 \times 32$ SOM grid, 10 training iterations.

From this shared default, we generate three different scale benchmarking datasets. Sample scaling varies the sample count over $\{10^6, 5 \times 10^6, 10^7, 5 \times 10^7, 10^8, 5 \times 10^8, 10^9\}$. Dimension scaling varies the feature dimension over $\{50, 100, 200, 500, 1000, 2000, 5000\}$. Grid-size scaling varies the hexagonal grid side length over $\{8, 16, 24, 32, 48, 64\}$; equivalently, the number of SOM nodes in this panel is $g^2$ for grid side length $g$.

Additional CPU and RAM resources attached to each GPU are scaled linearly with GPU count while keeping the software environment and benchmark procedure consistent across runs. For scaling benchmarks, each run was timed out if it exceeded 30 minutes of wall time. Per-GPU resource allocation was fixed at one NVIDIA V100 (32 GB VRAM), 12 CPU cores, 90 GB system RAM, with one full node comprising 4 GPUs and their associated CPUs, and with 400 GB of associated local disk storage. 

### 4.3 Hyperparameter Tuning and Stability

To quantify parameter-tuning benefit, we performed an explicit paired analysis between a tuned configuration and an untuned reference on seed-preserving Optuna exports. For each sampling-mode and topology combination, we first extracted the parameter settings from the best-performing Optuna runs under the benchmark objective for that combination. We then collapsed these per-seed best-performing tuned settings into deployable default configurations by taking the mean of numeric parameters and the mode of categorical parameters.

#### 4.3.1 Tuned Configuration versus Untuned Reference Analysis

This fixed derived configuration was then rerun and paired against the untuned default XPySOM (hexagonal) reference within matched dataset, seed, and dataset-split units.

Dataset-wise tuned-configuration-versus-untuned-reference summaries use the same forest-plot convention. The global overall summary pools all matched tuned-configuration-versus-untuned-reference pairs across datasets and applies the same paired $t$-test and confidence-interval construction to that pooled paired set.

#### 4.3.2 Hyperparameter stability and dataset-type stratification

An additional aim was to determine whether the topology and sampling strategies considered here differ in hyperparameter stability. In this context, stability refers to the extent to which similar high-performing hyperparameter settings are replicated across repeated runs and matched experimental conditions, such that a single parameter configuration can be applied with confidence across a broad range of scenarios while still yielding good performance. Greater stability is practically important because it reduces the need for user hyperparameter retuning and increases confidence in the robustness of the resulting map quality.

Hyperparameter stability was evaluated from the top-ranked tuned Optuna trial selected separately for each set. Stability was then quantified as the variation in those selected hyperparameter values across seeds within each topology. 

For numeric parameters, stability was measured across within-topology seed pairs using the relative difference:

$$
\frac{|a-b|}{\max(|a|,|b|,\varepsilon)}
$$

Where $a$ and $b$ are the values of a given numeric parameter for the two compared seeds, with a small $\varepsilon=10^{-12}$. These relative differences were then averaged over the compared numeric parameters; lower values indicate higher stability. For categorical parameters, stability was defined as the mismatch rate of the compared categorical parameters across the same seed pairs, again with lower values indicating higher stability. Topology comparisons report both per-parameter stability scores and an equal-weight overall summary.

For dataset-type stratification, we use the same synthetic/real group definitions introduced in Section 4.1.1. This enables direct synthetic-versus-non-synthetic interpretation for both tuning and topology-stability outcomes.

### 4.4 Statistical analysis

All Optuna comparisons use matched pairs within dataset, seed, and split units to control for substantial between-run heterogeneity. Within each matched unit, trials were ranked by the target metric, the top five were retained, and each condition was summarized by the median of those retained trials, yielding a top-$k$ summary with $k=5$ intended to estimate near-optimal attainable performance under a fixed Optuna budget. Paired effects were then computed as simple condition differences, with negative values favoring the first condition for lower-is-better metrics. Dataset-level and global summaries are shown as forest plots with 95% confidence intervals from two-sided paired one-sample $t$-tests; top-$k$ sensitivity analyses are provided in the Supplementary figures.

## 5. Results

### 5.1 XPySOM calibration (Equivalence)

Under matched-configuration XPySOM-versus-FloatSOM calibration on hexagonal $QE$ (Fig. S3), the two implementations are numerically equivalent up to expected floating-point accumulation-order effects (e.g., backend/kernel reduction order and host-device execution details), not algorithmic-update differences. Using the paired-testing pipeline defined in Section 4.3, we do not detect significant $QE$ differences (Supplementary Table S5). Accordingly, we treat hexagonal FloatSOM batch as a valid proxy for XPySOM in the benchmarks that follow.

Given FloatSOM's implementation usage of JIT kernels, these require a small one-time startup cost from JIT kernel compilation. This overhead is most visible on small workloads, but is progressively amortized as sample count and workload size increase. Hence, FloatSOM performs slightly more slowly ($<0.1s$) than XPySOM on small datasets. On the largest benchmark datasets (covertype and kddcup99), runtime is on par with or faster than XPySOM under this matched protocol, consistent with compilation-cost amortization. This trend is expected to strengthen further as dataset scale increases (Sections 6.1-6.2).

### 5.2 Comparison of Different Sampling Methods

We first report a focused HDSSSOM pilot as an elimination comparison rather than as part of the broader sampling benchmark. This pilot used a restricted run envelope relative to the later full-versus-random analysis and is summarized in Supplementary Table S2.

Under that pilot configuration, HDSSSOM was materially worse than the other sampling options. In the hexagonal full-batch view shown in Fig. 3, full outperformed HDSSSOM in all 50 paired comparisons, and the matched random-versus-HDSSSOM comparison showed the same direction across all datasets and paired units. We therefore focused on full and random sampling for the remainder of the manuscript.

![Figure 3](assets_manual/figures/fig_3.svg)
*Figure 3. HDSSSOM screening pilot on $QE_B$ (hexagonal topology): full vs HDSSSOM, using the restricted pilot configuration summarized in Supplementary Table S2 (10 datasets, 5 seeds, with all other pilot settings held fixed within that run envelope). Panels report dataset-matched paired top-$k$ within-unit medians plus dataset-level paired-effect summaries (Section 4.3). Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

Conversely, the differences between full and random sampling are much smaller. We observe that the effectiveness of random sampling is scale-dependent: above $10{,}000$ samples, paired $QE$ differences are not meaningfully detected, whereas in smaller datasets the random arm shows higher variability and less stable outcomes, consistent with reduced per-iteration sample support under random subsampling (Fig. 4D-F). In this benchmark, the $>10{,}000$ regime is therefore a useful practical proxy for more stable random-sampling behavior.

![Figure 4](assets_manual/figures/fig_4.svg)
*Figure 4. Sampling-mode comparison focused on full versus random under the paired analysis pipeline (Section 4.3). In the empirically larger-dataset regime observed here (>10,000 samples), little paired $QE$ separation is detected; at smaller dataset scales, random is more variable and less stable. In the forest panels, whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

Nevertheless, full sampling remains the best sampling strategy for optimal $QE$ results. Accordingly, all remaining analyses reported below rely on full-batch training unless specified.

### 5.3 Topology Results

Topology comparisons are reported with $QE$-only endpoints. We treat the Optuna hexagonal batch setting as the primary regular-topology baseline in this panel and compare MST and RNG against it. To anchor the topology results qualitatively, Fig. 5 shows representative neighborhood overlays for hexagonal, MST, and RNG on a synthetic sklearn circles dataset when run on XPySOM's default parameters. Fig. 5 demonstrates RNG's ability to contain both tree-like and mesh-like structures within the same representation, unlike MST and hexagonal.  

![Figure 5](assets_manual/figures/fig_5.svg)
*Figure 5. Representative neighborhood node and connection overlays for default XPySOM hexagonal, MST, and RNG runs on a 30,000 data-point synthetic sklearn circles dataset.*

#### 5.3.1 MST

We report dataset-wise paired improvement summaries (hexagonal over MST) with the same reporting logic as Section 5.1 in Fig. 6.

<!-- AUTO-TOPOLOGY-MST-PVALUES:START -->
Overall, MST outperforms matched hexagonal on balanced QE (Fig. 6A), indicating a net advantage across train and holdout performance. This aggregate gain is driven more clearly by train QE (Fig. 6C), while holdout QE is more mixed across datasets (Fig. 6B) and shows no clear overall holdout advantage. The overall paired t-test p-values are balanced QE (p=1.12e-05), holdout QE (p=0.15), and train QE (p=0.0064). Supplementary Table S6 lists the per-dataset and overall hexagonal-comparison p-values for MST and RNG.
<!-- AUTO-TOPOLOGY-MST-PVALUES:END -->

Across the tested top-$k$ range, which varies the number of best-ranked retained trials per matched unit, the hexagonal-versus-MST comparison remains directionally stable, indicating that the observed effect is not an artifact of a single pairing cutoff; the corresponding sensitivity analysis is provided in Fig. S5.

![Figure 6](assets_manual/figures/fig_6.svg)
*Figure 6. Hexagonal versus MST topology on $QE$ endpoints under full sampling only. Panels A-C report paired full-sampling-only $QE$ effects for $QE_B$, $QE_H$, and $QE_T$ across the available full-sampling datasets. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

#### 5.3.2 RNG

To evaluate RNG topology performance, we reuse the paired reporting logic on hexagonal versus RNG, again centered on $QE_B$ with $QE_H$ and $QE_T$ in Fig. 7. 

<!-- AUTO-TOPOLOGY-RNG-PVALUES:START -->
RNG has lower QE than matched hexagonal on the reported QE metrics (Fig. 7A-C), with overall paired t-test p-values of balanced QE (p=7.4e-10), holdout QE (p=0.0232), and train QE (p=4.69e-06). Supplementary Table S6 lists the per-dataset and overall hexagonal-comparison p-values for MST and RNG.
<!-- AUTO-TOPOLOGY-RNG-PVALUES:END -->

The main trend in Fig. 7 is that RNG improves on hexagonal most clearly in balanced QE and especially in train QE, with the separation most apparent in the real and larger datasets where the added flexibility of the graph neighborhood appears more useful than the fixed regular lattice.

Similar to the MST results, across the tested top-$k$ range, the hexagonal-versus-RNG comparison likewise remains directionally stable; the corresponding sensitivity analysis is provided in Fig. S6. The direct MST-versus-RNG sensitivity comparison is reported separately in Fig. S7.

![Figure 7](assets_manual/figures/fig_7.svg)
*Figure 7. Hexagonal versus RNG topology on $QE$ endpoints under full sampling only. Panels A-C report paired full-sampling-only $QE$ effects for $QE_B$, $QE_H$, and $QE_T$ across the available full-sampling datasets. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

### 5.4 Hyperparameter Tuning and Stability

We analyse hyperparameter effects on these results in two ways. First, we assess how tuned hyperparameters, relative to the XPySOM defaults, improve attainable $QE$. Second, we assess the stability of the "optimal" hyperparameters across seeds, topologies, and sampling modes, where greater stability permits greater confidence in the reliability of algorithm outputs. We therefore separate the hyperparameter results into tuning benefit (Fig. 8) and hyperparameter stability (Fig. 9).

#### 5.4.1 Performance Gains from Hyperparameter Tuning

The pooled tuned-versus-reference $QE$ comparison is shown in Fig. 8, with topology-specific hexagonal, MST, and RNG breakdowns provided in Fig. S8-S10. The tuned configuration is a fixed setting derived from the Optuna workflow by aggregating selected tuned settings across seeds and rerunning that single deployable configuration on the datasets. This comparison therefore estimates the gain from adopting a tuned setting rather than an untuned reference, and the pooled results show broad $QE$ improvement under tuning.

![Figure 8](assets_manual/figures/fig_8.svg)
*Figure 8. Tuned-configuration-versus-untuned-reference $QE$ comparison across $QE_B$, $QE_H$, and $QE_T$, pooled across all topology runs under the matched pairing keys. Positive values indicate the tuned configuration outperforms the untuned reference; the global overall row pools all matched tuned-configuration/untuned-reference pairs across datasets. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

At the pooled overall level, the paired summaries across all matched tuned-configuration/untuned-reference pairs also favor tuning for all three metrics, consistent with the per-dataset pattern in Fig. 8. <!-- AUTO-DEFAULT-AWARE-TOPOLOGY-STATS:START --> The same tuning pattern is observed across topologies, indicating that tuning affects all topology families rather than a single-architecture artifact.
<!-- AUTO-DEFAULT-AWARE-TOPOLOGY-STATS:END -->

#### 5.4.2 Hyperparameter Stability Across Topology and Sampling

Hyperparameter stability analyses (Section 4.3.2) compare within-topology seed-to-seed tuned-parameter drift and then contrast those internal-stability scores between topology pairs. The resulting pattern indicates that MST and RNG reach lower stability scores than hexagonal when matching dataset, sampling mode, and seed structure. Fig. 9A summarizes the full-sampling stratum and Fig. 9B the corresponding random-sampling analysis; across these panels, full sampling is generally the more stable setting.

![Figure 9](assets_manual/figures/fig_9.svg)
*Figure 9. Hyperparameter stability by sampling mode. A: selected-parameter stability under full sampling for hexagonal, MST, and RNG topologies (lower stability score is better). B: selected-parameter stability under random sampling for the same topologies. C: dataset-size stability regression under random sampling, using the selected-parameter stability score against sample size (log10) across the included topology families.*

Overall, the stability results indicate that full sampling remains the more stable choice, while random sampling becomes more viable as dataset size increases. Within that picture, MST and RNG show lower selected-parameter stability scores than hexagonal across the matched comparisons, suggesting that the graph topologies are less sensitive to tuning variation even when subsampling is introduced.

## 6. Speed Scaling

This section addresses three practical deployment questions: how much runtime full sampling adds relative to random sampling, how additional GPUs scale performance with dataset size and runtime, and whether choosing MST or RNG topologies imposes a meaningful scalability penalty relative to a hexagonal topology. 

### 6.1 Random versus Full Sampling Runtime

We report sample-scaling runtime comparisons for full versus random sampling, stratified by hexagonal, MST, and RNG, for 1, 2, and 4 GPUs in Fig. 10.

![Figure 10](assets_manual/figures/fig_10.svg)

*Figure 10. Sample-scaling runtime comparison of full versus random sampling in batch mode across $G\in\{1,2,4\}$ GPUs (A,B,C). Curves report mean wall-clock training time (s) under harmonized settings; error bars denote $\pm 1$ standard deviation across $n=3$ repeated runs per configuration. Color encodes topology (hexagonal, MST, RNG), and line style encodes sampling mode (full vs. random). Shaded x-axis regions indicate sample-size ranges that could not be run in that panel relative to the shared axis maximum due to timeouts. Lower values indicate faster execution.*

Across topologies, random sampling is faster than full sampling across all 1-, 2-, and 4-GPU comparisons, with similar proportional reductions at a given dataset size. With more GPUs, larger datasets can also be processed before timing out, although the runtime advantage of random over full narrows at the largest sample counts.

For the 1-GPU random-sampling runs, the last successful point occurs at 100M samples, as in full sampling, despite random otherwise being much faster. We interpret the failed 500M point as the stage at which the single-GPU path has tipped into disk-backed operation, so the dominant cost is no longer only the reduced number of selected samples. We therefore treat this failure as a disk-mode systems limitation rather than as evidence against the general random-versus-full runtime ordering.

### 6.2 Multi-GPU topology scaling and OOM context

To further explore the effects of parallelising operations across multiple GPUs, we conducted a series of scaling runtime benchmarks. We maximally stress-test FloatSOM's speed performance by benchmarking with full-sampling across $G\in\{1,2,4,8\}$ GPUs, up to datasets that exceed RAM. This allows us to probe the computational limits leading up to out-of-memory (OOM) conditions. Under these conditions, runtime and efficiency exhibit consistent scaling behavior across workloads (Fig. 11).

![Figure 11](assets_manual/figures/fig_11.svg)

*Figure 11. Multi-GPU full-batch scaling across $G\in\{1,2,4,8\}$ GPUs. Panels A-C show runtime (s) for dimension-, sample-, and grid-size-scaling workloads, respectively. Panels D-F show scaling efficiency for the same workloads, computed from the single-GPU baseline and the corresponding $G$-GPU runtime. Runtime error bars denote $\pm 1$ standard deviation across $n=3$ repeated runs per configuration; the 100\% efficiency reference line indicates ideal linear scaling.*

#### 6.2.1 GPU Scaling and OOM Runtime Acceleration

<!-- AUTO-SYSTEMS-SCALING-STATS:START -->
Fig. 11 suggests that increasing GPU count improves performance in the sample-scaling regime through three related mechanisms. First, computation is distributed across a larger number of workers, thereby increasing parallel throughput. Second, the onset of disk-backed execution is deferred to larger workloads because the aggregate worker-memory pool increases with GPU count. In the sample-scaling benchmark, for example, the 500,000,000 sample dataset requires disk backing under the 2-GPU configuration, whereas the 8-GPU configuration remains in RAM mode until the 1,000,000,000 sample dataset. Third, when disk-backed staging is still required, higher GPU counts appear to improve runtime because staging and disk-to-GPU transfers are distributed across more nodes. As per-node disk bandwidth is limited, distributing the workload across additional nodes may reduce transfer-path saturation and enable more stable high-throughput operation.

The 8-GPU RNG configuration processes 1,000,000,000 samples in 369.41 s (6.16 min), demonstrating billion-sample training at a runtime measured in minutes rather than hours. To reiterate, this is on a relatively complex 50-feature dataset, using a 1024-node network ($32 \times 32 = 1024$), under multi-node distributed execution, and including the time taken to remotely stage data from shared non-local storage to node-local shards before training.

The grid-size scaling panel proves to be the main exception: at the largest tested grid size (64), runtime shortens from 934.01 s (15.57 min) on 1 GPU to only 880.83 s (14.68 min) on 8 GPUs, a 5.69% reduction, indicating that once map-size/topology-refresh costs dominate, additional GPUs contribute little extra speedup. 

<!-- AUTO-SYSTEMS-SCALING-STATS:END -->

#### 6.2.2 GPU Scaling Efficiency

We next consider GPU efficiency under strong scaling. We interpret scaling efficiency relative to ideal linear scaling, such that 100\% efficiency means that doubling or octupling the GPU count yields the corresponding 2x or 8x runtime reduction, while lower values indicate increasing overhead from communication, orchestration, or I/O. In Fig. 11D-F, efficiencies are high at the higher GPU counts, including values above 100\%. When a direct 1-GPU baseline was unavailable at a given axis value, the efficiency denominator was constructed by local linear extrapolation from the last available 1-GPU point on that curve (Section 4.2), so the exact magnitude of some values should be interpreted with care if the underlying 1-GPU runtime is nonlinear over that range. Fig. 11 shows the scaling for the RNG topology, while Fig. S11 shows the corresponding supplementary hexagonal and MST outputs, which show the same super-linear pattern.

### 6.3 Topology Runtime Comparisons

With that systems context in place, we next compare topology runtimes across hexagonal, MST, and RNG configurations on 8 GPUs (Fig. 12).

![Figure 12](assets_manual/figures/fig_12.svg)

*Figure 12. Topology runtime comparison at fixed $G=8$ GPUs under full-batch processing. Panels A-C report mean wall-clock runtime (s) for dimension-, sample-, and grid-size-scaling workloads, respectively, with topology traces for hexagonal, MST, and RNG. Error bars denote $\pm 1$ standard deviation across $n=3$ repeated runs per configuration. The largest-axis 8-GPU topology runtime summaries are listed in Supplementary Table S10.*

<!-- AUTO-FIGURE10-TOPOLOGY-RUNTIME-STATS:START -->
In Fig. 12A-B, the topologies scale similarly as input complexity and data volume increase: even at the largest tested axis values, the maximum pairwise runtime spread remains modest at dimension scaling (4.70% at 5,000 dimensions) and sample scaling (3.26% at 1,000,000,000 samples).
<!-- AUTO-FIGURE10-TOPOLOGY-RUNTIME-STATS:END -->

<!-- AUTO-FIGURE10-GRID-SIZE-DISCUSSION:START -->
However, when the grid itself is enlarged in Fig. 12C, topology-dependent runtime differences become readily evident. At the largest tested grid size (grid size 64), the 8-GPU mean runtimes are 32.54 s (0.54 min) for hexagonal, 266.45 s (4.44 min) for MST, and 880.83 s (14.68 min) for RNG, corresponding to 8-GPU MST and RNG runtimes that are 8.19x and 27.07x the hexagonal runtime, respectively.
<!-- AUTO-FIGURE10-GRID-SIZE-DISCUSSION:END -->

## 7. Final FloatSOM RNG Comparison with XPySOM

With the scaling story established, Fig. 13 then tests whether the $QE$ gains from topology choice and tuning persist in deployment against XPySOM, a current high-performance Python SOM baseline in this benchmark context [@manciniXPySomHighPerformanceSelfOrganizing2020].

![Figure 13](assets_manual/figures/fig_13.svg)

*Figure 13. Integrated deployment comparison of default hexagonal XPySOM versus tuned FloatSOM RNG. Panels A-C compare $QE_B$, $QE_H$, and $QE_T$ using the untuned hexagonal XPySOM baseline against matched tuned FloatSOM RNG full-sampling runs. Panel D provides the scaling/runtime context for the same comparison, with the separately executed targeted 1B-sample runs discussed in the text rather than plotted directly. Taken together, this integrated figure summarizes the operating point observed for tuned FloatSOM RNG once workload size is large enough for steady-state execution to dominate startup overhead. Per-dataset and `GLOBAL_OVERALL` panel summaries are listed in Supplementary Table S7.*

<!-- AUTO-FIG11-DEPLOYMENT-QE-STATS:START -->
At the overall level, Fig. 13 shows median percentage improvements of $QE_B$ (14.5%); $QE_H$ (9.1%); and $QE_T$ (22.5%) for tuned FloatSOM RNG relative to default hexagonal XPySOM, capturing the combined deployment effect of topology choice and tuning on $QE$.
<!-- AUTO-FIG11-DEPLOYMENT-QE-STATS:END -->

For the default hexagonal XPySOM reference in Fig. 13, workloads beyond the $10^8$-sample case were not processed because they exceeded available VRAM and XPySOM requires the full dataset to be loaded into memory. In sum, tuned FloatSOM RNG delivers better $QE$ than the default hexagonal XPySOM baseline, while also running faster and scaling to larger workloads (Supplementary Table S7). 

## 8. Discussion

This work introduces FloatSOM as a GPU-oriented SOM framework that combines topology flexibility, out-of-memory execution, and distributed multi-GPU training in a single implementation. The study also provides a unified empirical analysis of sampling strategy, graph-based versus fixed-lattice topology, topology-aware hyperparameter tuning, and workload scaling across synthetic and real benchmarks. Taken together, these results clarify how algorithmic and systems choices jointly shape SOM quality and runtime under practical deployment constraints.

### 8.1 Sampling tradeoff (random versus full)

The sampling trade-off is strongly scale dependent. In smaller datasets, random subsampling produces less stable outcomes, whereas above $10{,}000$ samples paired $QE$ differences between full and random are not meaningfully detected. Full sampling is therefore the safer choice when stability is the priority, while random sampling is better viewed as a throughput-oriented option at larger scales. In the OOM regime, this advantage narrows because random still requires each worker-local chunk to be read before subsampling, so it reduces compute more directly than dataset I/O.

### 8.2 Topology Comparisons (MST and RNG)

Globally, both MST and RNG outperform the fixed hexagonal topology in these comparisons, with RNG showing the strongest overall $QE$ results. We attribute this to the degree of structural flexibility available to each topology to best conform to the underlying data distribution. Hexagonal neighborhoods are the most restrictive, imposing  a fixed mesh [@kohonenEssentialsSelforganizingMap2013]; whilst MST relaxes that structure but still limits propagation to tree paths [@kangasVariantsSelforganizingMaps1990]. Finally, RNG is not constrained by either of these limitations can accordingly can form both tree-list and mesh type structures as prototype geometry evolves [@toussaintRelativeNeighbourhoodGraph1980]. These results are corroborated with the sensitivity analyses in Supplementary Figs. S5-S7.

### 8.3 Tuning benefit under matched defaults

Across the matched Fig. 8 comparisons, tuned configurations consistently outperform untuned reference settings. Hyperparameter tuning should therefore be treated as part of the method configuration rather than as optional post-processing.

The key implication is that topology choice and hyperparameter choice are coupled. Gains remain positive across hexagonal, MST, and RNG, so tuning is not confined to a single topology. A practical deployment strategy therefore requires topology-aware tuning.

### 8.4 Hyperparameter stability and dataset-type interpretation

The stability analysis suggests that graph topologies are recovered more consistently than the fixed-lattice baseline. Hexagonal maps cannot rebuild connectivity once initialized, whereas MST and RNG recompute connectivity from the evolving prototype geometry. This likely reduces sensitivity to seed-level variation in the tuned region. It may also help explain why the graph topologies are better suited to higher-dimensional, non-synthetic datasets [@kohonenEssentialsSelforganizingMap2013; @kangasVariantsSelforganizingMaps1990].

### 8.5 Systems implications and limits

Distributed execution helps most in large workloads, but not as a uniform multiplicative speedup. At small problem sizes, Ray startup and orchestration overhead can offset the benefit. At larger workloads, the gains are more plausibly explained by improved memory residency and distributed data movement than by compute scaling alone.Multi-GPU execution is therefore most useful once workload size is large enough for memory pressure and steady-state throughput to dominate orchestration overhead.

Overall, these results support a practical deployment strategy that uses the maximum GPU count permitted by file I/O, RNG topology, and the derived default hyperparameters, with sampling chosen by scale: full for smaller datasets when stability is critical, and random as a throughput-oriented option in the empirically larger-dataset regime observed here (>10,000 samples) where paired $QE$ differences are not meaningfully detected. When workloads are dominated by very large grid-size scaling, MST remains a reasonable alternative because its graph-construction path scales more favorably than RNG.

## 9. Acknowledgements

This work was supported by computational resources provided by the Australian Government through the National Computational Infrastructure (NCI) under the ANU Merit Allocation Scheme.

We also acknowledge the computational services provided by the University of Bern, the University of Sydney, and the Walter and Eliza Hall Institute.

We thank Prof. Hanna Suominen for her input and advice.

## 10. References

::: {#refs}
:::

## 11. Supplementary Tables (End Matter)

**Supplementary Table S1. Dataset metadata and numbered point key for the Figure 4 sampling-mode analysis.**

| dataset_index | dataset | dimension_count | sample_size | dataset_type |
| --- | --- | --- | --- | --- |
| 1 | iris | 4 | 150 | real |
| 2 | wine | 13 | 178 | real |
| 3 | olivetti_faces | 4096 | 400 | real |
| 4 | diabetes | 10 | 442 | real |
| 5 | breast_cancer | 30 | 569 | real |
| 6 | digits | 64 | 1797 | real |
| 7 | california_housing | 8 | 20640 | real |
| 8 | blobs | 2 | 30000 | synthetic |
| 9 | circles | 2 | 30000 | synthetic |
| 10 | moons | 2 | 30000 | synthetic |
| 11 | s_curve | 3 | 30000 | synthetic |
| 12 | swiss_roll | 3 | 30000 | synthetic |
| 13 | kddcup99 | 41 | 494021 | real |
| 14 | covertype | 54 | 581012 | real |

**Supplementary Table S2. HDSSSOM pilot configuration summary for Figure 3.**

| field | value |
| --- | --- |
| pilot purpose | Initial HDSSSOM screening before the broader sampling comparison |
| run envelope | Restricted hexagonal Optuna pilot |
| datasets | `swiss_roll`, `moons`, `circles`, `blobs`, `s_curve`, `breast_cancer`, `wine`, `iris`, `digits`, `olivetti_faces` |
| dataset count | 10 |
| seed count | 5 |
| sampling methods present | `full`, `random`, `hdsssom` |
| topology | `hexagonal` |
| algorithm families in campaign | `batch`, `colors` |
| optimization split setup | `evaluation-split=both`, yielding `QE_H` and `QE_T`; Figure 3 reports paired `QE_B` |
| trials per scenario | 200 |
| main-text figure slice | hexagonal / `full_batch` / `full vs hdsssom` |

**Supplementary Table S3. FloatSOM-versus-XPySOM calibration $QE$ summary for the MST topology path.** The `dataset_index` column matches the numbered points in Supplementary Figure S1 panel D. See `assets/tables/supp_xpysom_calibration_qe_mst.tsv`.

**Supplementary Table S4. FloatSOM-versus-XPySOM calibration $QE$ summary for the RNG topology path.** The `dataset_index` column matches the numbered points in Supplementary Figure S2 panel D. See `assets/tables/supp_xpysom_calibration_qe_rng.tsv`.

**Supplementary Table S5. FloatSOM-versus-XPySOM calibration $QE$ summary for the hexagonal topology path.** The `dataset_index` column matches the numbered points in Supplementary Figure S3 panel D. See `assets/tables/supp_xpysom_calibration_QE_Hexagonal.tsv`.

<!-- AUTO-TOPOLOGY-PVALUE-SUPP-TABLE:START -->
**Supplementary Table S6. Paired topology-comparison p-values for hexagonal versus MST and hexagonal versus RNG across balanced QE, holdout QE, and train QE.** Rows list metric/dataset entries, including the OVERALL row. The MST and RNG columns report p-values using the manuscript reporting convention. See `assets/tables/supp_table_topology_hex_vs_mst_rng_pvalues.tsv`.
<!-- AUTO-TOPOLOGY-PVALUE-SUPP-TABLE:END -->

**Supplementary Table S7. Figure 13 deployment-comparison percent summary for tuned FloatSOM RNG versus default hexagonal XPySOM across $QE_B$, $QE_H$, and $QE_T$.** Rows list per-dataset and `GLOBAL_OVERALL` entries with the plotted median percent change and 95% confidence interval. See `assets/tables/supp_table_figure_12_xpysom_rng_deployment_summary.tsv`.

**Supplementary Table S8. Supplementary Figure S12 deployment-comparison percent summary for tuned FloatSOM hexagonal versus default hexagonal XPySOM across $QE_B$, $QE_H$, and $QE_T$.** Rows list per-dataset and `GLOBAL_OVERALL` entries with the plotted median percent change and 95% confidence interval. See `assets/tables/supp_table_s13_xpysom_hexagonal_deployment_summary.tsv`.

**Supplementary Table S9. Supplementary Figure S13 deployment-comparison percent summary for tuned FloatSOM MST versus default hexagonal XPySOM across $QE_B$, $QE_H$, and $QE_T$.** Rows list per-dataset and `GLOBAL_OVERALL` entries with the plotted median percent change and 95% confidence interval. See `assets/tables/supp_table_s14_xpysom_mst_deployment_summary.tsv`.

**Supplementary Table S10. Figure 12 topology runtime summary at the largest common 8-GPU axis value for the dimension-, sample-, and grid-size-scaling workloads.** Rows report the plotted 8-GPU mean runtimes for hexagonal, MST, and RNG, together with the fastest and slowest topology at that axis value and the maximum pairwise runtime spread. See `assets/tables/supp_table_figure_12_topology_runtime_summary.tsv`.

## 12. Supplementary Figures (End Matter)

![Supplementary Figure S1](assets_manual/figures/supp_fig_s1.svg)
*Supplementary Figure S1. FloatSOM-versus-XPySOM calibration under default settings for the MST topology path. Panels A-C report paired $QE$ effects for $QE_B$, $QE_H$, and $QE_T$. Panel D reports dataset-level median runtime deltas against dataset size, where each numbered dot is the median matched-seed value of `FloatSOM time - XPySOM time`; negative values favor FloatSOM and positive values favor XPySOM. The point numbers map to Supplementary Table S3. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

![Supplementary Figure S2](assets_manual/figures/supp_fig_s2.svg)
*Supplementary Figure S2. FloatSOM-versus-XPySOM calibration under default settings for the RNG topology path. Panels A-C report paired $QE$ effects for $QE_B$, $QE_H$, and $QE_T$. Panel D reports dataset-level median runtime deltas against dataset size, where each numbered dot is the median matched-seed value of `FloatSOM time - XPySOM time`; negative values favor FloatSOM and positive values favor XPySOM. The point numbers map to Supplementary Table S4. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

![Supplementary Figure S3](assets_manual/figures/supp_fig_s3.svg)
*Supplementary Figure S3. FloatSOM-versus-XPySOM calibration under default settings for the hexagonal topology path. Panels A-C report paired $QE$ effects for $QE_B$, $QE_H$, and $QE_T$. Panel D reports dataset-level median runtime deltas against dataset size, where each numbered dot is the median matched-seed value of `FloatSOM time - XPySOM time`; negative values favor FloatSOM and positive values favor XPySOM. The point numbers map to Supplementary Table S5. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

![Supplementary Figure S4](assets_manual/figures/supp_fig_s4.svg)
*Supplementary Figure S4. MST versus RNG topology on $QE$ endpoints under full sampling only. Panels A-C report paired full-sampling-only $QE$ effects for $QE_B$, $QE_H$, and $QE_T$ across the available full-sampling datasets. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

![Supplementary Figure S5](assets_manual/figures/supp_fig_s5.svg)
*Supplementary Figure S5. Hexagonal versus MST topology sensitivity under full sampling only. Panels A-C report the matched top-$k$ paired sensitivity analysis for $QE_B$, $QE_H$, and $QE_T$.*

![Supplementary Figure S6](assets_manual/figures/supp_fig_s6.svg)
*Supplementary Figure S6. Hexagonal versus RNG topology sensitivity under full sampling only. Panels A-C report the matched top-$k$ paired sensitivity analysis for $QE_B$, $QE_H$, and $QE_T$.*

![Supplementary Figure S7](assets_manual/figures/supp_fig_s7.svg)
*Supplementary Figure S7. MST versus RNG topology sensitivity under full sampling only. Panels A-C report the matched top-$k$ paired sensitivity analysis for $QE_B$, $QE_H$, and $QE_T$.*

![Supplementary Figure S8](assets_manual/figures/supp_fig_s8.svg)
*Supplementary Figure S8. Tuned-configuration-versus-untuned-reference $QE$ comparison for the hexagonal topology only, across $QE_B$, $QE_H$, and $QE_T$ under the matched pairing keys. Positive values indicate the tuned configuration outperforms the untuned reference. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

![Supplementary Figure S9](assets_manual/figures/supp_fig_s9.svg)
*Supplementary Figure S9. Tuned-configuration-versus-untuned-reference $QE$ comparison for the MST topology only, across $QE_B$, $QE_H$, and $QE_T$ under the matched pairing keys. The tuned configuration is derived from the Optuna-selected settings by taking the mean of numeric parameters and the mode of categorical parameters across seeds. Positive values indicate the tuned configuration outperforms the untuned reference. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

![Supplementary Figure S10](assets_manual/figures/supp_fig_s10.svg)
*Supplementary Figure S10. Tuned-configuration-versus-untuned-reference $QE$ comparison for the RNG topology only, across $QE_B$, $QE_H$, and $QE_T$ under the matched pairing keys. The tuned configuration is derived from the Optuna-selected settings by taking the mean of numeric parameters and the mode of categorical parameters across seeds. Positive values indicate the tuned configuration outperforms the untuned reference. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

![Supplementary Figure S11](assets_manual/figures/supp_fig_s11.svg)
*Supplementary Figure S11. Full GPU-count scaling context for MST and hexagonal under matched full-batch settings. Panels A-C show MST runtime across dimension-, sample-, and grid-size-scaling workloads; panels D-F show the corresponding hexagonal runs. Within each panel, curves correspond to $G\in\{1,2,4,8\}$ GPUs and report mean wall-clock runtime (s) with $\pm 1$ standard-deviation error bars across $n=3$ repeated runs per configuration.*

![Supplementary Figure S12](assets_manual/figures/supp_fig_s12.svg)
*Supplementary Figure S12. Deployment comparison of default hexagonal XPySOM versus tuned FloatSOM hexagonal. Panels A-C report paired $QE$ effects for $QE_B$, $QE_H$, and $QE_T$ under the matched dataset/seed comparison keys. Positive values indicate tuned FloatSOM hexagonal outperforms default hexagonal XPySOM. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect. Per-dataset and `GLOBAL_OVERALL` panel summaries are listed in Supplementary Table S8.*

![Supplementary Figure S13](assets_manual/figures/supp_fig_s13.svg)
*Supplementary Figure S13. Deployment comparison of default hexagonal XPySOM versus tuned FloatSOM MST. Panels A-C report paired $QE$ effects for $QE_B$, $QE_H$, and $QE_T$ under the matched dataset/seed comparison keys. Positive values indicate tuned FloatSOM MST outperforms default hexagonal XPySOM. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect. Per-dataset and `GLOBAL_OVERALL` panel summaries are listed in Supplementary Table S9.*

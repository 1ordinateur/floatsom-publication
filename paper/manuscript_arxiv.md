\begin{center}
{\LARGE\bfseries FloatSOM: GPU-Accelerated, Distributed, Topology-Flexible Self-Organizing Maps\par}
\end{center}

\vspace{1.8em}

\noindent
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l r}
\textbf{Tony Xu} & \textit{tony.xu@anu.edu.au} \\
\textit{John Curtin School of Medical Research, Australian National University} & \\
\\[0.8em]
\textbf{Sarah Klamt} & \\
\textit{Department of Information Technology and Electrical Engineering, ETH Zürich} & \\
\\[0.8em]
\textbf{Katherine Turner} & \\
\textit{Mathematical Data Science Centre, Australian National University} & \\
\\[0.8em]
\textbf{Anne Brüstle} & \\
\textit{John Curtin School of Medical Research, Australian National University} & \\
\\[0.8em]
\textbf{Felix Marsh-Wakefield$^\dagger$} & \\
\textit{Centenary Institute of Cancer Medicine and Cell Biology} & \\
\textit{University of Sydney} & \\
\\[0.8em]
\textbf{Givanna Putri$^\dagger$} & \\
\textit{Walter and Eliza Hall Institute} & \\
\end{tabular*}

\vspace{0.8em}

\noindent\textit{$^\dagger$ Givanna Putri and Felix Marsh-Wakefield contributed equally as co-last authors.}

\vspace{1.4em}

## Abstract

GPU-accelerated Self-Organizing Map (SOM) implementations are among the most competitive options for large-scale SOM analysis, but growing dataset sizes increasingly challenge their practical use because workloads no longer fit cleanly within device-memory limits. We introduce FloatSOM, a SOM framework for scalable training and deployment that supports multi-GPU execution, out-of-memory disk-backed streaming, and scalable training-time graph topologies beyond regular lattices. We evaluate FloatSOM on 14 synthetic and real benchmark datasets together with controlled speed scaling benchmarks, and show that these graph-based topologies, combined with topology-aware hyperparameter fine-tuning, yield lower quantization error than current state-of-the-art SOM baselines. FloatSOM also sustains this performance at large scale with high-throughput distributed execution; in the largest benchmark, it trains a 1024-node SOM network on 1,000,000,000 samples with 50 features in 6.16 minutes on 8 GPUs across two separate high-performance-computing nodes.

## 1. Introduction

Self-Organizing Map (SOM), originally introduced by Kohonen [@kohonenSelforganizingMap1990], is an unsupervised machine-learning method that uses competitive learning to organize nodes such that they capture the topology of the data. In practice, this topology-preserving representation means SOMs are commonly used to map dataset topology, produce dimensionality-reduced visualizations, and conduct clustering at scale [@kangasVariantsSelforganizingMaps1990]. As dataset size and heterogeneity increase, the computational requirements of SOMs grow in both time and memory. Many current implementations, however, remain constrained to single-device workloads that must fit within video random-access memory (VRAM, GPU memory), with limited support for distributed compute, out-of-core execution, and modern GPU orchestration.

SOM topology presents a second limitation. Classical SOMs are traditionally trained on regular rectangular or hexagonal lattices because fixed grids make neighborhood definition, visualization, and optimization straightforward. However, these regular lattices also impose a strong geometric prior on the learned representation. Accordingly, prior work on dynamic, growing, and graph-structured SOM variants reflects a long-standing recognition that fixed lattices are not always the best match for irregular data geometry [@alahakoonDynamicSelforganizingMaps2000; @vasighiDirectedBatchGrowing2017; @kangasVariantsSelforganizingMaps1990]. However, these alternatives have generally not been developed or evaluated in the high-throughput, large-sample regime targeted by modern GPU-enabled applications and are broadly impractical for deployment at scale.

FloatSOM is designed to address these combined systems and topology limitations. Within FloatSOM, we implement distributed multi-GPU execution, out-of-memory disk-backed streaming, and scalable topology-flexible training beyond standard fixed lattices. Specifically, we implement highly scalable minimum-spanning-tree (MST) and relative-neighborhood-graph (RNG) topologies that define the SOM neighborhood during training rather than being overlaid on a completed map. We also evaluate multiple sampling strategies as a route to further computational acceleration and derive fine-tuned hyperparameter configurations across diverse datasets and recommended operating regimes. FloatSOM is suitable for both high-performance-computing (HPC) operation and consumer-grade desktop GPUs.

## 2. Related Work

Related work on practical SOM deployment spans software implementations, sampling-efficient training, topology design, and model selection.

### 2.1 Open-Source SOM Implementations and Systems

Open sourced SOM libraries range from lightweight to more performance-oriented implementations. MiniSom is a compact Python implementation of the classical online regime [@vettigliJustGlowingMinisom2018], whereas XPySOM is a Python-based batch SOM implementation designed for efficient GPU-backed execution [@manciniXPySomHighPerformanceSelfOrganizing2020]. At larger scales, Somoclu and GigaSOM provide mature parallel SOM systems for large workloads [@wittekSomocluEfficientParallel2017; @kratochvilGigaSOMjlHighperformanceClustering2020]. However, an important systems gap remains: XPySOM is limited to a single GPU and requires the full dataset to fit in VRAM, while GigaSOM emphasizes distributed CPU execution rather than distributed GPU training within a broadly used Python workflow. FloatSOM is intended to address that gap.

### 2.2 Sampling Methodologies for SOM Training

Classical online and batch SOM training schemes traditionally sample every data-point in every training iteration [@kohonenSelfOrganizingMaps2001; @liuRobustPhenotypingHighly2023]. Consequently, one obvious route to improving SOM speed and scalability is subsampling, which reduces the amount of data used to train the final SOM.

To date, numerous subsampling strategies have been proposed. Beyond naive random sampling, guided subsampling methods include hierarchical dynamic subset selection SOM (HDSSSOM), which concentrates computation on difficult or stale regions of the data [@wetmoreSpeedingSelfOrganizingFeature2005], and adaptive sampling strategies for design-space exploration [@itoDesignSpaceExploration2016]. However, systematically benchmarked and openly maintained implementations that compare full-data, random, and guided sampling within a modern high-performance SOM workflow remain limited.

### 2.3 SOM Lattice and Adaptive Topologies

Most practical SOM implementations retain regular rectangular or hexagonal lattices because they simplify neighborhood indexing, visualization, and vectorized updates [@kohonenSelforganizingMap1990; @kohonenEssentialsSelforganizingMap2013]. Hexagonal lattices are often preferred in the literature and serve as the regular-topology baseline in this manuscript [@whiteTopologyMattersNetwork2008; @kohonenEssentialsSelforganizingMap2013; @forestSurveyImplementationPerformance2020].

The SOM literature has also explored alternatives to fixed lattices, including dynamic maps and graph-structured neighborhoods [@vasighiDirectedBatchGrowing2017; @spanakisAMSOMAdaptiveMoving2016; @kangasVariantsSelforganizingMaps1990; @jangUseMinimalSpanning2009]. MSTs have previously appeared in SOM analyses, but prior uses generally treat the MST as an interpretive structure over an already trained map rather than as the neighborhood relation that drives SOM learning. For example, Jang et al. use MSTs for interpretation, subnode embedding, and map-shape assessment, while FlowSOM overlays an MST on trained SOM codes to visualize relationships among metaclusters [@jangUseMinimalSpanning2009; @vangassenFlowSOMUsingSelforganizing2015]. This is distinct from the training-time role used here: in FloatSOM, the MST is the operative neighborhood graph during SOM updates, is recalculated from the evolving node-weight geometry, and directly changes the update influence matrix used during learning. Thus, current deployed MST uses in SOM workflows generally configure post hoc connections on trained maps, whereas FloatSOM uses MST/RNG graphs as the training neighborhood itself. A post hoc MST overlay would also leave the quantization error of the underlying trained SOM unchanged, because $QE$ is determined by distances between samples and the fixed learned prototypes; the $QE$ improvements reported here therefore require graph-mediated training that changes those prototype locations. Earlier SOM variants also discussed MST-defined neighborhoods during learning [@kangasVariantsSelforganizingMaps1990], but these alternatives have not been assessed on large-scale datasets and do not have implementations that are either publicly available or suitable for distributed GPU computation. We have not identified prior work implementing dynamically refreshed MST or RNG topologies as scalable GPU-compatible SOM training neighborhoods. FloatSOM's topology contribution is therefore a scalable GPU-compatible implementation and large-scale quantification of refreshed graph-based SOM training, rather than a post hoc MST overlay on a conventional trained SOM.

Relative Neighborhood Graphs (RNGs) [@toussaintRelativeNeighbourhoodGraph1980] are of particular interest here. To our knowledge, dynamically refreshed RNG neighborhoods have not previously been used as a SOM training topology; we return to the full rationale and implementation for RNG in Section 3.2.2.

### 2.4 Hyperparameter Optimization and Fair Comparison

SOM performance depends strongly on parameters such as map size, initialization, learning-rate schedule, and neighborhood schedule. Prior work shows that these choices can substantially affect observed performance [@akindukoSOMStochasticInitialization2016; @forestSurveyImplementationPerformance2020]. This makes comparisons based only on untuned defaults difficult to interpret.

More generally, modern machine-learning workflows increasingly rely on automated hyperparameter optimization rather than manual tuning alone. Tools such as Optuna provide bounded search over large hyperparameter spaces and can support multi-objective optimization, allowing parameter settings to be selected with respect to several benchmark criteria simultaneously rather than collapsed into a single score [@akibaOptunaNextgenerationHyperparameter2019].

## 3. Methods

FloatSOM combines four methodological components: sample selection, topology definition, SOM updates, and the compute architecture used to execute them. The framework supports random, full, and HDSSSOM sampling; rectangular, hexagonal, MST, and RNG topologies; and execution modes ranging from local GPU training to single-node and multi-node multi-GPU operation. Hyperparameter optimization is treated as a separate layer applied across these configurations. Figure 1 summarizes this design.

![Figure 1](assets_manual/figures/fig_1.svg){.half-width width=50%}

*Figure 1. Schematic overview of the FloatSOM methods framing used in this manuscript. Sample selection and topology definition have configurable components that feed into the standard SOM training step, while the compute architecture determines how that same training procedure is executed in practice. The options shown here summarize the FloatSOM configurations discussed in the following subsections.*

### 3.1 Sampling Selector Mathematics

This section formalizes the sampling policies evaluated in this manuscript.

Let the full dataset be $X=\{x_i\}_{i=1}^{N}$, where $N$ is the total number of samples in the dataset. Let $m$ denote the number of samples presented to the SOM in a given training iteration, or the sampling budget. This budget is either fixed directly or determined as a proportion $\rho$ of the dataset:

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

If $m\ge N$, random returns the full dataset (no subsampling). Otherwise, to avoid biasing training toward a single fixed subsample, the random selector redraws $\mathcal{I}_t^{\mathrm{random}}$ from the entire dataset at every training iteration. The selected training subset is $X_t^{(s)}=\{x_i:i\in\mathcal{I}_t^{(s)}\}$. Because random subsampling draws uniformly from size-$m$ subsets, we are able to better obtain coverage over the full dataset.

HDSSSOM is implemented in multi-GPU-compatible form as an adaptive sampler that preferentially revisits difficult, under-trained, or stale samples [@wetmoreSpeedingSelfOrganizingFeature2005].

### 3.2 Topology Definition

After initialization, neighborhood relations are defined by the selected topology. For regular-lattice baselines, we support both grid and hexagonal layouts. Consistent with prior SOM guidance, we treat hexagonal as the standard reference topology in this manuscript. We implement the hexagonal lattice topology in accordance with [@vettigliJustGlowingMinisom2018]. For MST and RNG, neighborhood structure is derived from the current node-weight geometry using the methodologies below.

Note that regardless of the selected topology, FloatSOM uses the same node-weight initialization options. Initialization determines only the starting node weights; neighborhood relations are applied afterward according to the selected topology to establish node connections, keeping the initial state comparable across regular-lattice and graph-based runs.

#### 3.2.1 MST Topology Implementation

The MST topology replaces fixed lattice neighborhood distance with graph hop distance on a minimum spanning tree built from current SOM nodes. The MST is therefore part of the SOM training rule: it determines which nodes receive neighborhood influence during each topology-refresh interval, rather than serving as a visualization or analysis layer after training. For $P$ nodes in feature dimension $d$, the pairwise node-distance matrix is formed with the standard squared-distance Gram identity, which avoids 3D broadcast tensors and preserves $O(P^2 d)$ dense linear-algebra structure.

In a standard lattice SOM, neighborhood distance is determined by fixed node grid coordinates, which are assigned before training and are independent of the learned node-weight geometry in data space. MST instead uses the node weight vectors to locate the nodes in data space, then calculates the minimum spanning tree between those node locations [@kruskalShortestSpanningSubtree1956]. The node distances therefore determine which tree edges are selected, while the neighborhood distance used by the Gaussian update is the graph hop distance along the resulting tree. Topology-derived influence matrices are cached and refreshed according to the topology-refresh schedule.

At the beginning of training, the topology is refreshed every iteration: the previous tree edges are discarded, pairwise distances are recalculated from the updated node weights, and the MST is recomputed from those updated node locations. The refresh rate then decays as training proceeds, so topology updates become less frequent later in training. If the current iteration does not trigger recomputation, the previous graph state and cached influences are reused. Otherwise, the MST is recalculated and the influence cache is rebuilt.

```text
Input: node weights W_t, iteration t, topology-refresh policy
Output: topology state (E_t, g_t, cached influences)
 1: Query the refresh policy for the current iteration
 2: if no topology refresh is due then
 3:         return previous topology state
 4: end if
 5: Compute pairwise squared node distances on GPU
 6: Transfer distances to CPU and run Kruskal to obtain MST edges E_t
 7: Build adjacency from E_t
 8: Compute all-pairs graph hop distances g_t with chunked GPU Floyd-Warshall
 9: Deduplicate active radii and rebuild/update cached influence maps
10: Commit E_t, g_t, and cache state
11: return topology state
```
*Algorithm 1. Dynamic MST topology update with refresh-triggered recomputation and cached influence reuse.*

#### 3.2.2 RNG Topology Implementation

Our RNG topology constructs a Relative Neighborhood Graph over current node distances using the standard RNG criterion [@toussaintRelativeNeighbourhoodGraph1980]. Unlike MST, RNG is not restricted to a single spanning-tree backbone, so multiple locally supported neighborhood relations can be retained. Conceptually, the absence of this restriction permits the RNG topology to represent both tree-like and denser mesh-like local structures simultaneously throughout the training process. Consequently, we hypothesise that this greater freedom in topology will allow RNG-based SOM maps to better represent a variety of distributions.

```text
Input: node weights W_t, iteration t, topology-refresh policy
Output: topology state (E_t, g_t, cached influences)
 1: Query the refresh policy for the current iteration
 2: if no topology refresh is due then
 3:         return previous topology state
 4: end if
 5: Compute pairwise squared node distances on GPU
 6: Evaluate RNG candidate elimination in chunks using the blocker test
 7: Retain surviving RNG edges E_t and build adjacency
 8: Compute all-pairs graph hop distances g_t with chunked GPU Floyd-Warshall
 9: Deduplicate active radii and rebuild/update cached influence maps
10: Commit E_t, g_t, and cache state
11: return topology state
```
*Algorithm 2. Dynamic RNG topology update with refresh-triggered recomputation and cached influence reuse.*

### 3.3 Multi-GPU + Larger-Than-Memory Methodology and Implementation

FloatSOM is implemented for distributed GPU execution, with support for RAM (Random Access Memory, CPU memory) spilling and disk-backed execution when datasets exceed available VRAM and RAM, respectively.

![Figure 2](assets_manual/figures/fig_2.svg){.half-width width=50%}

*Figure 2. Multi-GPU data loading and NCCL synchronization schematic. In disk-backed mode, data are sharded to worker-local storage, loaded chunk-wise into pinned host memory, transferred to GPU with transfer overlapped with compute, processed locally, and synchronized by NCCL all-reduce before one weight update per iteration. In RAM mode, data are instead pre-sharded into worker-local CPU RAM and follow the same pinned-memory-to-GPU path without disk reads.*

#### 3.3.1 General Multi-GPU Logic

Distributed execution uses Ray actors with one GPU per worker and NCCL collectives for synchronous aggregation [@moritzRayDistributedFramework2018]. At each iteration, every worker processes its assigned shard locally in `n_chunks`, accumulates worker-local statistics across those chunks, and synchronizes once per iteration.

For worker $g \in \{1,\dots,G\}$, let $X_{t,g}^{(s)}$ denote the worker-local shard of the selected iteration subset $X_t^{(s)}$. Let $j$ index SOM nodes, let $w_j^{(t)}$ denote the weight vector of node $j$ at iteration $t$, let $b(x)$ denote the best-matching unit (BMU) of sample $x$ under the current weights, let $h_{j,b(x)}^{(t)}$ denote the iteration-$t$ neighborhood influence between node $j$ and the BMU (Best Matching Unit)of $x$, and let $\eta_t$ denote the learning rate at iteration $t$. The local accumulators are:

$$
\begin{aligned}
 U_j^{(g)} &= \sum_{x \in X_{t,g}^{(s)}} \eta_t\,h_{j,b(x)}^{(t)}(x-w_j^{(t)}), \\
 H_j^{(g)} &= \sum_{x \in X_{t,g}^{(s)}} h_{j,b(x)}^{(t)}.
\end{aligned}
\tag{3}
$$

$U_j^{(g)}$ is worker $g$'s summed learning-rate-scaled displacement for node $j$, and $H_j^{(g)}$ is the corresponding summed neighborhood influence used to normalize that displacement.

Global synchronized accumulators are:

$$
\begin{aligned}
U_j &= \sum_{g=1}^{G} U_j^{(g)}, \\
H_j &= \sum_{g=1}^{G} H_j^{(g)}.
\end{aligned}
\tag{4}
$$

In implementation, $U_j^{(g)}$ and $H_j^{(g)}$ are accumulated across the worker's `n_chunks` and synchronized once per iteration. After synchronization, $U_j$ and $H_j$ define the global update numerator and normalization denominator for node $j$. Under weighted normalization, the normalized update is given by $U_j/H_j$ up to numerical safeguards; optional momentum is then added to this normalized update before applying the resulting change to the node weight $w_j^{(t)}$.

#### 3.3.2 Multi-GPU Implementation Details

As shown in Fig. 2, each worker uses a chunked loading path from CPU memory to GPU memory. In streaming mode, data are distributed to worker-local disk shards and then read chunk-by-chunk into pinned host memory before transfer to GPU; in RAM mode, data are pre-sharded into worker-local CPU RAM and follow the same pinned-memory path without disk reads. FloatSOM overlaps data transfer and compute across CUDA streams, uses JIT-compiled kernels (Just-In-Time) for the core BMU and update steps, and synchronizes worker-local accumulators once per iteration with NCCL all-reduce. Weights therefore remain resident on worker GPUs across iterations. For large topologies that would otherwise exceed VRAM, FloatSOM also supports spilling the associated graph-distance or influence structures from VRAM to system RAM.

#### 3.3.3 Larger-Than-Memory Capable Topology Updates

Additional larger-than-memory support for topology updates is provided through topological chunking. Topological chunking applies the same idea to topology-side computations within each worker. When graph-distance or influence structures would otherwise exceed a worker's VRAM, those computations are tiled and evaluated in bounded pieces rather than materialized at once. This topology-side chunking is not depicted in the Fig. 2 data path, but it follows the same per-worker bounded-memory execution rule.

### 3.4 Multi-Objective Hyperparameter Optimization

We use Optuna as an automated multi-objective hyperparameter optimization tool to derive near-optimal performance and corresponding hyperparameters for each FloatSOM configuration under a given sampling and topology combination [@akibaOptunaNextgenerationHyperparameter2019].

## 4. Experimental Setup

We use two benchmark protocols: an Optuna quality benchmark and a speed scaling benchmark. The first evaluates algorithmic quality and tuned attainable performance, and the second evaluates runtime and distributed scaling behavior. All production Optuna and speed benchmarks reported in this manuscript were executed on Gadi at the National Computational Infrastructure (NCI), Australia, on gpuvolta nodes [@HPCSystemsNCI], using a consistent multi-GPU environment across runs.

### 4.1 Optuna benchmark protocol

We use Optuna-based multi-objective optimization to determine the best attainable performance and corresponding hyperparameters for each sampling (full and random) and topology (hexagonal, MST, and RNG) combination, optimizing jointly for train and holdout quantization error ($QE_T$ and $QE_H$). In this protocol, each dataset-topology-sampling configuration is optimized for 200 trials across 10 seeds under variant-appropriate search constraints. The reported runs used the standard in-memory batch path rather than the Ray-distributed execution stack. This corresponds to:

$$
14_{\text{Datasets}} \times 10_{\text{Seeds}} \times 3_{\text{Topologies}} \times 2_{\text{SamplingMethods}} \times 200_{\text{Trials}}= 168{,}000 \text{ Runs}
$$

Sampling comparisons in Section 5.2 compare full versus random paired analyses pooled across all topologies. Topology comparisons in Sections 5.3-5.4 use full sampling runs across hexagonal, MST, and RNG. Otherwise, the remaining analyses use full sampling. A separate focused HDSSSOM pilot is reported in Section 5.2 and summarized in Supplementary Table S2. This pilot additionally included the HDSSSOM sampling methodology alongside the full and random sampling methodologies.

#### 4.1.1 Optuna benchmark datasets and preprocessing

The Optuna benchmark uses 14 synthetic and real scikit-learn datasets spanning low- and high-dimensional regimes [@pedregosaScikitlearnMachineLearning2011]; dataset metadata are listed in Supplementary Table S1. Synthetic datasets are generated according to the random seed, whereas real datasets are loaded directly from sklearn. Across the Optuna protocol, inputs are standardized feature-wise to zero mean and unit variance using `StandardScaler`, then deterministically permuted and split into fixed 70/30 train-holdout partitions. We retain both partitions to evaluate both training-set representation and transfer to unseen samples.

#### 4.1.2 Optuna quality metrics

Our primary quality metric is Quantization Error ($QE$) [@kohonenSelfOrganizingMaps2001]. We report $QE$ on both the training partition ($QE_T$) and the holdout partition ($QE_H$). The training value reflects how well the fitted SOM represents the samples it was trained on, whereas the holdout value evaluates the same trained map after projecting previously unseen samples onto it. The holdout partition is therefore not used during fitting; it is used only to evaluate the trained map, analogous to a test set in conventional model-training workflows.

Balanced $QE$, denoted $QE_B$, is defined as the mean of $QE_T$ and $QE_H$. $QE_B$ is therefore a composite endpoint that weights data representation fidelity (train) and generalisability (holdout) equally. Note that in the Optuna runs, $QE_T$ and $QE_H$ are optimized jointly as a two-objective vector, with $QE_B$ only calculated *post hoc*. 

### 4.2 Speed scaling benchmark protocol

The speed scaling benchmark evaluates runtime and distributed scaling behavior in different compute and algorithm configurations. Speed scaling is evaluated with runs across $G\in\{1,2,4,8\}$ GPUs under a series of fixed scaling protocols. Runtime summaries are computed from repeated executions per configuration and reported as both absolute training time and efficiency relative to the corresponding 1-GPU comparison. We compute scaling efficiency as $E_G=(T_1/T_G)/G \times 100\%$, where $T_1$ is the 1-GPU runtime and $T_G$ is the runtime on $G$ GPUs. A value of 100\% corresponds to ideal linear scaling; lower values indicate that communication, orchestration, or I/O overhead has reduced the realized speedup, while values above 100\% can occur when additional GPUs also change the memory or data-staging regime. These scaling runs use the Ray-orchestrated distributed execution layer built on top of the standard FloatSOM training path [@moritzRayDistributedFramework2018]. Accordingly, the scaling figures in Sections 6.1-6.2 and the runtime/scaling comparison reported later against XPySOM should be interpreted as distributed-execution results rather than the in-memory Optuna path.

For scaling-efficiency calculations, when a single-GPU baseline was missing at a given axis value due to timeout, we estimated that baseline by local linear extrapolation from the last available 1-GPU point on the same curve. That is, runtime was assumed to scale proportionally with the axis variable for the extrapolation step (for example, doubling sample count or doubling dimensionality doubles the estimated 1-GPU runtime).

Topology speed comparisons include hexagonal, MST, and RNG, with harmonized workload settings so ratios isolate topology-associated runtime effects. A dedicated random versus full comparison, run on $G\in\{1,2,4\}$ GPUs, is additionally included to isolate sampling-specific runtime effects independently of the multi-GPU scaling runs.

#### 4.2.1 Scaling benchmark datasets

The speed benchmark uses synthetic random matrices with uniform values in $[0,1]$. No train-holdout split is used here because the objective is runtime rather than generalization, so all generated data are used for training. We use a shared default configuration of $10^7$ samples, 50 dimensions, a $32 \times 32$ SOM grid, and 10 training iterations, and then vary one workload axis at a time: sample count ($10^6$--$10^9$), feature dimension (50--5000), or grid side length (8--64). The exact axis values used for each scaling panel are provided in Supplementary Table S3.

Additional CPU and RAM resources attached to each GPU are scaled linearly with GPU count while keeping the software environment and benchmark procedure consistent across runs. For scaling benchmarks, each run is timed out if it exceeds 30 minutes of wall time. Per-GPU resource allocation is fixed at one NVIDIA V100 (32 GB VRAM), 12 CPU cores, 90 GB system RAM, with one full node comprising 4 GPUs and their associated CPUs, and with 400 GB of associated local disk storage.

### 4.3 Hyperparameter Tuning and Stability

To quantify parameter-tuning benefit, we performed an explicit paired analysis between the tuned configuration and the XPySOM untuned default reference. For each sampling mode and topology combination, we first extracted the parameter settings from the best-performing Optuna runs under the benchmark objective for that combination. We then distilled these per-seed best-performing tuned settings into deployable default configurations by taking the mean of numeric parameters and the mode of categorical parameters.

The stability analysis focuses on four tuned hyperparameters that govern SOM training dynamics. These settings determine how the map is initialized and how updates evolve over training: the initial radius sets the early neighborhood scale around each BMU, the initialization method sets the starting node weights, the radius decay type controls the shift from broad global organization toward local refinement, and the momentum-use parameter determines whether each update retains part of the previous update direction.

#### 4.3.1 Tuned Configuration versus Untuned Reference Analysis

This fixed derived configuration was then rerun and paired against the untuned default XPySOM (hexagonal) reference within matched dataset, seed, and dataset-split units.

Dataset-wise tuned configuration versus untuned reference summaries use the same forest-plot convention. The global overall summary pools all matched tuned configuration versus untuned reference pairs across datasets and applies the same paired $t$-test and confidence-interval construction to that pooled paired set.

#### 4.3.2 Hyperparameter stability and dataset-type stratification

Hyperparameter stability is important because it determines whether a method can be deployed reliably without bespoke retuning. We therefore extracted the top-ranked tuned Optuna trial separately for each seed and compared the recovered hyperparameter values across seeds. We define hyperparameter stability as the variation in these recovered settings. For numeric parameters, stability is measured by the relative difference between runs, $|a-b| / \max(|a|,|b|,\varepsilon)$, where $a$ and $b$ are the values of a given parameter for the two compared seeds and $\varepsilon=10^{-12}$. These relative differences are then averaged across numeric parameters, with lower values indicating higher stability. For categorical parameters, stability is defined as the mismatch rate across the same seed pairs. We report both mean per-parameter stability and an equal-weight overall stability summary across datasets.

### 4.4 Statistical analysis

All Optuna comparisons use matched pairs within dataset, seed, and split units to control for substantial between-run heterogeneity. Within each matched unit, trials were ranked by the target metric, the top five were retained, and each condition was summarized by the median of those retained trials, yielding a top-$k$ summary with $k=5$ intended to estimate near-optimal attainable performance under a fixed Optuna budget. Paired effects were then computed as simple condition differences, with negative values favoring the first condition for lower-is-better metrics. Dataset-level and global summaries are shown as forest plots with 95% confidence intervals from two-sided paired one-sample $t$-tests; top-$k$ sensitivity analyses are provided in the Supplementary figures.

## 5. Results

### 5.1 XPySOM calibration (Equivalence)

Under matched-configuration XPySOM versus FloatSOM calibration on hexagonal $QE$ (Fig. S3), the two implementations perform equivalently, with no detected $QE$ differences. Accordingly, we treat hexagonal FloatSOM batch as a valid proxy for XPySOM in the benchmarks that follow. The runtime differences are attributable to FloatSOM's JIT kernels, which incur a small startup cost. This overhead is progressively amortized as workload size increases, after which FloatSOM runs faster than XPySOM on larger datasets.

### 5.2 Comparison of Different Sampling Methods

We first report a focused HDSSSOM pilot as an elimination comparison rather than as part of the broader sampling benchmark. This pilot used a smaller, more limited configuration than the later full versus random analysis and is summarized in Supplementary Table S2.

Under that pilot configuration, HDSSSOM was materially worse than full sampling. In the hexagonal view shown in Fig. 3, full outperformed HDSSSOM in all 50 paired comparisons (10 datasets $\times$ 5 seeds; 50 wins, 0 losses, 0 ties), with dataset-level median balanced-$QE$ improvements ranging from 2.5% to 209.7% and a global median improvement of 38.7%. Given these large differences, we focused on the full and random sampling methodologies for the remainder of the manuscript.

![Figure 3](assets_manual/figures/fig_3.svg)
*Figure 3. HDSSSOM screening pilot on $QE_B$ (hexagonal topology): full vs HDSSSOM, using the smaller pilot configuration summarized in Supplementary Table S2 (10 datasets, 5 seeds, with all other pilot settings held fixed). Panels report dataset-matched paired top-$k$ within-unit medians plus dataset-level paired-effect summaries (Section 4.3). Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

The full versus random analysis in Fig. 4 was generated from matched hexagonal Optuna runs in which the sampling selector was switched from full to random. Effects were computed within matched seed-specific units and then summarized across seeds. Fig. 4A-C summarize the matched full versus random paired effects for balanced, holdout, and train $QE$, respectively, showing that full sampling generally provides equal or better $QE$ than random sampling. Notably, this full versus random separation is much smaller than the full versus HDSSSOM pilot effect shown in Fig. 3; in most dataset-level comparisons, the full versus HDSSSOM improvement is at least twice as large - as evidenced by the difference in axis scales between Fig. 3 and Fig. 4.

We find the effectiveness of random sampling relative to full sampling is scale-dependent. Above $10{,}000$ samples, paired $QE$ differences are not meaningfully detected, whereas in smaller datasets the random arm shows higher variability and less stable outcomes (Fig. 4D-F). This is consistent with reduced per-iteration sample support under random subsampling (Fig. 4D-F). In this benchmark, the $>10{,}000$ regime is therefore a useful practical proxy for more stable random sampling behavior.

![Figure 4](assets_manual/figures/fig_4.svg)
*Figure 4. Full versus random sampling under the hexagonal topology. Panels A-C report paired full-versus-random effects for $QE_B$, $QE_H$, and $QE_T$. Panels D-F show the corresponding dataset-size regressions for the same three metrics, with numbered datasets keyed in Supplementary Table S1. In the larger-dataset regime observed here (>10,000 samples), little paired $QE$ separation is detected; at smaller dataset scales, random is more variable and less stable.*

Nevertheless, full sampling remains the best sampling strategy for optimal $QE$ results. Accordingly, all remaining analyses reported below rely on full sampling unless specified.

### 5.3 Topology Results

Topology comparisons are reported with $QE$-only metrics, with the Optuna hexagonal batch setting as the primary regular-topology baseline; Fig. 5 provides a qualitative illustration of the neighborhood structures produced by hexagonal, MST, and RNG.

![Figure 5](assets_manual/figures/fig_5.svg)
*Figure 5. Representative neighborhood node and connection overlays for default XPySOM hexagonal, MST, and RNG runs on a 30,000 data-point synthetic sklearn circles dataset.*

#### 5.3.1 MST

We report dataset-wise paired improvement summaries (hexagonal over MST) with the same reporting logic as Section 5.1 in Fig. 6.

<!-- AUTO-TOPOLOGY-MST-PVALUES:START -->
Overall, MST outperforms matched hexagonal on balanced QE (Fig. 6A), indicating a net advantage across train and holdout performance. This aggregate gain is driven more clearly by train QE (Fig. 6C), while holdout QE is more mixed across datasets (Fig. 6B) and shows no clear overall holdout advantage. The overall paired t-test p-values are balanced QE (p=1.12e-05), holdout QE (p=0.15), and train QE (p=0.0064).
<!-- AUTO-TOPOLOGY-MST-PVALUES:END -->

![Figure 6](assets_manual/figures/fig_6.svg)
*Figure 6. Hexagonal versus MST topology on $QE$ metrics under full sampling only. Panels A-C report paired full sampling only $QE$ effects for $QE_B$, $QE_H$, and $QE_T$ across the available full sampling datasets. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

#### 5.3.2 RNG

To evaluate RNG topology performance, we reuse the paired reporting logic on hexagonal versus RNG, again centered on $QE_B$ with $QE_H$ and $QE_T$ in Fig. 7. 

<!-- AUTO-TOPOLOGY-RNG-PVALUES:START -->
RNG has lower QE than matched hexagonal on the reported QE metrics (Fig. 7A-C), with overall paired t-test p-values of balanced QE (p=7.4e-10), holdout QE (p=0.0232), and train QE (p=4.69e-06).
<!-- AUTO-TOPOLOGY-RNG-PVALUES:END -->

The main trend in Fig. 7 is that RNG improves on hexagonal most clearly in balanced QE and especially in train QE, with the separation most apparent in the real and larger datasets where the added flexibility of the graph neighborhood appears more useful than the fixed regular lattice.

![Figure 7](assets_manual/figures/fig_7.svg)
*Figure 7. Hexagonal versus RNG topology on $QE$ metrics under full sampling only. Panels A-C report paired full sampling only $QE$ effects for $QE_B$, $QE_H$, and $QE_T$ across the available full sampling datasets. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

### 5.4 Hyperparameter Tuning and Stability

#### 5.4.1 Performance Gains from Hyperparameter Tuning

To quantify the practical benefit of deploying tuned settings, we compare tuned and untuned configurations in Fig. 8 using pooled topology-level $QE$ summaries. Topology-stratified versions of the same comparison are provided in Figs. S8-S10. The tuned configurations are the Section 4.3 derived settings, and the untuned reference is the default XPySOM configuration. Broadly, tuning yields substantial and significant $QE$ improvement.

![Figure 8](assets_manual/figures/fig_8.svg)
*Figure 8. Tuned configuration versus untuned reference $QE$ comparison across $QE_B$, $QE_H$, and $QE_T$, pooled across all topology runs under the matched pairing keys. Positive values indicate the tuned configuration outperforms the untuned reference; the global overall row pools all matched tuned configuration/untuned reference pairs across datasets. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

At the pooled overall level, the paired summaries across all matched tuned configuration/untuned reference pairs also favor tuning for all three metrics, consistent with the per-dataset pattern in Fig. 8.
<!-- AUTO-DEFAULT-AWARE-TOPOLOGY-STATS:START -->
The same tuning pattern is observed across topologies, indicating that tuning affects all topology families rather than a single-architecture artifact.
<!-- AUTO-DEFAULT-AWARE-TOPOLOGY-STATS:END -->

#### 5.4.2 Hyperparameter Stability Across Topology and Sampling

To assess whether the derived hyperparameters are robust across runs and datasets, we compare the within-topology seed-to-seed tuned-parameter drift (Section 4.3.2). Overall, MST and RNG exhibits less variability in performance compared to hexagons (Fig. 9A using full dataset, 9B using random sampling). Across these panels, full sampling is generally the more stable setting. Mirroring the sample-size-dependent $QE$ performance, Fig. 9C shows that random sampling hyperparameter stability also improves with increasing sample size.

![Figure 9](assets_manual/figures/fig_9.svg)
*Figure 9. Hyperparameter stability by sampling mode. A: selected parameter stability under full sampling for hexagonal, MST, and RNG topologies (lower stability score is better). B: selected parameter stability under random sampling for the same topologies. C: dataset-size stability regression under random sampling, using the selected parameter stability score against sample size (log10) across the included topology families.*

## 6. Speed Scaling

We next examine three aspects of FloatSOM runtime performance: the cost of full relative to random sampling, the scaling behavior obtained with additional GPUs, and the runtime implications of MST and RNG relative to a hexagonal topology.

### 6.1 Random versus Full Sampling Runtime

We report sample scaling runtime comparisons for full versus random sampling using the synthetic datasets as outlined in Section 4.2.1. We stratify by hexagonal, MST, and RNG, for 1, 2, and 4 GPUs (Fig. 10).

![Figure 10](assets_manual/figures/fig_10.svg)

*Figure 10. Sample scaling runtime comparison of full versus random sampling across $G\in\{1,2,4\}$ GPUs (A,B,C). Curves report mean wall-clock training time (s) under harmonized settings; error bars denote $\pm 1$ standard deviation across $n=3$ repeated runs per configuration. Color encodes topology (hexagonal, MST, RNG), and line style encodes sampling mode (full vs. random). Shaded x-axis regions indicate sample-size ranges that could not be run in that panel relative to the shared axis maximum due to timeouts. Lower values indicate faster execution.*

Across topologies, random sampling is faster than full sampling across all 1-, 2-, and 4-GPU comparisons, with similar proportional reductions at a given dataset size. With more GPUs, larger datasets can also be processed before timing out, although the runtime advantage of random over full narrows at the largest sample counts.

For the 1-GPU random sampling runs, the last successful 1-GPU run was at 100M samples, the same number of samples as run on full samples, despite random being much faster. We reason that the failed subsequent runs on larger samples were due to the transition from RAM operation to disk-backed operation. Even in random sampling mode, the implementation still stages the full dataset initially and loads full chunks before the subsampler selects the training samples. Therefore, the relevant cost is no longer only the reduced number of selected samples. We therefore treat the 1-GPU random failure at 500M samples as a disk-mode systems limitation rather than as evidence against the general random versus full runtime ordering.

### 6.2 Multi-GPU topology scaling and Larger-Than-Memory context

To examine parallel scaling, we benchmarked FloatSOM under full sampling across $G\in\{1,2,4,8\}$ GPUs using workload-scaling benchmarks that extend from standard in-memory settings to larger workloads that exceed available memory. Section 6.2 primarily concerns sample scaling. In that setting, the topologies exhibit similar scaling characteristics: as sample count increases, the GPU-count response and efficiency curves have similar qualitative shapes for RNG (Fig. 11B,E) and for the corresponding hexagonal and MST outputs (Fig. S11). Conversely, when the number of SOM nodes is increased (grid-size scaling), the topologies differ substantially. That grid-size regime is analyzed in Section 6.3, where MST and RNG take 8.19x and 27.07x the hexagonal runtime, respectively, at the largest tested grid size.

![Figure 11](assets_manual/figures/fig_11.svg)

*Figure 11. Multi-GPU scaling across $G\in\{1,2,4,8\}$ GPUs. Panels A-C show runtime (s) for dimension, sample, and grid size scaling workloads, respectively. Panels D-F show scaling efficiency for the same workloads, computed from the single-GPU baseline and the corresponding $G$-GPU runtime. Runtime error bars denote $\pm 1$ standard deviation across $n=3$ repeated runs per configuration; the 100\% efficiency reference line indicates ideal linear scaling.*

#### 6.2.1 GPU Scaling and Larger-Than-Memory Runtime Performance

<!-- AUTO-SYSTEMS-SCALING-STATS:START -->
Fig. 11 shows that increasing GPU count improves performance in the sample scaling regime by increasing parallel throughput and delaying the transition to disk-backed execution. In the sample scaling benchmark, the 500,000,000 sample dataset requires disk backing under the 2-GPU configuration, whereas the 8-GPU configuration remains in RAM mode until the 1,000,000,000 sample dataset; when staging is still required, the disk-to-GPU path is distributed across more nodes. The 8-GPU RNG configuration processes 1,000,000,000 samples in 369.41 s (6.16 min). Note that this speed includes the time required to transfer data from shared storage to node-local shards, alongside the overhead associated with operating across multiple HPC nodes.

The grid size scaling panel proves to be the main exception: at the largest tested grid size (64), runtime shortens from 934.01 s (15.57 min) on 1 GPU to only 880.83 s (14.68 min) on 8 GPUs, a 5.69% reduction, indicating that once map-size/topology-refresh costs dominate, additional GPUs contribute little extra speedup.

<!-- AUTO-SYSTEMS-SCALING-STATS:END -->

#### 6.2.2 GPU Scaling Efficiency

We next consider GPU efficiency under strong scaling, relative to ideal linear scaling. At smaller dataset sizes, efficiency is lower. As workload size increases, efficiency rises sharply and in some regions exceeds 100\%. When a direct 1-GPU baseline was unavailable at a given axis value, the efficiency denominator was constructed by local linear extrapolation from the last available 1-GPU point on that curve (Section 4.2), so some values should be interpreted with care if the underlying 1-GPU runtime is nonlinear over that range. Together, Fig. 11 and Fig. S11 support the same qualitative sample-scaling efficiency trend across topologies. This does not extend to grid-size scaling; Section 6.3 shows that when node count increases, topology-dependent absolute runtimes diverge.

### 6.3 Topology Runtime Comparisons

With that systems context in place, we next compare topology runtimes across hexagonal, MST, and RNG configurations on 8 GPUs (Fig. 12).

![Figure 12](assets_manual/figures/fig_12.svg)

*Figure 12. Topology runtime comparison at fixed $G=8$ GPUs under full sampling. Panels A-C report mean wall-clock runtime (s) for dimension, sample, and grid size scaling workloads, respectively, with topology traces for hexagonal, MST, and RNG. Error bars denote $\pm 1$ standard deviation across $n=3$ repeated runs per configuration. The largest-axis 8-GPU topology runtime summaries are listed in Supplementary Table S11.*

<!-- AUTO-FIGURE12-TOPOLOGY-RUNTIME-STATS:START -->
In Fig. 12A-B, the topologies scale similarly as input complexity and data volume increase: even at the largest tested axis values, the maximum pairwise runtime spread remains modest at dimension scaling (4.70% at 5,000 dimensions) and sample scaling (3.26% at 1,000,000,000 samples).
<!-- AUTO-FIGURE12-TOPOLOGY-RUNTIME-STATS:END -->

<!-- AUTO-FIGURE12-GRID-SIZE-DISCUSSION:START -->
However, when the grid itself is enlarged in Fig. 12C, topology-dependent runtime differences become readily evident. At the largest tested grid size (grid size 64), the 8-GPU mean runtimes are 32.54 s (0.54 min) for hexagonal, 266.45 s (4.44 min) for MST, and 880.83 s (14.68 min) for RNG, corresponding to 8-GPU MST and RNG runtimes that are 8.19x and 27.07x the hexagonal runtime, respectively.
<!-- AUTO-FIGURE12-GRID-SIZE-DISCUSSION:END -->

## 7. Final FloatSOM RNG Comparison with XPySOM

Fig. 13 demonstrates the $QE$ gains from topology choice and tuning persist in deployment against default hexagonal XPySOM.  [@manciniXPySomHighPerformanceSelfOrganizing2020].

![Figure 13](assets_manual/figures/fig_13.svg)

*Figure 13. Integrated deployment comparison of default hexagonal XPySOM versus tuned FloatSOM RNG. Panels A-C compare $QE_B$, $QE_H$, and $QE_T$ using the untuned hexagonal XPySOM baseline against matched tuned FloatSOM RNG full sampling runs. Panel D provides the scaling/runtime context for the same comparison.*
<!-- AUTO-FIGURE13-DEPLOYMENT-QE-STATS:START -->
At the overall level, Fig. 13 shows median percentage improvements of $QE_B$ (14.5%); $QE_H$ (9.1%); and $QE_T$ (22.5%) for tuned FloatSOM RNG relative to default hexagonal XPySOM, capturing the combined deployment effect of topology choice and tuning on $QE$.
<!-- AUTO-FIGURE13-DEPLOYMENT-QE-STATS:END -->

For the default hexagonal XPySOM reference in Fig. 13, workloads beyond the $10^8$-sample case were not processed because they exceeded available VRAM and XPySOM requires the full dataset to be loaded into memory. Taken together, Fig. 13 shows the point at which the workload size is large enough to warrant the extra startup overhead incurred by tuned FloatSOM RNG: tuned FloatSOM RNG delivers better $QE$ than the default hexagonal XPySOM baseline, while also running faster and scaling to larger workloads.

## 8. Discussion

This manuscript introduces FloatSOM as a GPU-oriented SOM framework that combines topology flexibility, out-of-memory execution, and distributed multi-GPU training in a single implementation. The study also provides a unified empirical analysis of sampling strategy, graph-based versus fixed-lattice topology, topology-aware hyperparameter tuning, and workload scaling across synthetic and real benchmarks. Taken together, these results clarify how algorithmic and systems choices jointly shape SOM quality and runtime under practical deployment constraints.

### 8.1 Sampling tradeoff (random versus full)

The sampling trade-off is strongly scale dependent. In smaller datasets, random subsampling produces less stable outcomes, whereas above $10{,}000$ samples paired $QE$ differences between full and random are not meaningfully detected. Full sampling is therefore the safer choice when stability is the priority, while random sampling is better viewed as a throughput-oriented option at larger scales. In the larger-than-memory regime, this advantage narrows because random still requires each worker-local chunk to be read before subsampling, so it reduces compute more directly than dataset I/O.

### 8.2 Topology Comparisons (MST and RNG)

Globally, both MST and RNG outperform the fixed hexagonal topology in these comparisons, with RNG showing the strongest overall $QE$ results. One interpretation is that this ordering reflects increasing structural flexibility across the topology families, with the more flexible graph-based neighborhoods conforming more effectively to the underlying data distribution than the fixed lattice baseline [@kohonenEssentialsSelforganizingMap2013; @kangasVariantsSelforganizingMaps1990; @toussaintRelativeNeighbourhoodGraph1980]. The denser connected structure available under RNG may also provide additional regularization, because nodes can receive information from more neighbors during updating rather than being limited to a single tree path. This could support more precise local updates, although the present benchmark does not isolate that mechanism directly.

### 8.3 Tuning benefit under matched defaults

Across the matched Fig. 8 comparisons, tuned configurations consistently outperform untuned reference settings. Hyperparameter tuning should therefore be treated as part of the method configuration rather than as optional post-processing.

The key implication is that topology choice and hyperparameter choice are coupled. Gains remain positive across hexagonal, MST, and RNG, so tuning is not confined to a single topology. A practical deployment strategy therefore requires topology-aware tuning.

### 8.4 Hyperparameter stability and dataset-type interpretation

The stability results suggest that graph topologies are recovered more consistently than the fixed-lattice baseline. Hexagonal maps cannot rebuild connectivity once initialized, whereas MST and RNG recompute connectivity from the evolving node-weight geometry. It may also help explain why the graph topologies are better suited to higher-dimensional, non-synthetic datasets [@kohonenEssentialsSelforganizingMap2013; @kangasVariantsSelforganizingMaps1990].

### 8.5 Systems implications and limits

Distributed execution should therefore be interpreted as workload-dependent rather than as a uniform multiplicative speedup. At small problem sizes, Ray startup, communication, and staging overheads appear to dominate, consistent with the lower efficiencies observed at the low end of the scaling curves. At larger workloads, the sharp rise in efficiency likely reflects both increased parallel throughput and a change in memory regime, where some multi-GPU configurations remain in RAM while the 1-GPU baseline has already entered disk-backed execution. Apparent efficiencies above 100\% should therefore be interpreted as reflecting this systems transition rather than literal super-linear compute scaling.

Overall, these results support a practical deployment strategy that uses RNG with topology-aware tuned defaults. For optimal speed, consider using the largest number of GPUs available, especially if this enables data to be kept in RAM. Sampling should be chosen by scale: full for smaller datasets when stability is critical, and random as a throughput-oriented option in the larger-dataset regime where optimal $QE$ is not essential. For workloads dominated by very large grid size scaling, MST remains a reasonable alternative to RNG and Hexagonal.

This manuscript presents FloatSOM as a unified large-scale SOM framework that combines a novel graph-based topology with sampling options, optimised hyperparameters, and distributed out-of-memory GPU execution. This integrated design supports practical SOM deployment at scale, where topology choice, sampling, quantization performance, and systems constraints can be managed together rather than in isolation.

## 9. Acknowledgements

This work was supported by computational resources provided by the Australian Government through the National Computational Infrastructure (NCI) under the ANU Merit Allocation Scheme.

We also acknowledge the computational services provided by the University of Bern, the University of Sydney, and the Walter and Eliza Hall Institute.

We thank Prof. Hanna Suominen for her input and advice.

## 10. References

::: {#refs}
:::

## 11. Supplementary Tables (End Matter)

**Supplementary Table S1. Dataset metadata and numbered point key for the Figure 4 sampling mode analysis.**

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
| configuration scope | Smaller hexagonal Optuna pilot |
| datasets | `swiss_roll`, `moons`, `circles`, `blobs`, `s_curve`, `breast_cancer`, `wine`, `iris`, `digits`, `olivetti_faces` |
| dataset count | 10 |
| seed count | 5 |
| sampling methods present | full, random, HDSSSOM |
| topology | `hexagonal` |
| algorithm family in campaign | `colors` |
| optimization split setup | `evaluation-split=both`, yielding `QE_H` and `QE_T`; Figure 3 reports paired `QE_B` |
| trials per scenario | 200 |
| main text figure slice | hexagonal / full vs HDSSSOM |

**Supplementary Table S3. Exact axis values used for the sample count, feature dimension, and grid size scaling benchmarks in Section 4.2.1.**

| scaling axis | exact values |
| --- | --- |
| sample count | $10^6$, $5 \times 10^6$, $10^7$, $5 \times 10^7$, $10^8$, $5 \times 10^8$, $10^9$ |
| feature dimension | 50, 100, 200, 500, 1000, 2000, 5000 |
| grid side length | 8, 16, 24, 32, 48, 64 |

**Supplementary Table S4. FloatSOM versus XPySOM calibration summary for the MST topology path.** The `dataset_index` column matches the numbered points in Supplementary Figure S1 panel D. Positive percentage values and positive signed effects favor FloatSOM; the compact embedded table lists wins as FloatSOM/XPySOM/ties. The full machine-readable table is also provided in `assets/tables/supp_xpysom_calibration_qe_mst.tsv`.


| idx | dataset | metric | split | wins F/X/tie | median % | mean % | 95% CI | p | n | signed effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | blobs | QE | train | 0/10/0 | -3.6521 | -3.5612 | [-4.6291, -2.687] | 0.002 | 10 | -1 |
| 5 | breast_cancer | QE | train | 10/0/0 | 6.0461 | 5.7839 | [5.0244, 6.3682] | 0.002 | 10 | 1 |
| 7 | california_housing | QE | train | 10/0/0 | 4.2199 | 4.3194 | [3.7617, 4.9195] | 0.002 | 10 | 1 |
| 9 | circles | QE | train | 0/10/0 | -1.7629 | -2.1879 | [-3.655, -1.0873] | 0.002 | 10 | -1 |
| 14 | covertype | QE | train | 10/0/0 | 15.3837 | 15.3479 | [13.9347, 16.925] | 0.002 | 10 | 1 |
| 4 | diabetes | QE | train | 10/0/0 | 7.0091 | 7.0306 | [6.1295, 7.9882] | 0.002 | 10 | 1 |
| 6 | digits | QE | train | 10/0/0 | 5.3038 | 5.3987 | [5.0889, 5.8056] | 0.002 | 10 | 1 |
| 1 | iris | QE | train | 10/0/0 | 28.4174 | 28.3671 | [24.2551, 32.3913] | 0.002 | 10 | 1 |
| 13 | kddcup99 | QE | train | 10/0/0 | 19.4133 | 19.693 | [16.1432, 23.2133] | 0.002 | 10 | 1 |
| 10 | moons | QE | train | 1/9/0 | -3.7058 | -3.3568 | [-5.5241, -1.575] | 0.0098 | 10 | -0.8 |
| 3 | olivetti_faces | QE | train | 10/0/0 | 8.4513 | 8.5474 | [7.1779, 9.6246] | 0.002 | 10 | 1 |
| 11 | s_curve | QE | train | 10/0/0 | 3.3827 | 3.4166 | [1.906, 5.1934] | 0.002 | 10 | 1 |
| 12 | swiss_roll | QE | train | 10/0/0 | 5.271 | 5.0983 | [3.8948, 6.3949] | 0.002 | 10 | 1 |
| 2 | wine | QE | train | 10/0/0 | 20.0992 | 20.1816 | [18.2671, 22.1871] | 0.002 | 10 | 1 |
| - | GLOBAL | QE | train | 111/29/0 | 6.8961 | 8.1485 | [5.6474, 9.0003] | 1.49e-17 | 140 | 0.5857 |
| 8 | blobs | QE | holdout | 0/10/0 | -4.1422 | -3.9793 | [-5.2037, -2.9583] | 0.002 | 10 | -1 |
| 5 | breast_cancer | QE | holdout | 2/8/0 | -0.6423 | -0.4735 | [-1.2282, 0.3513] | 0.1309 | 10 | -0.6 |
| 7 | california_housing | QE | holdout | 10/0/0 | 3.8135 | 3.8367 | [3.1391, 4.5531] | 0.002 | 10 | 1 |
| 9 | circles | QE | holdout | 0/10/0 | -2.5762 | -3.1314 | [-4.9792, -1.4919] | 0.002 | 10 | -1 |
| 14 | covertype | QE | holdout | 10/0/0 | 15.4626 | 15.3225 | [13.8381, 16.8307] | 0.002 | 10 | 1 |
| 4 | diabetes | QE | holdout | 4/6/0 | -0.1376 | 0.0531 | [-1.2671, 1.4709] | 1 | 10 | -0.2 |
| 6 | digits | QE | holdout | 10/0/0 | 2.5919 | 2.7034 | [2.2909, 3.2567] | 0.002 | 10 | 1 |
| 1 | iris | QE | holdout | 4/6/0 | 0.1297 | -0.4991 | [-4.9164, 4.9562] | 1 | 10 | -0.2 |
| 13 | kddcup99 | QE | holdout | 10/0/0 | 19.2029 | 19.4801 | [15.8968, 23.1041] | 0.002 | 10 | 1 |
| 10 | moons | QE | holdout | 1/9/0 | -3.6371 | -3.4648 | [-5.2819, -1.8559] | 0.0059 | 10 | -0.8 |
| 3 | olivetti_faces | QE | holdout | 10/0/0 | 1.915 | 1.8944 | [1.55, 2.2547] | 0.002 | 10 | 1 |
| 11 | s_curve | QE | holdout | 10/0/0 | 3.1294 | 3.2429 | [1.9458, 4.4824] | 0.002 | 10 | 1 |
| 12 | swiss_roll | QE | holdout | 10/0/0 | 5.1413 | 5.1294 | [3.9965, 6.4501] | 0.002 | 10 | 1 |
| 2 | wine | QE | holdout | 7/3/0 | 0.1337 | -0.0378 | [-1.2204, 0.9745] | 0.8457 | 10 | 0.4 |
| - | GLOBAL | QE | holdout | 88/52/0 | 1.7869 | 2.8626 | [0.8691, 2.7426] | 6.33e-05 | 140 | 0.2571 |
| 8 | blobs | QE_B | both | 0/10/0 | -3.91 | -3.7703 | [-4.93, -2.7417] | 0.002 | 10 | -1 |
| 5 | breast_cancer | QE_B | both | 10/0/0 | 2.4588 | 2.4478 | [1.8439, 2.987] | 0.002 | 10 | 1 |
| 7 | california_housing | QE_B | both | 10/0/0 | 3.9644 | 4.0771 | [3.6418, 4.7324] | 0.002 | 10 | 1 |
| 9 | circles | QE_B | both | 0/10/0 | -2.0639 | -2.6605 | [-4.3931, -1.3869] | 0.002 | 10 | -1 |
| 14 | covertype | QE_B | both | 10/0/0 | 15.3975 | 15.3352 | [13.8864, 16.8854] | 0.002 | 10 | 1 |
| 4 | diabetes | QE_B | both | 10/0/0 | 3.1657 | 3.1796 | [2.2103, 4.1719] | 0.002 | 10 | 1 |
| 6 | digits | QE_B | both | 10/0/0 | 3.997 | 4.0139 | [3.6762, 4.3228] | 0.002 | 10 | 1 |
| 1 | iris | QE_B | both | 10/0/0 | 9.1407 | 9.5616 | [5.9156, 13.6397] | 0.002 | 10 | 1 |
| 13 | kddcup99 | QE_B | both | 10/0/0 | 19.3064 | 19.5854 | [16.0187, 23.1588] | 0.002 | 10 | 1 |
| 10 | moons | QE_B | both | 1/9/0 | -3.4326 | -3.4102 | [-5.2683, -1.7163] | 0.0059 | 10 | -0.8 |
| 3 | olivetti_faces | QE_B | both | 10/0/0 | 4.8444 | 4.8685 | [4.365, 5.4342] | 0.002 | 10 | 1 |
| 11 | s_curve | QE_B | both | 10/0/0 | 3.292 | 3.3299 | [1.8567, 4.777] | 0.002 | 10 | 1 |
| 12 | swiss_roll | QE_B | both | 10/0/0 | 5.2033 | 5.1138 | [4.0306, 6.4161] | 0.002 | 10 | 1 |
| 2 | wine | QE_B | both | 10/0/0 | 7.8522 | 7.7725 | [6.9044, 8.686] | 0.002 | 10 | 1 |
| - | GLOBAL | QE_B | both | 111/29/0 | 4.2807 | 4.9603 | [3.4978, 5.2317] | 1.35e-13 | 140 | 0.5857 |
| 8 | blobs | train time | train | 0/10/0 | -730.1165 | -757.4247 | [-864.8227, -727.7904] | 0.002 | 10 | -1 |
| 5 | breast_cancer | train time | train | 0/10/0 | -1011.2684 | -1004.5725 | [-1016.3642, -976.4259] | 0.002 | 10 | -1 |
| 7 | california_housing | train time | train | 0/10/0 | -972.0213 | -950.4554 | [-1011.7249, -888.6339] | 0.002 | 10 | -1 |
| 9 | circles | train time | train | 0/10/0 | -728.984 | -728.8823 | [-733.0182, -724.2827] | 0.002 | 10 | -1 |
| 14 | covertype | train time | train | 0/10/0 | -37.4988 | -37.8968 | [-42.3539, -32.87] | 0.002 | 10 | -1 |
| 4 | diabetes | train time | train | 0/10/0 | -1010.0585 | -1043.324 | [-1183.0245, -1002.8581] | 0.002 | 10 | -1 |
| 6 | digits | train time | train | 0/10/0 | -1004.8808 | -1005.1876 | [-1011.5931, -998.5865] | 0.002 | 10 | -1 |
| 1 | iris | train time | train | 0/10/0 | -1037.1907 | -1036.0797 | [-1042.8531, -1029.1162] | 0.002 | 10 | -1 |
| 13 | kddcup99 | train time | train | 0/10/0 | -36.9051 | -35.1756 | [-38.6464, -32.0512] | 0.002 | 10 | -1 |
| 10 | moons | train time | train | 0/10/0 | -734.37 | -734.8929 | [-740.0781, -731.5506] | 0.002 | 10 | -1 |
| 3 | olivetti_faces | train time | train | 0/10/0 | -1019.8923 | -1016.4567 | [-1030.2974, -1000.3188] | 0.002 | 10 | -1 |
| 11 | s_curve | train time | train | 0/10/0 | -736.215 | -736.5435 | [-741.9456, -732.008] | 0.002 | 10 | -1 |
| 12 | swiss_roll | train time | train | 1/9/0 | -726.5489 | -649.8592 | [-732.2277, -336.7629] | 0.0039 | 10 | -0.8 |
| 2 | wine | train time | train | 0/10/0 | -1030.4764 | -1064.4839 | [-1204.4846, -1025.0949] | 0.002 | 10 | -1 |
| - | GLOBAL | train time | train | 1/139/0 | -868.8976 | -771.5168 | [-875.2748, -760.3798] | 1.59e-24 | 140 | -0.9857 |


**Supplementary Table S5. FloatSOM versus XPySOM calibration summary for the RNG topology path.** The `dataset_index` column matches the numbered points in Supplementary Figure S2 panel D. Positive percentage values and positive signed effects favor FloatSOM; the compact embedded table lists wins as FloatSOM/XPySOM/ties. The full machine-readable table is also provided in `assets/tables/supp_xpysom_calibration_qe_rng.tsv`.


| idx | dataset | metric | split | wins F/X/tie | median % | mean % | 95% CI | p | n | signed effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | blobs | QE | train | 0/10/0 | -2.7298 | -2.6927 | [-3.7585, -1.6597] | 0.002 | 10 | -1 |
| 5 | breast_cancer | QE | train | 10/0/0 | 5.4998 | 5.465 | [5.0315, 5.9377] | 0.002 | 10 | 1 |
| 7 | california_housing | QE | train | 10/0/0 | 3.6187 | 3.641 | [3.0746, 4.2359] | 0.002 | 10 | 1 |
| 9 | circles | QE | train | 1/9/0 | -1.7013 | -2.1091 | [-3.9805, -0.3778] | 0.0137 | 10 | -0.8 |
| 14 | covertype | QE | train | 10/0/0 | 13.6239 | 13.5016 | [10.9438, 15.887] | 0.002 | 10 | 1 |
| 4 | diabetes | QE | train | 10/0/0 | 6.2882 | 6.3332 | [5.2073, 7.5187] | 0.002 | 10 | 1 |
| 6 | digits | QE | train | 10/0/0 | 4.4749 | 4.5437 | [4.073, 5.0017] | 0.002 | 10 | 1 |
| 1 | iris | QE | train | 10/0/0 | 25.6985 | 25.9471 | [20.7469, 31.2163] | 0.002 | 10 | 1 |
| 13 | kddcup99 | QE | train | 10/0/0 | 18.8172 | 18.2227 | [14.5027, 22.0324] | 0.002 | 10 | 1 |
| 10 | moons | QE | train | 2/8/0 | -2.6397 | -2.3844 | [-4.0907, -0.5701] | 0.0195 | 10 | -0.6 |
| 3 | olivetti_faces | QE | train | 10/0/0 | 6.9757 | 7.4422 | [5.9654, 9.3791] | 0.002 | 10 | 1 |
| 11 | s_curve | QE | train | 10/0/0 | 3.0675 | 2.9986 | [1.5732, 4.5318] | 0.002 | 10 | 1 |
| 12 | swiss_roll | QE | train | 10/0/0 | 5.7819 | 5.3025 | [4.1229, 6.2356] | 0.002 | 10 | 1 |
| 2 | wine | QE | train | 10/0/0 | 18.7804 | 18.6217 | [15.9669, 21.416] | 0.002 | 10 | 1 |
| - | GLOBAL | QE | train | 113/27/0 | 6.1574 | 7.4881 | [5.0617, 8.1204] | 5.8e-18 | 140 | 0.6143 |
| 8 | blobs | QE | holdout | 0/10/0 | -3.0403 | -2.9146 | [-3.8854, -1.8988] | 0.002 | 10 | -1 |
| 5 | breast_cancer | QE | holdout | 3/7/0 | -0.1679 | -0.0809 | [-0.6344, 0.5026] | 0.4316 | 10 | -0.4 |
| 7 | california_housing | QE | holdout | 10/0/0 | 3.5473 | 3.4616 | [2.8161, 3.9656] | 0.002 | 10 | 1 |
| 9 | circles | QE | holdout | 2/8/0 | -2.435 | -2.5817 | [-4.2591, -0.7108] | 0.0098 | 10 | -0.6 |
| 14 | covertype | QE | holdout | 10/0/0 | 13.5339 | 13.4773 | [11.0872, 15.767] | 0.002 | 10 | 1 |
| 4 | diabetes | QE | holdout | 8/2/0 | 0.993 | 0.9845 | [0.0524, 1.9348] | 0.0488 | 10 | 0.6 |
| 6 | digits | QE | holdout | 10/0/0 | 2.6213 | 2.6133 | [2.0362, 3.1169] | 0.002 | 10 | 1 |
| 1 | iris | QE | holdout | 3/7/0 | -1.3579 | -1.2948 | [-4.6315, 2.0699] | 0.375 | 10 | -0.4 |
| 13 | kddcup99 | QE | holdout | 10/0/0 | 18.8346 | 18.178 | [14.6321, 21.9247] | 0.002 | 10 | 1 |
| 10 | moons | QE | holdout | 1/9/0 | -2.9592 | -2.5631 | [-4.2128, -0.9289] | 0.0137 | 10 | -0.8 |
| 3 | olivetti_faces | QE | holdout | 8/2/0 | 1.5548 | 1.5684 | [0.0193, 3.2179] | 0.0371 | 10 | 0.6 |
| 11 | s_curve | QE | holdout | 10/0/0 | 2.8467 | 3.0331 | [1.8244, 4.3726] | 0.002 | 10 | 1 |
| 12 | swiss_roll | QE | holdout | 10/0/0 | 5.5918 | 5.3626 | [3.9067, 6.5351] | 0.002 | 10 | 1 |
| 2 | wine | QE | holdout | 5/5/0 | -0.0877 | -0.2419 | [-1.2255, 0.6779] | 0.6953 | 10 | 0 |
| - | GLOBAL | QE | holdout | 90/50/0 | 1.7513 | 2.7858 | [0.9907, 2.6251] | 1.02e-05 | 140 | 0.2857 |
| 8 | blobs | QE_B | both | 0/10/0 | -2.8719 | -2.8032 | [-3.8029, -1.9134] | 0.002 | 10 | -1 |
| 5 | breast_cancer | QE_B | both | 10/0/0 | 2.4644 | 2.4959 | [2.075, 2.9683] | 0.002 | 10 | 1 |
| 7 | california_housing | QE_B | both | 10/0/0 | 3.6308 | 3.5512 | [2.9851, 4.109] | 0.002 | 10 | 1 |
| 9 | circles | QE_B | both | 2/8/0 | -2.1386 | -2.3461 | [-3.9949, -0.5829] | 0.0098 | 10 | -0.6 |
| 14 | covertype | QE_B | both | 10/0/0 | 13.6109 | 13.4894 | [11.0155, 15.826] | 0.002 | 10 | 1 |
| 4 | diabetes | QE_B | both | 10/0/0 | 3.3183 | 3.3817 | [2.369, 4.1991] | 0.002 | 10 | 1 |
| 6 | digits | QE_B | both | 10/0/0 | 3.5551 | 3.5512 | [3.034, 3.9851] | 0.002 | 10 | 1 |
| 1 | iris | QE_B | both | 10/0/0 | 7.914 | 8.1514 | [4.7671, 10.8707] | 0.002 | 10 | 1 |
| 13 | kddcup99 | QE_B | both | 10/0/0 | 18.8261 | 18.1997 | [14.5584, 21.9789] | 0.002 | 10 | 1 |
| 10 | moons | QE_B | both | 2/8/0 | -2.8707 | -2.4735 | [-4.1471, -0.7217] | 0.0195 | 10 | -0.6 |
| 3 | olivetti_faces | QE_B | both | 10/0/0 | 4.1052 | 4.1971 | [2.6366, 5.7848] | 0.002 | 10 | 1 |
| 11 | s_curve | QE_B | both | 10/0/0 | 2.9385 | 3.0161 | [1.6726, 4.3754] | 0.002 | 10 | 1 |
| 12 | swiss_roll | QE_B | both | 10/0/0 | 5.7209 | 5.3327 | [4.0146, 6.334] | 0.002 | 10 | 1 |
| 2 | wine | QE_B | both | 10/0/0 | 7.1548 | 7.0442 | [5.7434, 8.3772] | 0.002 | 10 | 1 |
| - | GLOBAL | QE_B | both | 114/26/0 | 3.9446 | 4.6277 | [3.2069, 4.8515] | 1.27e-14 | 140 | 0.6286 |
| 8 | blobs | train time | train | 0/10/0 | -725.754 | -726.3513 | [-729.2884, -724.0566] | 0.002 | 10 | -1 |
| 5 | breast_cancer | train time | train | 0/10/0 | -1028.7409 | -1021.2214 | [-1033.9516, -990.4514] | 0.002 | 10 | -1 |
| 7 | california_housing | train time | train | 0/10/0 | -959.1137 | -935.4774 | [-997.3973, -863.1072] | 0.002 | 10 | -1 |
| 9 | circles | train time | train | 0/10/0 | -728.9236 | -729.3583 | [-734.228, -723.7248] | 0.002 | 10 | -1 |
| 14 | covertype | train time | train | 0/10/0 | -38.0038 | -37.1495 | [-41.0157, -32.7859] | 0.002 | 10 | -1 |
| 4 | diabetes | train time | train | 0/10/0 | -1029.3356 | -1028.7558 | [-1034.5884, -1022.6478] | 0.002 | 10 | -1 |
| 6 | digits | train time | train | 0/10/0 | -1029.1819 | -1028.6713 | [-1034.196, -1022.846] | 0.002 | 10 | -1 |
| 1 | iris | train time | train | 0/10/0 | -1030.5871 | -1029.7524 | [-1035.7818, -1023.3311] | 0.002 | 10 | -1 |
| 13 | kddcup99 | train time | train | 0/10/0 | -36.6803 | -35.3473 | [-39.0245, -32.0055] | 0.002 | 10 | -1 |
| 10 | moons | train time | train | 0/10/0 | -731.1551 | -730.6761 | [-733.0006, -727.7656] | 0.002 | 10 | -1 |
| 3 | olivetti_faces | train time | train | 0/10/0 | -1054.7323 | -1050.5264 | [-1064.173, -1034.3248] | 0.002 | 10 | -1 |
| 11 | s_curve | train time | train | 0/10/0 | -749.2016 | -800.7466 | [-1011.2814, -744.996] | 0.002 | 10 | -1 |
| 12 | swiss_roll | train time | train | 1/9/0 | -741.1606 | -661.2161 | [-745.7479, -339.5738] | 0.0039 | 10 | -0.8 |
| 2 | wine | train time | train | 0/10/0 | -1040.5015 | -1039.5076 | [-1043.8579, -1034.1717] | 0.002 | 10 | -1 |
| - | GLOBAL | train time | train | 1/139/0 | -877.7086 | -775.3398 | [-884.2881, -753.573] | 1.59e-24 | 140 | -0.9857 |


**Supplementary Table S6. FloatSOM versus XPySOM calibration summary for the hexagonal topology path.** The `dataset_index` column matches the numbered points in Supplementary Figure S3 panel D. Positive percentage values and positive signed effects favor FloatSOM; the compact embedded table lists wins as FloatSOM/XPySOM/ties. The full machine-readable table is also provided in `assets/tables/supp_xpysom_calibration_qe_hexagonal.tsv`.


| idx | dataset | metric | split | wins F/X/tie | median % | mean % | 95% CI | p | n | signed effect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | blobs | QE | train | 7/3/0 | 0.0042 | 0.0034 | [-0.0023, 0.0089] | 0.1602 | 10 | 0.4 |
| 5 | breast_cancer | QE | train | 1/1/8 | 1.46e-08 | 2.92e-09 | [-1.04e-05, 1.04e-05] | 1 | 10 | 0 |
| 7 | california_housing | QE | train | 7/1/2 | 1.13e-05 | 0.0016 | [-0.0046, 0.0095] | 0.1484 | 10 | 0.75 |
| 9 | circles | QE | train | 6/4/0 | -0.0156 | -0.0142 | [-0.0446, 0.0107] | 0.5566 | 10 | 0.2 |
| 14 | covertype | QE | train | 3/7/0 | -0.000333 | -0.0021 | [-0.0152, 0.0059] | 0.3223 | 10 | -0.4 |
| 4 | diabetes | QE | train | 2/2/6 | -8.76e-08 | -3.5e-08 | [-8.88e-06, 8.68e-06] | 0.625 | 10 | 0 |
| 6 | digits | QE | train | 0/2/8 | -1.08e-05 | -2.16e-06 | [-1.08e-05, -1.08e-05] | 0.5 | 10 | -1 |
| 1 | iris | QE | train | 3/1/6 | 7.85e-06 | 1.66e-06 | [-7.64e-06, 8.43e-06] | 0.25 | 10 | 0.5 |
| 13 | kddcup99 | QE | train | 4/6/0 | -0.037 | -0.2112 | [-0.5893, 0.0103] | 0.1309 | 10 | -0.2 |
| 10 | moons | QE | train | 8/2/0 | 0.0021 | 0.0114 | [-0.000313, 0.0481] | 0.084 | 10 | 0.6 |
| 3 | olivetti_faces | QE | train | 2/4/4 | -0.6875 | -0.3985 | [-1.5487, 0.3296] | 0.2188 | 10 | -0.3333 |
| 11 | s_curve | QE | train | 9/1/0 | 0.0024 | 0.0041 | [1.78e-05, 0.0092] | 0.0195 | 10 | 0.8 |
| 12 | swiss_roll | QE | train | 6/4/0 | 0.0019 | 0.0019 | [-0.0046, 0.0087] | 0.625 | 10 | 0.2 |
| 2 | wine | QE | train | 0/1/9 | -1.02e-05 | -1.02e-06 | [-1.02e-05, -1.02e-05] | 1 | 10 | -1 |
| - | GLOBAL | QE | train | 58/39/43 | 1.51e-05 | -0.0431 | [-0.000899, 0.0016] | 0.484 | 140 | 0.1959 |
| 8 | blobs | QE | holdout | 7/3/0 | 0.0026 | 0.0032 | [-0.001, 0.0074] | 0.2324 | 10 | 0.4 |
| 5 | breast_cancer | QE | holdout | 1/1/8 | -2.38e-07 | -4.76e-08 | [-9.3e-06, 8.82e-06] | 1 | 10 | 0 |
| 7 | california_housing | QE | holdout | 7/2/1 | 7.32e-06 | -0.0011 | [-0.0113, 0.0069] | 0.4961 | 10 | 0.5556 |
| 9 | circles | QE | holdout | 6/4/0 | 0.0017 | 0.0011 | [-0.0504, 0.0522] | 0.9219 | 10 | 0.2 |
| 14 | covertype | QE | holdout | 3/7/0 | -0.000325 | -0.0022 | [-0.0154, 0.0059] | 0.3223 | 10 | -0.4 |
| 4 | diabetes | QE | holdout | 0/2/8 | -7.11e-06 | -1.42e-06 | [-7.17e-06, -7.05e-06] | 0.5 | 10 | -1 |
| 6 | digits | QE | holdout | 1/1/8 | 6.39e-08 | 1.28e-08 | [-1.05e-05, 1.06e-05] | 1 | 10 | 0 |
| 1 | iris | QE | holdout | 4/2/4 | 5.77e-06 | 3.63e-06 | [-8.4e-06, 1.94e-05] | 0.4375 | 10 | 0.3333 |
| 13 | kddcup99 | QE | holdout | 4/6/0 | -0.0381 | -0.2277 | [-0.6099, 0.0098] | 0.1602 | 10 | -0.2 |
| 10 | moons | QE | holdout | 6/3/1 | 0.0026 | 0.0049 | [-0.009, 0.0308] | 0.4961 | 10 | 0.3333 |
| 3 | olivetti_faces | QE | holdout | 4/2/4 | 0.2291 | 0.1813 | [-0.3845, 1.4392] | 0.3125 | 10 | 0.3333 |
| 11 | s_curve | QE | holdout | 10/0/0 | 0.003 | 0.0034 | [0.000108, 0.0061] | 0.002 | 10 | 1 |
| 12 | swiss_roll | QE | holdout | 6/4/0 | 0.000823 | 0.0011 | [-0.0056, 0.007] | 0.625 | 10 | 0.2 |
| 2 | wine | QE | holdout | 4/1/5 | 6.17e-06 | 2.54e-06 | [-1.18e-05, 1.31e-05] | 0.4375 | 10 | 0.6 |
| - | GLOBAL | QE | holdout | 63/38/39 | 0.000105 | -0.0026 | [-0.000154, 0.0021] | 0.1633 | 140 | 0.2475 |
| 8 | blobs | QE_B | both | 8/2/0 | 0.0025 | 0.0033 | [0.000283, 0.0064] | 0.0488 | 10 | 0.6 |
| 5 | breast_cancer | QE_B | both | 2/2/6 | -1.11e-08 | -4.43e-09 | [-4.92e-06, 4.92e-06] | 1 | 10 | 0 |
| 7 | california_housing | QE_B | both | 9/1/0 | 7.42e-06 | 0.000257 | [-0.0069, 0.0082] | 0.0645 | 10 | 0.8 |
| 9 | circles | QE_B | both | 6/4/0 | -0.0012 | -0.0065 | [-0.0372, 0.0231] | 0.9219 | 10 | 0.2 |
| 14 | covertype | QE_B | both | 3/7/0 | -0.000329 | -0.0022 | [-0.0153, 0.0059] | 0.3223 | 10 | -0.4 |
| 4 | diabetes | QE_B | both | 1/3/6 | -3.92e-06 | -7.82e-07 | [-3.93e-06, 3.95e-06] | 0.875 | 10 | -0.5 |
| 6 | digits | QE_B | both | 1/2/7 | -3.98e-06 | -1.06e-06 | [-1.06e-05, 5.35e-06] | 0.75 | 10 | -0.3333 |
| 1 | iris | QE_B | both | 4/2/4 | 4.93e-06 | 2.86e-06 | [-7.76e-06, 1.4e-05] | 0.2188 | 10 | 0.3333 |
| 13 | kddcup99 | QE_B | both | 4/6/0 | -0.0376 | -0.2194 | [-0.5996, 0.0099] | 0.1602 | 10 | -0.2 |
| 10 | moons | QE_B | both | 7/3/0 | 0.0018 | 0.0081 | [-0.003, 0.0377] | 0.2754 | 10 | 0.4 |
| 3 | olivetti_faces | QE_B | both | 4/3/3 | -0.0802 | -0.0794 | [-0.789, 0.4756] | 0.9375 | 10 | 0.1429 |
| 11 | s_curve | QE_B | both | 10/0/0 | 0.0037 | 0.0038 | [8.5e-05, 0.0073] | 0.002 | 10 | 1 |
| 12 | swiss_roll | QE_B | both | 6/4/0 | 0.000814 | 0.0015 | [-0.0051, 0.0079] | 0.6953 | 10 | 0.2 |
| 2 | wine | QE_B | both | 4/2/4 | 2.01e-06 | 1.15e-06 | [-7.44e-06, 7.84e-06] | 0.6875 | 10 | 0.3333 |
| - | GLOBAL | QE_B | both | 69/41/30 | 1.58e-05 | -0.0208 | [-3.92e-06, 0.0011] | 0.0923 | 140 | 0.2545 |
| 8 | blobs | train time | train | 0/10/0 | -82.1972 | -82.2698 | [-82.616, -81.8815] | 0.002 | 10 | -1 |
| 5 | breast_cancer | train time | train | 0/10/0 | -124.4887 | -123.0875 | [-125.5044, -117.1834] | 0.002 | 10 | -1 |
| 7 | california_housing | train time | train | 0/10/0 | -112.7705 | -107.7269 | [-120.1093, -94.1388] | 0.002 | 10 | -1 |
| 9 | circles | train time | train | 0/10/0 | -82.011 | -82.0468 | [-82.6543, -81.5143] | 0.002 | 10 | -1 |
| 14 | covertype | train time | train | 8/2/0 | 4.5544 | 3.1364 | [-0.2416, 4.7034] | 0.1055 | 10 | 0.6 |
| 4 | diabetes | train time | train | 0/10/0 | -124.7165 | -130.8947 | [-156.4431, -123.3642] | 0.002 | 10 | -1 |
| 6 | digits | train time | train | 0/10/0 | -124.2555 | -124.4075 | [-126.3659, -122.7873] | 0.002 | 10 | -1 |
| 1 | iris | train time | train | 0/10/0 | -126.2679 | -127.1989 | [-132.6232, -124.6813] | 0.002 | 10 | -1 |
| 13 | kddcup99 | train time | train | 10/0/0 | 12.295 | 12.5548 | [8.0787, 15.7426] | 0.002 | 10 | 1 |
| 10 | moons | train time | train | 0/10/0 | -82.0957 | -88.2408 | [-112.9703, -81.6979] | 0.002 | 10 | -1 |
| 3 | olivetti_faces | train time | train | 0/10/0 | -200.8962 | -199.5774 | [-203.2204, -194.7869] | 0.002 | 10 | -1 |
| 11 | s_curve | train time | train | 0/10/0 | -83.1004 | -83.8438 | [-87.0546, -82.4761] | 0.002 | 10 | -1 |
| 12 | swiss_roll | train time | train | 1/9/0 | -81.642 | -65.9893 | [-82.6048, -3.0855] | 0.0039 | 10 | -0.8 |
| 2 | wine | train time | train | 0/10/0 | -126.5213 | -126.5147 | [-127.7591, -125.2633] | 0.002 | 10 | -1 |
| - | GLOBAL | train time | train | 19/121/0 | -102.7688 | -94.7219 | [-103.9162, -95.6444] | 1.02e-22 | 140 | -0.7286 |


<!-- AUTO-TOPOLOGY-PVALUE-SUPP-TABLE:START -->
**Supplementary Table S7. Paired topology comparison p-values for hexagonal versus MST and hexagonal versus RNG across balanced QE, holdout QE, and train QE.** Rows list metric/dataset entries, including the OVERALL row. The MST and RNG columns report raw p-values and Benjamini-Hochberg q-values using the manuscript reporting convention. The embedded table is reproduced from `assets/tables/supp_table_topology_hex_vs_mst_rng_pvalues.tsv`.

| metric | dataset | MST_p | MST_q | RNG_p | RNG_q |
| --- | --- | --- | --- | --- | --- |
| Balanced QE | blobs | p=0.0163 | q=0.0456 | p=0.00134 | q=0.00313 |
| Balanced QE | circles | p=0.868 | q=0.868 | p=0.027 | q=0.0473 |
| Balanced QE | moons | p=0.494 | q=0.629 | p=0.0132 | q=0.0264 |
| Balanced QE | s_curve | p=0.000559 | q=0.0024 | p=8.08e-05 | q=0.000377 |
| Balanced QE | swiss_roll | p=0.247 | q=0.384 | p=0.0661 | q=0.0925 |
| Balanced QE | breast_cancer | p=0.208 | q=0.364 | p=0.364 | q=0.392 |
| Balanced QE | wine | p=0.000686 | q=0.0024 | p=0.000229 | q=0.000802 |
| Balanced QE | iris | p=0.58 | q=0.677 | p=0.000924 | q=0.00259 |
| Balanced QE | digits | p=0.165 | q=0.33 | p=0.266 | q=0.312 |
| Balanced QE | olivetti_faces | p=0.0663 | q=0.155 | p=0.0549 | q=0.0854 |
| Balanced QE | diabetes | p=0.633 | q=0.682 | p=0.267 | q=0.312 |
| Balanced QE | california_housing | p=0.000356 | q=0.0024 | p=5.65e-05 | q=0.000377 |
| Balanced QE | covertype | p=0.477 | q=0.629 | p=0.748 | q=0.748 |
| Balanced QE | kddcup99 | p=3.14e-08 | q=4.4e-07 | p=2.81e-05 | q=0.000377 |
| Balanced QE | OVERALL | p=1.12e-05 | q=NA | p=7.4e-10 | q=NA |
| Holdout QE | blobs | p=0.401 | q=0.468 | p=0.0358 | q=0.0626 |
| Holdout QE | circles | p=0.105 | q=0.147 | p=0.828 | q=0.828 |
| Holdout QE | moons | p=0.216 | q=0.275 | p=0.00122 | q=0.00427 |
| Holdout QE | s_curve | p=0.00226 | q=0.00791 | p=0.000724 | q=0.00338 |
| Holdout QE | swiss_roll | p=0.0913 | q=0.142 | p=0.0221 | q=0.0442 |
| Holdout QE | breast_cancer | p=0.0068 | q=0.0159 | p=0.0925 | q=0.13 |
| Holdout QE | wine | p=0.000275 | q=0.00128 | p=0.00244 | q=0.00683 |
| Holdout QE | iris | p=0.0125 | q=0.025 | p=0.0444 | q=0.0691 |
| Holdout QE | digits | p=0.0609 | q=0.107 | p=0.232 | q=0.295 |
| Holdout QE | olivetti_faces | p=3.93e-05 | q=0.000275 | p=9.96e-06 | q=0.000111 |
| Holdout QE | diabetes | p=0.984 | q=0.984 | p=0.351 | q=0.409 |
| Holdout QE | california_housing | p=0.00344 | q=0.00963 | p=0.0154 | q=0.0359 |
| Holdout QE | covertype | p=0.472 | q=0.508 | p=0.804 | q=0.828 |
| Holdout QE | kddcup99 | p=4.8e-08 | q=6.72e-07 | p=1.58e-05 | q=0.000111 |
| Holdout QE | OVERALL | p=0.15 | q=NA | p=0.0232 | q=NA |
| Train QE | blobs | p=0.0162 | q=0.0454 | p=0.00083 | q=0.00244 |
| Train QE | circles | p=0.119 | q=0.278 | p=0.000309 | q=0.00144 |
| Train QE | moons | p=0.445 | q=0.558 | p=0.0117 | q=0.0205 |
| Train QE | s_curve | p=0.000219 | q=0.00102 | p=0.000132 | q=0.000924 |
| Train QE | swiss_roll | p=0.362 | q=0.507 | p=0.0774 | q=0.108 |
| Train QE | breast_cancer | p=0.17 | q=0.34 | p=0.653 | q=0.691 |
| Train QE | wine | p=0.000156 | q=0.00102 | p=0.00114 | q=0.00266 |
| Train QE | iris | p=0.565 | q=0.608 | p=0.000872 | q=0.00244 |
| Train QE | digits | p=0.234 | q=0.364 | p=0.0623 | q=0.0969 |
| Train QE | olivetti_faces | p=0.211 | q=0.364 | p=0.191 | q=0.243 |
| Train QE | diabetes | p=0.75 | q=0.75 | p=0.24 | q=0.28 |
| Train QE | california_housing | p=0.00137 | q=0.00479 | p=0.00144 | q=0.00288 |
| Train QE | covertype | p=0.478 | q=0.558 | p=0.691 | q=0.691 |
| Train QE | kddcup99 | p=1.09e-08 | q=1.53e-07 | p=7.18e-05 | q=0.000924 |
| Train QE | OVERALL | p=0.0064 | q=NA | p=4.69e-06 | q=NA |
<!-- AUTO-TOPOLOGY-PVALUE-SUPP-TABLE:END -->


**Supplementary Table S8. Figure 13 deployment comparison percent summary for tuned FloatSOM RNG versus default hexagonal XPySOM across $QE_B$, $QE_H$, and $QE_T$.** Rows list per-dataset and `GLOBAL_OVERALL` entries with the plotted median percent change and 95% confidence interval. The embedded table is reproduced from `assets/tables/supp_table_figure_13_xpysom_rng_deployment_summary.tsv`.


| metric | dataset | median % change | 95% CI |
| --- | --- | --- | --- |
| QE_B | blobs | 5.4588 | [3.8282, 7.0895] |
| QE_B | circles | 5.6542 | [4.552, 6.7563] |
| QE_B | moons | 4.4217 | [2.6294, 6.2141] |
| QE_B | s_curve | 10.0599 | [9.2501, 10.8696] |
| QE_B | swiss_roll | 19.7271 | [18.4881, 20.9661] |
| QE_B | breast_cancer | 7.2273 | [6.3913, 8.0633] |
| QE_B | wine | 20.5163 | [19.3744, 21.6583] |
| QE_B | iris | 23.2197 | [20.569, 25.8704] |
| QE_B | digits | 8.3021 | [7.8283, 8.7758] |
| QE_B | olivetti_faces | 10.9797 | [10.0517, 11.9077] |
| QE_B | diabetes | 9.2085 | [8.1874, 10.2296] |
| QE_B | california_housing | 9.4344 | [9.0497, 9.8191] |
| QE_B | covertype | 30.3468 | [29.1115, 31.5821] |
| QE_B | kddcup99 | 38.2338 | [35.8971, 40.5705] |
| QE_B | GLOBAL_OVERALL | 14.485 | [12.7836, 16.1865] |
| QE_H | blobs | 4.8943 | [3.3372, 6.4515] |
| QE_H | circles | 4.98 | [3.8583, 6.1016] |
| QE_H | moons | 3.8767 | [1.9564, 5.797] |
| QE_H | s_curve | 9.4896 | [8.5416, 10.4377] |
| QE_H | swiss_roll | 19.2924 | [17.9583, 20.6264] |
| QE_H | breast_cancer | 0.1143 | [-0.6627, 0.8913] |
| QE_H | wine | -1.8427 | [-3.2635, -0.4218] |
| QE_H | iris | 0.4853 | [-2.8955, 3.8661] |
| QE_H | digits | 4.9934 | [4.5072, 5.4796] |
| QE_H | olivetti_faces | 2.6384 | [1.2928, 3.9839] |
| QE_H | diabetes | 1.3763 | [-0.3528, 3.1055] |
| QE_H | california_housing | 8.5235 | [8.0055, 9.0414] |
| QE_H | covertype | 30.3196 | [29.0984, 31.5408] |
| QE_H | kddcup99 | 37.9291 | [35.5645, 40.2937] |
| QE_H | GLOBAL_OVERALL | 9.0765 | [7.1203, 11.0326] |
| QE_T | blobs | 6.0226 | [4.2924, 7.7527] |
| QE_T | circles | 6.3297 | [5.1892, 7.4702] |
| QE_T | moons | 4.968 | [3.2744, 6.6616] |
| QE_T | s_curve | 10.6309 | [9.9339, 11.3279] |
| QE_T | swiss_roll | 20.1631 | [18.9915, 21.3347] |
| QE_T | breast_cancer | 15.8475 | [14.5024, 17.1926] |
| QE_T | wine | 56.1875 | [54.54, 57.8351] |
| QE_T | iris | 63.2439 | [60.2553, 66.2326] |
| QE_T | digits | 11.8094 | [11.1554, 12.4634] |
| QE_T | olivetti_faces | 21.029 | [19.7281, 22.3298] |
| QE_T | diabetes | 18.9602 | [18.1282, 19.7921] |
| QE_T | california_housing | 10.3513 | [9.7384, 10.9642] |
| QE_T | covertype | 30.3738 | [29.1225, 31.6251] |
| QE_T | kddcup99 | 38.5352 | [36.2102, 40.8602] |
| QE_T | GLOBAL_OVERALL | 22.4609 | [19.4613, 25.4605] |


**Supplementary Table S9. Supplementary Figure S12 deployment comparison percent summary for tuned FloatSOM hexagonal versus default hexagonal XPySOM across $QE_B$, $QE_H$, and $QE_T$.** Rows list per-dataset and `GLOBAL_OVERALL` entries with the plotted median percent change and 95% confidence interval. The embedded table is reproduced from `assets/tables/supp_table_s13_xpysom_hexagonal_deployment_summary.tsv`.


| metric | dataset | median % change | 95% CI |
| --- | --- | --- | --- |
| QE_B | blobs | 4.5418 | [3.0375, 6.0462] |
| QE_B | circles | 4.5998 | [3.6022, 5.5974] |
| QE_B | moons | 3.5895 | [2.051, 5.128] |
| QE_B | s_curve | 9.7727 | [9.0814, 10.464] |
| QE_B | swiss_roll | 17.6702 | [16.7526, 18.5878] |
| QE_B | breast_cancer | 6.399 | [5.5895, 7.2084] |
| QE_B | wine | 15.0518 | [13.113, 16.9905] |
| QE_B | iris | 19.6158 | [17.7567, 21.475] |
| QE_B | digits | 7.7568 | [7.3648, 8.1489] |
| QE_B | olivetti_faces | 10.367 | [9.5746, 11.1594] |
| QE_B | diabetes | 9.133 | [8.2954, 9.9707] |
| QE_B | california_housing | 8.9193 | [8.4319, 9.4067] |
| QE_B | covertype | 29.6621 | [28.3242, 31] |
| QE_B | kddcup99 | 35.5053 | [34.0328, 36.9779] |
| QE_B | GLOBAL_OVERALL | 13.0417 | [11.4631, 14.6204] |
| QE_H | blobs | 3.8869 | [2.4806, 5.2931] |
| QE_H | circles | 4.1388 | [3.1274, 5.1503] |
| QE_H | moons | 2.9919 | [1.3076, 4.6761] |
| QE_H | s_curve | 9.2419 | [8.4744, 10.0095] |
| QE_H | swiss_roll | 17.3207 | [16.3909, 18.2504] |
| QE_H | breast_cancer | -0.997 | [-1.683, -0.311] |
| QE_H | wine | -2.8868 | [-4.4124, -1.3613] |
| QE_H | iris | -0.4904 | [-2.1865, 1.2058] |
| QE_H | digits | 4.5103 | [4.0172, 5.0034] |
| QE_H | olivetti_faces | 1.9861 | [0.769, 3.2032] |
| QE_H | diabetes | 2.245 | [1.1198, 3.3702] |
| QE_H | california_housing | 8.0301 | [7.4144, 8.6458] |
| QE_H | covertype | 29.6665 | [28.3026, 31.0305] |
| QE_H | kddcup99 | 35.1253 | [33.6747, 36.5759] |
| QE_H | GLOBAL_OVERALL | 8.1978 | [6.3289, 10.0667] |
| QE_T | blobs | 5.1961 | [3.5689, 6.8233] |
| QE_T | circles | 5.0606 | [4.0085, 6.1127] |
| QE_T | moons | 4.1898 | [2.7833, 5.5963] |
| QE_T | s_curve | 10.3044 | [9.6672, 10.9416] |
| QE_T | swiss_roll | 18.021 | [17.101, 18.9411] |
| QE_T | breast_cancer | 15.3534 | [14.1184, 16.5884] |
| QE_T | wine | 43.6818 | [39.113, 48.2507] |
| QE_T | iris | 55.1557 | [50.9804, 59.331] |
| QE_T | digits | 11.2012 | [10.7183, 11.6841] |
| QE_T | olivetti_faces | 20.4609 | [19.2364, 21.6853] |
| QE_T | diabetes | 17.7191 | [16.8369, 18.6013] |
| QE_T | california_housing | 9.8167 | [9.2321, 10.4012] |
| QE_T | covertype | 29.6576 | [28.3443, 30.9708] |
| QE_T | kddcup99 | 35.8819 | [34.3431, 37.4206] |
| QE_T | GLOBAL_OVERALL | 20.1214 | [17.5722, 22.6707] |


**Supplementary Table S10. Supplementary Figure S13 deployment comparison percent summary for tuned FloatSOM MST versus default hexagonal XPySOM across $QE_B$, $QE_H$, and $QE_T$.** Rows list per-dataset and `GLOBAL_OVERALL` entries with the plotted median percent change and 95% confidence interval. The embedded table is reproduced from `assets/tables/supp_table_s14_xpysom_mst_deployment_summary.tsv`.


| metric | dataset | median % change | 95% CI |
| --- | --- | --- | --- |
| QE_B | blobs | 5.2243 | [3.6173, 6.8314] |
| QE_B | circles | 5.5236 | [4.4413, 6.6059] |
| QE_B | moons | 4.2266 | [2.4056, 6.0477] |
| QE_B | s_curve | 9.968 | [9.2244, 10.7116] |
| QE_B | swiss_roll | 19.5015 | [18.1673, 20.8356] |
| QE_B | breast_cancer | 7.2275 | [6.4917, 7.9633] |
| QE_B | wine | 18.0952 | [16.4439, 19.7464] |
| QE_B | iris | 17.3842 | [13.8051, 20.9634] |
| QE_B | digits | 8.5168 | [7.9287, 9.105] |
| QE_B | olivetti_faces | 11.9472 | [10.795, 13.0994] |
| QE_B | diabetes | 9.2961 | [8.2097, 10.3825] |
| QE_B | california_housing | 9.474 | [9.0518, 9.8963] |
| QE_B | covertype | 32.2987 | [30.9976, 33.5997] |
| QE_B | kddcup99 | 40.8705 | [39.5821, 42.159] |
| QE_B | GLOBAL_OVERALL | 14.2539 | [12.4871, 16.0207] |
| QE_H | blobs | 4.5705 | [3.0306, 6.1104] |
| QE_H | circles | 4.8064 | [3.6639, 5.9489] |
| QE_H | moons | 3.768 | [1.7919, 5.744] |
| QE_H | s_curve | 9.4507 | [8.6251, 10.2763] |
| QE_H | swiss_roll | 19.0489 | [17.593, 20.5048] |
| QE_H | breast_cancer | -0.7772 | [-1.3919, -0.1624] |
| QE_H | wine | -2.2046 | [-3.9272, -0.4821] |
| QE_H | iris | -0.872 | [-3.7238, 1.9798] |
| QE_H | digits | 5.2512 | [4.6436, 5.8587] |
| QE_H | olivetti_faces | 2.5542 | [0.9211, 4.1873] |
| QE_H | diabetes | 1.059 | [-0.5135, 2.6316] |
| QE_H | california_housing | 8.4673 | [7.9248, 9.0099] |
| QE_H | covertype | 32.2708 | [30.9787, 33.5629] |
| QE_H | kddcup99 | 40.3684 | [39.0453, 41.6915] |
| QE_H | GLOBAL_OVERALL | 9.1258 | [7.0316, 11.2201] |
| QE_T | blobs | 5.8778 | [4.178, 7.5776] |
| QE_T | circles | 6.2415 | [5.1655, 7.3175] |
| QE_T | moons | 4.6869 | [3.0023, 6.3715] |
| QE_T | s_curve | 10.486 | [9.7854, 11.1866] |
| QE_T | swiss_roll | 19.9554 | [18.7303, 21.1804] |
| QE_T | breast_cancer | 16.9227 | [15.7815, 18.0639] |
| QE_T | wine | 50.5129 | [47.2136, 53.8122] |
| QE_T | iris | 49.5872 | [43.4267, 55.7476] |
| QE_T | digits | 11.9794 | [11.277, 12.6817] |
| QE_T | olivetti_faces | 23.2531 | [21.9339, 24.5724] |
| QE_T | diabetes | 19.5575 | [18.2819, 20.8331] |
| QE_T | california_housing | 10.4893 | [9.9385, 11.04] |
| QE_T | covertype | 32.3264 | [31.0143, 33.6385] |
| QE_T | kddcup99 | 41.3715 | [40.0794, 42.6636] |
| QE_T | GLOBAL_OVERALL | 21.6605 | [19.0539, 24.2672] |


**Supplementary Table S11. Figure 12 topology runtime summary at the largest common 8-GPU axis value for the dimension, sample, and grid size scaling workloads.** Rows report the plotted 8-GPU mean runtimes for hexagonal, MST, and RNG, together with the fastest and slowest topology at that axis value and the maximum pairwise runtime spread. The embedded table is reproduced from `assets/tables/supp_table_figure_12_topology_runtime_summary.tsv`.


| axis | axis value | hexagonal s | MST s | RNG s | fastest | slowest | spread % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dimension Scaling | 5000 | 404.17 | 423.15 | 409.94 | hexagonal | mst | 4.7 |
| Sample Scaling | 1e+09 | 363.58 | 375.44 | 369.41 | hexagonal | mst | 3.26 |
| Grid-Size Scaling | 64 | 32.54 | 266.45 | 880.83 | hexagonal | rng | 2606.61 |

## 12. Supplementary Figures (End Matter)

![Supplementary Figure S1](assets_manual/figures/supp_fig_s1.svg)
*Supplementary Figure S1. FloatSOM versus XPySOM calibration under default settings for the MST topology path. Panels A-C report paired $QE$ effects for $QE_B$, $QE_H$, and $QE_T$. Panel D reports dataset-level median runtime deltas against dataset size, where each numbered dot is the median matched-seed value of `FloatSOM time - XPySOM time`; negative values favor FloatSOM and positive values favor XPySOM. The point numbers map to Supplementary Table S4. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

![Supplementary Figure S2](assets_manual/figures/supp_fig_s2.svg)
*Supplementary Figure S2. FloatSOM versus XPySOM calibration under default settings for the RNG topology path. Panels A-C report paired $QE$ effects for $QE_B$, $QE_H$, and $QE_T$. Panel D reports dataset-level median runtime deltas against dataset size, where each numbered dot is the median matched-seed value of `FloatSOM time - XPySOM time`; negative values favor FloatSOM and positive values favor XPySOM. The point numbers map to Supplementary Table S5. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

![Supplementary Figure S3](assets_manual/figures/supp_fig_s3.svg)
*Supplementary Figure S3. FloatSOM versus XPySOM calibration under default settings for the hexagonal topology path. Panels A-C report paired $QE$ effects for $QE_B$, $QE_H$, and $QE_T$. Panel D reports dataset-level median runtime deltas against dataset size, where each numbered dot is the median matched-seed value of `FloatSOM time - XPySOM time`; negative values favor FloatSOM and positive values favor XPySOM. The point numbers map to Supplementary Table S6. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

![Supplementary Figure S4](assets_manual/figures/supp_fig_s4.svg)
*Supplementary Figure S4. MST versus RNG topology on $QE$ metrics under full sampling only. Panels A-C report paired full sampling only $QE$ effects for $QE_B$, $QE_H$, and $QE_T$ across the available full sampling datasets. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

![Supplementary Figure S5](assets_manual/figures/supp_fig_s5.svg)
*Supplementary Figure S5. Hexagonal versus MST topology sensitivity under full sampling only. Panels A-C report the matched top-$k$ paired sensitivity analysis for $QE_B$, $QE_H$, and $QE_T$.*

![Supplementary Figure S6](assets_manual/figures/supp_fig_s6.svg)
*Supplementary Figure S6. Hexagonal versus RNG topology sensitivity under full sampling only. Panels A-C report the matched top-$k$ paired sensitivity analysis for $QE_B$, $QE_H$, and $QE_T$.*

![Supplementary Figure S7](assets_manual/figures/supp_fig_s7.svg)
*Supplementary Figure S7. MST versus RNG topology sensitivity under full sampling only. Panels A-C report the matched top-$k$ paired sensitivity analysis for $QE_B$, $QE_H$, and $QE_T$.*

![Supplementary Figure S8](assets_manual/figures/supp_fig_s8.svg)
*Supplementary Figure S8. Tuned configuration versus untuned reference $QE$ comparison for the hexagonal topology only, across $QE_B$, $QE_H$, and $QE_T$ under the matched pairing keys. Positive values indicate the tuned configuration outperforms the untuned reference. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

![Supplementary Figure S9](assets_manual/figures/supp_fig_s9.svg)
*Supplementary Figure S9. Tuned configuration versus untuned reference $QE$ comparison for the MST topology only, across $QE_B$, $QE_H$, and $QE_T$ under the matched pairing keys. The tuned configuration is derived from the Optuna-selected settings by taking the mean of numeric parameters and the mode of categorical parameters across seeds. Positive values indicate the tuned configuration outperforms the untuned reference. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

![Supplementary Figure S10](assets_manual/figures/supp_fig_s10.svg)
*Supplementary Figure S10. Tuned configuration versus untuned reference $QE$ comparison for the RNG topology only, across $QE_B$, $QE_H$, and $QE_T$ under the matched pairing keys. The tuned configuration is derived from the Optuna-selected settings by taking the mean of numeric parameters and the mode of categorical parameters across seeds. Positive values indicate the tuned configuration outperforms the untuned reference. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect.*

![Supplementary Figure S11](assets_manual/figures/supp_fig_s11.svg)
*Supplementary Figure S11. Full GPU-count scaling context for MST and hexagonal under matched full sampling settings. Panels A-C show MST runtime across dimension, sample, and grid size scaling workloads; panels D-F show the corresponding hexagonal runs. Within each panel, curves correspond to $G\in\{1,2,4,8\}$ GPUs and report mean wall-clock runtime (s) with $\pm 1$ standard-deviation error bars across $n=3$ repeated runs per configuration.*

![Supplementary Figure S12](assets_manual/figures/supp_fig_s12.svg)
*Supplementary Figure S12. Deployment comparison of default hexagonal XPySOM versus tuned FloatSOM hexagonal. Panels A-C report paired $QE$ effects for $QE_B$, $QE_H$, and $QE_T$ under the matched dataset/seed comparison keys. Positive values indicate tuned FloatSOM hexagonal outperforms default hexagonal XPySOM. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect. Per-dataset and `GLOBAL_OVERALL` panel summaries are listed in Supplementary Table S9.*

![Supplementary Figure S13](assets_manual/figures/supp_fig_s13.svg)
*Supplementary Figure S13. Deployment comparison of default hexagonal XPySOM versus tuned FloatSOM MST. Panels A-C report paired $QE$ effects for $QE_B$, $QE_H$, and $QE_T$ under the matched dataset/seed comparison keys. Positive values indicate tuned FloatSOM MST outperforms default hexagonal XPySOM. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect. Per-dataset and `GLOBAL_OVERALL` panel summaries are listed in Supplementary Table S10.*

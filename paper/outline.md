# Floatsom Paper Outline

## Core message

We evaluate five deployment-facing questions for large-scale SOMs: sampling efficiency (`random` vs `full`), topology quality (`mst`/`rng` vs `hexagonal`), tuned-versus-untuned performance, hyperparameter stability near optimum, and scalable execution under memory pressure. The current manuscript centers on QE-first topology and systems claims.

## Section goals and flow

1. Introduction: problem framing and deployment motivation.
2. Related Work: topology flexibility versus scalable implementations.
3. Methods: topology algorithms and systems execution.
4. Experimental Setup: datasets, metrics, Optuna protocol, and statistical design.
5. Results: sampling, topology, tuning, stability, and systems outcomes.
6. Discussion: practical interpretation, including synthetic versus non-synthetic behavior.
7. Conclusion: final deployment takeaways.

## Methods

### 3.1 MST Topology Implementation

- Dynamic MST graph updates from prototype geometry.
- Graph-distance influence and cache strategy.
- Expected QE behavior versus hexagonal baseline.

### 3.2 RNG Topology Implementation

- Relative Neighborhood Graph construction via the direct blocker criterion.
- Shared downstream distance/influence pipeline with MST.
- Expected flexibility-quality tradeoff versus MST/hexagonal.

### 3.3 Multi-GPU + OOM Methodology and Implementation

- Ray actor orchestration and NCCL synchronization.
- Disk-backed chunked loading and bounded host/device staging.
- Memory-pressure controls for BMU/update/Floyd-Warshall/RNG checks.

## Experimental setup

### 4.1 Datasets and preprocessing

- Fourteen benchmark datasets spanning synthetic and real-world structure.
- Explicit synthetic versus non-synthetic grouping for downstream interpretation.

### 4.2 Metrics

- Primary endpoint: QE (`holdout`, `train`, and Balanced QE).
- Distortion excluded from topology conclusions where undefined for MST.

### 4.3 Optuna protocol (topology and sampling campaign)

- Campaign-level Optuna evaluation over `topology ∈ {hexagonal, mst, rng}`.
- Sampling scenarios: `full`, `random`, `hdsssom`.
- Batch mode is scenario-dependent (`full_batch` and `minibatch` where applicable), not globally fixed.
- Fixed iteration budget for fair QE comparisons.

### 4.4 Speed benchmark protocol

- Harmonized 1/2/3/4/8-GPU runtime comparisons.
- Publication-level topology runtime ratios without command/cache implementation detail.

### 4.5 Statistical analysis

- Paired comparisons with Wilcoxon + BH for main topology/sampling panels.
- Explicit wording convention for “no detected difference.”

### 4.6 Default-aware tuned-versus-untuned analysis

- Pairing keys: `(dataset, processing, sampling, batch_mode, topology, seed[, split])`.
- Untuned comparator: default-enqueued trial (lowest trial number).
- Tuned comparator: best metric trial within key.
- Outputs from `diagnostics_publication_path/raw/` in this revision.

### 4.7 Hyperparameter stability analysis

- Topology-pair stability comparisons: `hexagonal:mst`, `hexagonal:rng`.
- Numeric stability: relative seed-pair parameter difference.
- Categorical stability: mismatch rate.
- Lower score indicates more stable near-optimal hyperparameter behavior.

## Results

### 5.1 XPySOM calibration (equivalence)

- Matched hexagonal calibration confirms FloatSOM batch comparability context.

### 5.2 Sampling Results

- `random` faster than `full` with no detected QE loss in iteration-matched setup.

### 5.3 MST Results

- QE-focused MST-versus-hexagonal comparisons (pooled `full+random`).

### 5.4 RNG Results

- QE-focused RNG-versus-hexagonal and RNG-versus-MST comparisons.

### 5.5 Multi-GPU + OOM Results

- Topology runtime-ratio behavior and sampling-speed ratios.
- OOM-safe execution diagnostics.

### 5.6 Parameter tuning versus untuned defaults

- Tuned runs substantially outperform matched untuned defaults.
- Report balanced, holdout, and train QE paired-improvement summaries.

### 5.7 Hyperparameter stability and synthetic/non-synthetic behavior

- MST/RNG require smaller effective hyperparameter displacement than hexagonal near optima.
- Topology gains are stronger in non-synthetic, higher-dimensional settings.

## Discussion

- Sampling and topology tradeoffs under fixed compute budgets.
- Why tuning must be treated as first-order, not optional.
- Why graph topologies are less constrained than hexagonal in higher-dimensional non-synthetic data.
- Practical systems constraints and scaling limits.

## Figure plan

- Keep currently committed figure set unchanged for this revision.
- Do not add new figures/tables in this pass.
- Integrate new tuning/stability/synthetic-vs-real content as text-only additions in Methods, Results, and Discussion.

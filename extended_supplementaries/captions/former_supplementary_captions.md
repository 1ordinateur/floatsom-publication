# Former supplementary captions

These labels record the numbering used before the revised manuscript supplement was shortened.

## Tables

**Former Supplementary Table S4. Untuned FloatSOM MST versus hexagonal XPySOM.** The `dataset_index` column matches the numbered points in former Supplementary Figure S1 panel D. Positive percentage values and positive signed effects favor FloatSOM; the compact table lists wins as FloatSOM/XPySOM/ties.

**Former Supplementary Table S5. Untuned FloatSOM RNG versus hexagonal XPySOM.** The `dataset_index` column matches the numbered points in former Supplementary Figure S2 panel D. Positive percentage values and positive signed effects favor FloatSOM; the compact table lists wins as FloatSOM/XPySOM/ties.

**Former Supplementary Table S9. Former Supplementary Figure S7 deployment-comparison percentage summary for tuned FloatSOM hexagonal versus untuned hexagonal XPySOM across $QE_B$, $QE_H$, and $QE_T$.** Rows list per-dataset and `GLOBAL_OVERALL` entries with the plotted median percentage change and 95% confidence interval.

**Former Supplementary Table S10. Former Supplementary Figure S8 deployment-comparison percentage summary for tuned FloatSOM MST versus untuned hexagonal XPySOM across $QE_B$, $QE_H$, and $QE_T$.** Rows list per-dataset and `GLOBAL_OVERALL` entries with the plotted median percentage change and 95% confidence interval.

**Former Supplementary Table S11. Figure 14 topology runtime summary at the largest common 8-GPU axis value for the dimension, sample, and grid-size scaling workloads.** Rows report the plotted 8-GPU mean runtimes for hexagonal, MST, and RNG, together with the fastest and slowest topology at that axis value and the maximum pairwise runtime spread.

**Former Supplementary Table S13. Full matched topology diagnostic means by profile, dataset, and topology.** Rows report the mean value across the final matched random seeds for each profile, dataset, topology, and full-sampling setting. `untuned` denotes the untuned reference profile using XPySOM-like default hyperparameters and `tuned` denotes the fixed QE-tuned profile. Lower is better for QE, MTR, and dead-node fraction; higher is better for node utilization. Hexagonal--graph MTR values are descriptive. Metric suffixes denote holdout (`_H`), train (`_T`), and balanced train--holdout (`_B`) summaries.

## Figures

**Former Supplementary Figure S1. Untuned FloatSOM MST versus hexagonal XPySOM.** Panels A--C report paired $QE$ effects for $QE_B$, $QE_H$, and $QE_T$. Panel D reports dataset-level median runtime differences against dataset size, where each numbered dot is the median matched-seed value of `FloatSOM time - XPySOM time`; negative values favor FloatSOM and positive values favor XPySOM. The point numbers map to former Supplementary Table S4. Forest whiskers denote exact distribution-free 95% intervals for the paired Wilcoxon location estimate.

**Former Supplementary Figure S2. Untuned FloatSOM RNG versus hexagonal XPySOM.** Panels A--C report paired $QE$ effects for $QE_B$, $QE_H$, and $QE_T$. Panel D reports dataset-level median runtime differences against dataset size, where each numbered dot is the median matched-seed value of `FloatSOM time - XPySOM time`; negative values favor FloatSOM and positive values favor XPySOM. The point numbers map to former Supplementary Table S5. Forest whiskers denote exact distribution-free 95% intervals for the paired Wilcoxon location estimate.

**Former Supplementary Figure S7. Deployment comparison of untuned hexagonal XPySOM versus tuned FloatSOM hexagonal.** Panels A--C report $QE_B$, $QE_H$, and $QE_T$ under the matched dataset/seed comparison keys. Positive values indicate that tuned FloatSOM hexagonal outperforms untuned hexagonal XPySOM. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect. Per-dataset and `GLOBAL_OVERALL` summaries are provided in former Supplementary Table S9.

**Former Supplementary Figure S8. Deployment comparison of untuned hexagonal XPySOM versus tuned FloatSOM MST.** Panels A--C report $QE_B$, $QE_H$, and $QE_T$ under the matched dataset/seed comparison keys. Positive values indicate that tuned FloatSOM MST outperforms untuned hexagonal XPySOM. Forest whiskers denote 95% paired $t$-test confidence intervals around the mean paired effect. Per-dataset and `GLOBAL_OVERALL` summaries are provided in former Supplementary Table S10.

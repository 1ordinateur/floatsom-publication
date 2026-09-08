# FloatSOM extended supplementaries

This archive contains detailed benchmark results removed from the revised FloatSOM manuscript to keep the paper focused on its primary distributed-systems and scaling contribution. The archived material remains available for readers who want the full dataset-level comparisons.

The table and figure numbers are the **former supplementary labels** used before the manuscript supplement was shortened. They are intentionally preserved here so that the archive remains traceable to the review history. They do not match the consecutively renumbered supplement in the revised manuscript.

## Authors

- Tony Xu
- Sarah Klamt
- Katharine Turner
- Anne Brüstle
- Felix Marsh-Wakefield
- Givanna Putri

## Contents

- Former Tables S4 and S5: untuned FloatSOM MST and RNG comparisons against hexagonal XPySOM.
- Former Tables S9 and S10: tuned FloatSOM hexagonal and MST deployment comparisons against untuned hexagonal XPySOM.
- Former Table S11: the exact Figure 14 topology-runtime values at the largest common 8-GPU axis settings.
- Former Table S13: full matched topology diagnostic means by profile, dataset, and topology.
- Former Figures S1 and S2: untuned FloatSOM MST and RNG comparisons against hexagonal XPySOM.
- Former Figures S7 and S8: tuned FloatSOM hexagonal and MST deployment comparisons against untuned hexagonal XPySOM.

`MANIFEST.tsv` maps each former manuscript item to its archived files. `captions/former_supplementary_captions.md` preserves the complete figure and table descriptions.

## Interpretation

Quantization error (`QE`) is lower when the learned prototypes lie closer to the observations. The suffixes `_H`, `_T`, and `_B` denote holdout, train, and balanced train--holdout summaries, respectively. For the XPySOM comparison tables, positive percentage values favor FloatSOM unless the table caption states otherwise. Node utilization is higher-is-better, whereas Mean Tied Rank and dead-node fraction are lower-is-better.

The revised manuscript retains the hexagonal FloatSOM--XPySOM equivalence result, the direct within-FloatSOM topology contrasts, Figure 14's topology-runtime curves, and the one-sentence decomposition of the full tuned-RNG comparison. These archived results provide the more detailed implementation, topology, runtime, and per-dataset views that are not needed to establish the revised manuscript's central claims.

## Files and formats

Tables are tab-separated UTF-8 text files. Figures are supplied as editable SVG and rendered PDF files. Checksums in `SHA256SUMS` can be used to verify the archived files.

## Citation

These extended supplementaries accompany the FloatSOM paper and are hosted in the
[publication repository](https://github.com/1ordinateur/floatsom-publication).
Download the self-contained [extended_supplementaries.zip](https://github.com/1ordinateur/floatsom-publication/blob/main/extended_supplementaries.zip).
Please cite the [FloatSOM paper on OpenReview](https://openreview.net/forum?id=n2NQNxu9Ei).

## License

The tables, figures, captions, and documentation in this archive are released under the Creative Commons Attribution 4.0 International license (CC BY 4.0).

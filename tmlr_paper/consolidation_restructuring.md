# Consolidation and restructuring

We thank the reviewers for their convergent feedback on the manuscript's central objective, the relative strength of its systems and topology contributions, and the length of the supplementary material. We have completed a focused consolidation and restructuring pass in response. This pass does not add further attribution experiments; instead, it aligns the manuscript's framing with the evidence already established by the revised analyses.

## Systems and scaling are now the primary contribution

We revised the Abstract, Introduction, contribution statement, and Discussion so that FloatSOM's distributed systems and scaling capabilities are the primary contribution. The manuscript now leads with multi-GPU execution, out-of-memory disk-backed streaming, automated hyperparameter tuning, and the demonstrated billion-sample workload across multiple HPC nodes.

Training-time graph topologies remain an important supported capability, but they are now presented as a secondary finding rather than as a co-equal or dominant scientific contribution. The Introduction explicitly identifies two contributions: a distributed, out-of-memory GPU SOM framework that supports multi-GPU execution and disk-backed streaming, followed by scalable training-time MST and RNG topology support.

The first paragraph of the Introduction was retained unchanged. The following paragraphs now state the scalability and neighborhood-topology limitations directly, distinguish training-time graph neighborhoods from post hoc graph overlays, and set out the paper's aims and contributions explicitly.

## Topology claims now match the attribution evidence

The revised framing reflects the existing decomposition of the final tuned-RNG comparison. On the common untuned-XPySOM reference scale, tuning contributes 13.04 percentage points of the full 14.49% balanced-quantization-error gain, while topology contributes the remaining 1.44 points. This decomposition already answers the attribution question and shows that topology provides a real but modest contribution rather than the dominant source of improvement.

The Abstract and Discussion now describe the graph-topology improvement over a tuned hexagonal lattice as modest and explicitly acknowledge the additional compute cost at large grid sizes. Among the graph topologies, RNG gave the most favorable quantization-error, stability, and node-utilization results, so we retain RNG as the default graph topology. The conventional hexagonal lattice remains an appropriate option, particularly where the modest graph-topology gain does not justify the additional compute.

The interpretation of Mean Tied Rank remains deliberately qualified. The fixed-adjacency permutation-null analysis produced effectively identical null values across topology families, so it does not provide a topology-specific normalization for graph--hexagonal comparisons. Those comparisons remain descriptive, while MTR is used inferentially only to compare MST and RNG, the two weight-derived graph families.

## Supplementary material was consolidated

We removed detailed or redundant tables and their directly associated figures from the manuscript supplement and placed them in an accompanying, non-anonymous Zenodo archive. This preserves the complete numerical record for readers who want to inspect the results in detail while keeping the manuscript focused on its main systems and scaling contribution.

The archived items are:

- Former Supplementary Tables S4 and S5: untuned FloatSOM MST and RNG comparisons against hexagonal XPySOM.
- Former Supplementary Tables S9 and S10: tuned FloatSOM hexagonal and MST comparisons against untuned hexagonal XPySOM.
- Former Supplementary Table S11: the exact topology-runtime values underlying the largest-axis summaries in Figure 14.
- Former Supplementary Table S13: full matched topology diagnostic means by profile, dataset, and topology.
- Former Supplementary Figures S1 and S2: the detailed untuned MST and RNG comparisons against hexagonal XPySOM.
- Former Supplementary Figures S7 and S8: the detailed tuned-hexagonal and tuned-MST deployment comparisons against untuned hexagonal XPySOM.

Figure 14 remains in the manuscript and provides the topology-runtime comparison visually; its former Supplementary Table S11 numerical summary is now available in the Zenodo archive. The archive includes machine-readable TSV tables, SVG and PDF figures, full captions, a manifest, named-author metadata, a CC BY 4.0 license, and checksums. The assigned Zenodo DOI will be inserted into the manuscript and this letter after deposition.

## Retained supplementary items were renumbered

After consolidation, the manuscript supplement contains Supplementary Tables S1--S13 and Supplementary Figures S1--S4. The retained items were renumbered as follows:

| Former label | Revised label |
| --- | --- |
| Tables S1--S3 | Tables S1--S3 |
| Table S6 | Table S4 |
| Table S7 | Table S5 |
| Table S8 | Table S6 |
| Table S12 | Table S7 |
| Table S14 | Table S8 |
| Tables S15--S19 | Tables S9--S13 |
| Figures S3--S6 | Figures S1--S4 |

All in-text references, methods references, table captions, and figure captions were updated to use the revised numbering. The retained supplementary figure files were also updated directly: both the SVG and PDF filenames and the visible figure titles now use the revised S1--S4 labels.

## Revised and tracked manuscript versions

The clean revised manuscript incorporates the final framing, supplement consolidation, and reference updates. A separate v2 tracked-changes manuscript shows these changes against the most recent pre-consolidation revision. The earlier tracked-changes manuscript has been preserved unchanged so that the two revision stages remain distinct and auditable.

Together, these changes make the paper's objective explicit, align the prominence of topology with its measured contribution, and substantially shorten the supplementary material without discarding the detailed results generated during review.

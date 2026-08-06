# Consolidation and restructuring

We thank the reviewers for their helpful comments on the manuscript's central objective, the relative importance of the systems and topology contributions, and the length of the supplementary material. We agree that FloatSOM's strongest contribution is its scalable systems framework, with graph-based topologies representing a supported secondary contribution.

## Manuscript framing

We revised the Abstract, Introduction, contribution statement, and Discussion accordingly. The manuscript now presents distributed multi-GPU execution, out-of-memory disk-backed streaming, automated hyperparameter tuning, and the billion-sample demonstration as the primary contribution. The Introduction states the paper's objective and contributions explicitly. Training-time MST and RNG topologies are presented as additional capabilities enabled by the framework rather than as the dominant source of improvement.

The topology claims have also been narrowed to match the evidence. On the common untuned-XPySOM reference scale, tuning accounts for 13.04 percentage points of the full tuned-RNG stack's 14.49% balanced-QE gain, while topology accounts for 1.44 points. We therefore do not attribute the overall improvement primarily to topology. Among the graph topologies, RNG gave the most favorable QE, stability, and node-utilization results and is retained as the default. For larger node counts, where RNG construction becomes substantially more expensive, we instead suggest MST.

We did not add further attribution experiments. The existing matched-configuration comparison, radius-sensitivity analysis, and gain decomposition already establish the size and limits of the topology effect. We also retain the qualification that graph--hexagonal MTR comparisons are descriptive because the permutation-null values were effectively identical across topology families; MTR is used inferentially only for MST--RNG comparisons.

## Supplementary consolidation

We moved former Supplementary Tables S4, S5, S9, S10, S11, and S13, together with former Supplementary Figures S1, S2, S7, and S8, to a non-anonymous Zenodo archive. This preserves the detailed results without requiring them in the journal supplement. The Zenodo DOI will be added after deposition.

The retained material is now numbered consecutively as Tables S1--S13 and Figures S1--S4. The table mapping is S1--S3 unchanged, S6 to S4, S7 to S5, S8 to S6, S12 to S7, S14 to S8, and S15--S19 to S9--S13. Former Figures S3--S6 are now Figures S1--S4. All manuscript references and SVG/PDF figure labels were updated to match.

## Revision files

The clean manuscript contains the revised framing and consolidated supplement. A separate v2 tracked-changes manuscript shows these changes against the immediately preceding revision, while the earlier tracked version remains unchanged.

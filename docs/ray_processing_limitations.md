# Local and Ray processing

The publication evaluates full-sampling execution-path concordance in
Supplementary Methods S1 and Supplementary Table S8. This result applies to
the evaluated configurations; it does not establish equivalence for every
sampling and processing mode.

Color processing, random sampling and full-batch processing have distinct
selection and synchronization semantics. Use their corresponding regression
tests when modifying chunking, worker assignment or buffering.

For current distributed setup instructions, see the standalone library's
[Ray execution guide](https://github.com/1ordinateur/floatsom/blob/main/docs/ray_execution.md).
The retained benchmark job scripts describe the experiment workflows; their
scheduler allocations and paths must be set for the target cluster.

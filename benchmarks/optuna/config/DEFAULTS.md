# Benchmark defaults

FloatSOM tuned profiles are maintained and packaged in the
[standalone library](https://github.com/1ordinateur/floatsom/blob/main/docs/defaults.md).
Benchmark constructors explicitly use `defaults_profile="publication"`, preserving
this checkout's settings even when installed beside the standalone library.

The fixed-profile CLI options accept a profile name or a custom JSON path:

```bash
python -m floatsom_benchmarks.optuna.run_matched_default_floatsom_batch \
  --fixed-params-by-sampling-topology-json publication --help
```

| Name | Use |
| --- | --- |
| `publication` | Main publication tuned defaults, all-random initialization. |
| `publication-rng-random` | Preserved RNG-random experimental variant. |
| `publication-rng-mst-config` | Preserved RNG-with-MST-configuration control. |
| `library` | Existing standalone-library defaults, including its PCA contexts. |
| `xpysom-untuned` | Untuned comparison baseline, stored in `baselines/`. |

The old bare JSON filenames remain accepted as aliases. Explicit existing file
paths take precedence. Named profiles work from any working directory, including
installed wheels and cluster workers. Historical absolute cluster paths must be
replaced with a profile name or an actual custom JSON path.

All profile JSON values were copied unchanged. The numerical results, supporting
tables and manually edited manuscript figures are independent of this move.

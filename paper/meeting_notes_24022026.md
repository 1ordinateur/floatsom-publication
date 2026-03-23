Things to change in paper: 

- Put random scaling together at top w randmo result
- Mention that this shows superior scaling
- But at larger datasets, we'd go into disk mode. To avoid this, we can do multiGPU operations. 
- We benchmark on 'full' to show worst case performance for random, because we recommend everyone run random. 

Things to do: 

- Equivalance: XPySOM v Batch
- Fix paper math
- Double check statistics for math correctness 

Paper flow: 

- OPTUNA - we pool random + full runs together for analysis (given they're not different) 
- OPTUNA results: 
- Colors v Batch 
- Hex v MST / RNG <- Show representative figure of RNG v MST v Hex on Circles
- Scaling: 
- Colors v Batch 
- Hex v MST / RNG 

Finish 

- As many GPUs as you have available 
- Run Random 
- Run Colors it's not much <- Batch at the extremity 
- Run RNG <- switch MST 

Sebastian presentation: 

- Rationale for why tRNA - problems with mRNA ST (show ST images of Felix)
- Plug in for Spatial Transcriptomics, where can we do this? Remove mRNA amplification steps
    - What resolution is required? 
    - Can we do meaningful analysis at this level? 
    - What questions do we have? <- Spatial relationship between tRNA modifications and rRNAs / their modifications, do they tend to co-occur within one cell? To do this, best to have very large cells. 
    - Lymphocytes bad candidate, but needs to be metabolically active... Liver is good candidate considering cell size
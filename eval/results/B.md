# Task B results

300 queries, top 100 results per arm.

```
#    Model           NDCG@10    P@10    P@10-l2    Recall@100-l2    MAP@100-l2
---  --------------  ---------  ------  ---------  ---------------  ------------
a    bm25            0.359      0.536   0.152      0.073ᵇ           0.015ᵇ
b    bge-small       0.342      0.518   0.136      0.058            0.011
c    bm25+bge-small  0.365ᵇ     0.541ᵇ  0.157ᵇ     0.074ᵇ           0.016ᵇ
```

Letters after a score list the arms it beats with p < 0.05 (paired randomization test, 2000 permutations).

- Silver truth from NTSB coding: 2 = same defining event and a shared cause finding, 1 = one of the two. Metrics ending in -l2 count only grade 2 as relevant.
- Narrative text only; finding/occurrence codes and the probable-cause statement are not in the indexed text.

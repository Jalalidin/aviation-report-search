# Task B results

300 queries, top 100 results per arm.

```
#    Model            NDCG@10    P@10       P@10-l2    Recall@100-l2    MAP@100-l2
---  ---------------  ---------  ---------  ---------  ---------------  ------------
a    bm25             0.386ᵇ     0.567ᵇ     0.182      0.079ᵇ           0.018ᵇ
b    bge-small        0.358      0.537      0.166      0.064            0.013
c    bm25+bge-small   0.400ᵇ     0.575ᵇ     0.193ᵇ     0.081ᵇ           0.018ᵇ
d    qwen3-0.6b       0.411ᵃᵇ    0.602ᵃᵇᶜ   0.189ᵇ     0.102ᵃᵇᶜ         0.022ᵇ
e    bm25+qwen3-0.6b  0.438ᵃᵇᶜᵈ  0.626ᵃᵇᶜᵈ  0.215ᵃᵇᶜᵈ  0.107ᵃᵇᶜ         0.023ᵃᵇᶜ
```

Letters after a score list the arms it beats with p < 0.05 (paired randomization test, 2000 permutations).

- Silver truth from NTSB coding: 2 = same defining event and a shared cause finding, 1 = one of the two. Metrics ending in -l2 count only grade 2 as relevant.
- Narrative text only; finding/occurrence codes and the probable-cause statement are not in the indexed text.

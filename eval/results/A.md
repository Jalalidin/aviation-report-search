# Task A results

120 queries, top 100 results per arm.

```
#    Model                  MRR@10    Recall@1    Recall@10    Recall@100    NDCG@10
---  ---------------------  --------  ----------  -----------  ------------  ---------
a    bm25                   0.433ᵇ    0.375       0.542        0.833         0.460ᵇ
b    bge-small              0.346     0.292       0.500        0.733         0.382
c    bm25[table]            0.445ᵇ    0.375       0.592        0.775         0.480ᵇ
d    bm25[merged]           0.507ᵃᵇᵉ  0.442ᵃᵇᵉ    0.650ᵃᵇ      0.883ᵇᶜ       0.541ᵃᵇᵉ
e    bm25+bge-small         0.432ᵇ    0.358ᵇ      0.617ᵇ       0.867ᵇ        0.475ᵇ
f    bm25[table⊕narr]       0.487ᵇ    0.408ᵇ      0.667ᵃᵇ      0.875ᵇᶜ       0.530ᵃᵇ
g    bm25+bge-small+rerank  0.522ᵃᵇᵉ  0.450ᵇ      0.667ᵃᵇ      0.867ᵇ        0.557ᵃᵇᵉ
```

Letters after a score list the arms it beats with p < 0.05 (paired randomization test, 2000 permutations).

- Known-item task: only the target report counts as relevant, so absolute scores understate quality. Compare arms with each other.
- qwen3-0.6b embeddings not built yet; reranking bm25+bge-small instead.

## MRR@10 by query style

```
style                  keywords  mismatch  question
arm                                                
bm25                      0.692     0.204     0.404
bge-small                 0.523     0.174     0.340
bm25[table]               0.796     0.066     0.472
bm25[merged]              0.785     0.246     0.490
bm25+bge-small            0.595     0.251     0.449
bm25[table⊕narr]          0.772     0.229     0.462
bm25+bge-small+rerank     0.724     0.412     0.429
```

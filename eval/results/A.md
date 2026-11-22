# Task A results

120 queries, top 100 results per arm.

```
#    Model                           MRR@10          Recall@1       Recall@10       Recall@100         NDCG@10
---  ------------------------------  --------------  -------------  --------------  -----------------  ---------------
a    bm25                            0.433ᵇᵉᶠʰ       0.375ᵉᶠʰ       0.542ᵉʰ         0.833ᵇᵉʰ           0.460ᵇᵉʰ
b    bge-small                       0.346ᵉ          0.292ᵉ         0.500           0.733              0.382ᵉ
c    qwen3-0.6b                      0.415ᵇᵉʰ        0.342ᵉʰ        0.592ᵉʰ         0.817ᵉʰ            0.457ᵇᵉʰ
d    bm25[table]                     0.445ᵇᵉᶠʰ       0.375ᵉᶠʰ       0.592ᵉʰ         0.775ᵉʰ            0.480ᵇᵉᶠʰ
e    bge-small[table]                0.230           0.175          0.400           0.667              0.270
f    qwen3-0.6b[table]               0.339ᵉ          0.258          0.525ᵉʰ         0.783ᵉʰ            0.383ᵉ
g    bm25[merged]                    0.507ᵃᵇᶜᵉᶠʰʲᵐ   0.442ᵃᵇᶜᵉᶠʰʲᵐ  0.650ᵃᵇᵉᶠʰ      0.883ᵇᵈᵉᶠʰ         0.541ᵃᵇᶜᵉᶠʰʲᵐ
h    bge-small[merged]               0.292           0.242          0.417           0.667              0.322
i    qwen3-0.6b[merged]              0.467ᵇᵉᶠʰᵐ      0.392ᵇᵉᶠʰ      0.642ᵇᵉᶠʰ       0.842ᵇᵉʰ           0.509ᵇᵉᶠʰ
j    bm25+bge-small                  0.432ᵇᵉᶠʰ       0.358ᵇᵉʰ       0.617ᵇᵉʰ        0.867ᵇᵈᵉʰ          0.475ᵇᵉᶠʰ
k    bm25+qwen3-0.6b                 0.465ᵇᵉᶠʰᵐ      0.400ᵇᵉᶠʰᵐ     0.600ᵉʰ         0.917ᵃᵇᶜᵈᵉᶠʰⁱ      0.497ᵇᵉᶠʰ
l    bm25[table⊕narr]                0.487ᵇᵉᶠʰᵐ      0.408ᵇᵉᶠʰᵐ     0.667ᵃᵇᵉᶠʰ      0.875ᵇᵈᵉᶠʰ         0.530ᵃᵇᵉᶠʰ
m    qwen3-0.6b[table⊕narr]          0.405ᵉᶠʰ        0.317ᵉ         0.650ᵃᵇᵉᶠʰ      0.875ᵇᵈᵉᶠʰ         0.462ᵇᵉᶠʰ
n    bm25+qwen3-0.6b+rerank          0.532ᵃᵇᶜᵉᶠʰʲᵐ   0.450ᵇᶜᵉᶠʰᵐ    0.708ᵃᵇᶜᵉᶠʰᵏ    0.917ᵃᵇᶜᵈᵉᶠʰ       0.574ᵃᵇᶜᵉᶠʰʲᵏᵐ
o    bm25[merged]+qwen3-0.6b         0.509ᵃᵇᶜᵉᶠʰʲᵏᵐ  0.417ᵇᶜᵉᶠʰᵐ    0.700ᵃᵇᶜᵈᵉᶠʰᵏ   0.950ᵃᵇᶜᵈᵉᶠᵍʰⁱʲˡᵐ  0.554ᵃᵇᶜᵉᶠʰʲᵏᵐ
p    bm25[merged]+qwen3-0.6b+rerank  0.544ᵃᵇᶜᵈᵉᶠʰʲᵐ  0.467ᵇᶜᵉᶠʰᵐ    0.742ᵃᵇᶜᵈᵉᶠʰʲᵏ  0.950ᵃᵇᶜᵈᵉᶠᵍʰⁱʲˡᵐ  0.591ᵃᵇᶜᵈᵉᶠʰʲᵏᵐ
```

Letters after a score list the arms it beats with p < 0.05 (paired randomization test, 2000 permutations).

- Known-item task: only the target report counts as relevant, so absolute scores understate quality. Compare arms with each other.
- bm25[merged]+qwen3-0.6b: keyword search reads table + narrative, qwen3-0.6b reads the narrative; its reranker reads table + narrative. The other reranked arm reads the narrative only.

## MRR@10 by query style

```
style                           keywords  mismatch  question
arm                                                         
bm25                               0.692     0.204     0.404
bge-small                          0.523     0.174     0.340
qwen3-0.6b                         0.663     0.243     0.339
bm25[table]                        0.796     0.066     0.472
bge-small[table]                   0.431     0.048     0.211
qwen3-0.6b[table]                  0.536     0.123     0.357
bm25[merged]                       0.785     0.246     0.490
bge-small[merged]                  0.458     0.093     0.326
qwen3-0.6b[merged]                 0.692     0.277     0.432
bm25+bge-small                     0.595     0.251     0.449
bm25+qwen3-0.6b                    0.719     0.269     0.405
bm25[table⊕narr]                   0.772     0.229     0.462
qwen3-0.6b[table⊕narr]             0.647     0.202     0.366
bm25+qwen3-0.6b+rerank             0.699     0.461     0.436
bm25[merged]+qwen3-0.6b            0.778     0.301     0.448
bm25[merged]+qwen3-0.6b+rerank     0.730     0.369     0.533
```

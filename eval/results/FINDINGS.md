# Findings so far

Hand-written reading of the result tables. Update whenever the tables change.
Each line starting with "- " is shown on the demo site.

- Searching table and narrative text together beats searching the narrative alone (top-10 hit rate 65% vs 54%, statistically significant). Joining them into one text and fusing two separate searches perform about the same.
- Keyword search depends heavily on wording: mean reciprocal rank 0.69 when the query uses technical keywords, 0.20 when it describes the same event in different words.
- A small general-purpose embedding model (BGE small) on its own is weaker than keyword search on this data.
- Adding a reranker to the hybrid gives the best scores so far and roughly doubles performance on differently-worded queries (0.20 to 0.41), at about 2.4 seconds per query.
- Qwen3-Embedding results are not in yet: its run was still in progress when this was written.

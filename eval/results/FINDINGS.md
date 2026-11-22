# Findings so far

Hand-written reading of the result tables. Update whenever the tables change.
Each line starting with "- " is shown on the demo site.

- Using table and narrative together helps keyword search and the hybrid, and the gain is statistically significant. Keyword search over both texts finds the right report in the top 10 for 65% of queries, against 54% on the narrative alone. Giving the hybrid's keyword half the table text lifts its top-10 hit rate from 60% to 70%.
- On short search queries, Qwen3-Embedding alone is level with keyword search (MRR@10 0.415 vs 0.433, no significant difference). It is significantly better than the small embedding model, BGE small (0.346).
- The best scores come from combining everything: keyword search on table + narrative, fused with Qwen3, then reranked (top-10 hit rate 74%, MRR@10 0.544). It is significantly better than keyword search or Qwen3 on the narrative alone. Its lead over plain keyword search on table + narrative (65%, 0.507) is not statistically proven with 120 queries.
- The reranker earns its cost on differently-worded queries, where it lifts MRR@10 from 0.27 to 0.46. It costs about 2.4 seconds per query; the hybrid without it takes about 20 milliseconds.
- Keyword search depends heavily on wording: MRR@10 is 0.69 when the query uses technical keywords and 0.20 when it describes the same event in other words. Table text makes this sharper: 0.80 with keywords, 0.07 with other words.
- Embedding models score lower on the coded table text alone than on the narrative (Qwen3: 0.339 vs 0.415; BGE small: 0.230 vs 0.346). Joined with the narrative into one text, Qwen3 scores higher (0.467) and BGE small lower (0.292); neither change against the narrative alone is statistically proven. For Qwen3, one joined text is significantly better than fusing two separate searches (0.467 vs 0.405).
- For finding similar incidents (the query is a whole report), Qwen3 beats keyword search (nDCG@10 0.411 vs 0.386) and the hybrid of the two beats both (0.438). All three differences are significant. The hybrid was best in all five random draws of 300 query reports.

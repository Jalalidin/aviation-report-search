# aviation-report-search

A small, reproducible test bench that answers one question with numbers:

> **Which search method finds the right aviation accident reports best — and how should the
> table data and the narrative text be used together?**

It runs on public data from the US National Transportation Safety Board (NTSB): 31,453
accidents and incidents from 2008 to 2026, of which 28,190 have a written narrative.

Nothing here invents a new algorithm. It takes existing methods and existing models, runs them
on the same reports with the same questions, and scores them the same way.

**Live demo and results: <https://jalalidin.github.io/aviation-report-search/>** — result
tables, a side-by-side view of what each method returned for every test query, and a search you
can try in the browser.

## The methods being compared

| Name in the tables | What it is, in plain words |
|---|---|
| `bm25` | Traditional keyword search. Counts matching words, weighting rare words more. No AI. Fast, exact, but blind to synonyms. |
| `qwen3-0.6b` | AI embedding model (Qwen3-Embedding, 0.6B size, from Alibaba). Turns each report into a list of numbers so reports with similar *meaning* sit close together. |
| `bge-small`, `bge-m3` | Other embedding models, for comparison: one much smaller, one similar in size. |
| `bm25+<model>` | Hybrid: run keyword search and embedding search separately, then merge the two ranked lists (reciprocal rank fusion). |
| `...+rerank` | Take the top 50 from the hybrid and let a second, slower model re-read each one next to the query and re-order them. |

And three ways of presenting a report to the search method:

| Label | Text that is searched |
|---|---|
| *(no label)* | **Narrative** only: the investigators' analysis and factual narrative. |
| `[table]` | **Table** only: the coded fields written out as a short description (aircraft, damage, injuries, weather, sequence of events, findings) plus the one-line probable cause. |
| `[merged]` | Both, joined into one text (*early fusion*). |
| `[table⊕narr]` | Both, searched separately and the two result lists merged (*late fusion*). |

## How the test works

**Task A — "find the report I'm thinking of."** 120 search queries for 40 known reports. Each
report has three queries: technical keywords, a natural question, and a deliberately
different wording that avoids the report's own terms (the "vocabulary mismatch" case, e.g.
*"engine seized on the way to an oil change"* for a connecting-rod bearing failure). A method
scores well if the known report comes back near the top. Queries are in
[eval/queries.yaml](eval/queries.yaml).

**Task B — "find incidents similar to this one."** 300 reports are used as queries. Two
reports count as similar if NTSB's own investigators coded them alike: grade 2 = same defining
event *and* a shared cause finding; grade 1 = one of the two. Those codes are hidden from the
searched text, so a method has to recognise similarity from the narrative alone.

**Reading the numbers** (all between 0 and 1, higher is better):

| Metric | Meaning |
|---|---|
| `recall@10` | In how many searches was the right report in the first 10 results? |
| `mrr@10` | How high up was it? 1.0 = always first, 0.5 = typically second. |
| `ndcg@10` | Quality of the first 10 results, rewarding better answers placed higher. |
| `precision@10-l2` | (Task B) Share of the first 10 results that are grade-2 similar. |
| `ms/query` | Search time per query in milliseconds. |

A small letter after a score (ᵃ, ᵇ, …) means "significantly better than the method labelled
with that letter" — the difference is unlikely to be luck (p < 0.05, paired randomization test).
**No letter means the difference is not proven**, however large it looks.

## Results so far

Status: keyword search, a small embedding model (BGE small), their hybrid and a reranker are
scored. **Qwen3-Embedding and BGE-M3 are not in these tables yet.** Full tables with
significance marks are in [eval/results/](eval/results/); a plain reading of them is in
[eval/results/FINDINGS.md](eval/results/FINDINGS.md).

Task A — find a known report (120 queries):

| Method | Top-10 hit rate | MRR@10 |
|---|---|---|
| Keyword, narrative only | 0.542 | 0.433 |
| Keyword, table only | 0.592 | 0.445 |
| Keyword, table + narrative as one text | 0.650 | 0.507 |
| Keyword, table and narrative fused | 0.667 | 0.487 |
| Embedding: BGE small | 0.500 | 0.346 |
| Hybrid: keyword + BGE small | 0.617 | 0.432 |
| Hybrid + reranker | 0.667 | 0.522 |

Same task, MRR@10 by how the query was phrased:

| Method | Keywords | Question | Mismatch |
|---|---|---|---|
| Keyword, narrative only | 0.692 | 0.404 | 0.204 |
| Embedding: BGE small | 0.523 | 0.340 | 0.174 |
| Hybrid + reranker | 0.724 | 0.429 | 0.412 |

Task B — find similar incidents (300 queries): keyword 0.359, BGE small 0.342, hybrid 0.365
(nDCG@10). Significance for these three is in [eval/results/B.md](eval/results/B.md).

## Running it

One-time setup (macOS): install [uv](https://docs.astral.sh/uv/) and `brew install mdbtools`,
download `avall.zip` from <https://data.ntsb.gov/avdata> and unzip it to
`data/raw/ntsb/extracted/avall.mdb`. Then:

```bash
uv sync                          # install Python packages
uv run avsearch snapshot         # Access database -> Parquet tables
uv run avsearch build            # one unified record per accident
uv run avsearch index            # keyword index
uv run avsearch embed bge-small  # embeddings (minutes)
uv run avsearch embed qwen3-0.6b # embeddings (about 80 minutes on an M1 Pro)
uv run avsearch eval             # run every method, write eval/results/
```

Try a search by hand:

```bash
uv run avsearch search "engine quit after takeoff, water in the fuel" --arm bm25
uv run avsearch search "engine quit after takeoff, water in the fuel" --arm qwen3-0.6b
```

`avsearch eval` uses whatever has been built: an embedding model that has not been run is
simply left out of the table.

Two ways to look at it in a browser:

```bash
uv run avsearch serve   # full search app over all reports, at http://127.0.0.1:8000
uv run avsearch site    # regenerate the static demo in docs/ (what the live demo link shows)
```

## Where things are

```
src/avsearch/
  sources/ntsb.py      read the NTSB database
  records/build.py     merge the tables into one record per accident; make the text views
  retrieve/            the methods: bm25.py, dense.py (embeddings), fusion.py, rerank.py
  eval/tasks.py        the two tasks and what counts as a right answer
  eval/run.py          runs everything and writes the tables
eval/queries.yaml      the Task A queries
eval/results/          result tables (.md to read, .csv to plot)
eval/runs/             the raw ranked lists behind every number
tests/                 checks that the building blocks behave
data/                  downloads, models, indexes (not in git)
```

To add another embedding model, add one line to `MODELS` in
[src/avsearch/retrieve/dense.py](src/avsearch/retrieve/dense.py), run `avsearch embed <name>`,
and re-run `avsearch eval`.

## Limits — read before quoting any number

- **Task A queries were written by an AI model, not an aviation engineer**, from the reports'
  own analysis text. A domain expert should review or rewrite them.
- **Task A counts one right answer per query.** Other reports may be equally good answers,
  so the absolute scores are pessimistic. The comparison between methods is the useful part.
- **Task B's "truth" is approximate.** It trusts NTSB's coding, and grade 1 is broad.
- **Long reports are cut at 512 tokens** (roughly the first 350–400 words) for the embedding
  models. The analysis section is placed first so it survives the cut. Splitting long reports
  into chunks is an experiment not yet run.
- **One dataset, one country.** Results on NTSB data may not carry over to other sources.
- 40 targets / 120 queries is small. Differences without a significance letter are not findings.

## Data source

Accident data comes from the NTSB aviation accident database (`avall.mdb`, snapshot dated
1 November 2026), published by the US National Transportation Safety Board at
<https://data.ntsb.gov/avdata>. The demo site in `docs/` contains excerpts of that public
data. This project is not affiliated with or endorsed by the NTSB.

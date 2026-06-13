# ps-sverige

Politisk stilometri: Sverige. Behavioural analysis of the Riksdag using open data.

**Live site:** https://freol35241.github.io/ps-sverige/

Context, motivation, and project options: see `IDEA.md`.
Working notes per day: see `diary/`.
Coding guidelines: see `CLAUDE.md`.
Static site (Swedish, hosted on GitHub Pages): see `site/`.

## What's here so far

End-to-end pipeline for a behavioural map of the current mandate
(2022–2026), built from the bulk `votering` dumps:

```
src/fetch.py              download & cache the votering + anforande zips + personlista
src/build_matrix.py       parse votes to a (424 MPs × 2,501 events) matrix
src/build_speeches.py     parse 54,914 speeches to a long table
src/embed.py              PCA into a 4-D behavioural space
src/figures.py            headline map + scree
src/defectors.py          per-MP defector index in full PCA space
src/annotated_map.py      composite figure with boundary-walkers called out
src/temporal.py           per-riksmöte PCA, Procrustes-aligned across years
src/drift_figures.py      trajectory map + small-multiples + Tidö-collapse zoom
src/agreement_trend.py    model-free vote-agreement trend per bloc
src/stylometry.py         TF-IDF speech embedding + speech-vs-vote comparison

# Phase 3 — revealed-preference compass
src/fetch_betankanden.py  pull utskottsforslag XML for every voted betänkande
src/parse_betankanden.py  parse per-punkt recommendation + reservation metadata
src/embed_recommendations.py     SBERT embed each vote event's recommendation
src/topics.py                    KMeans cluster events into 28 policy topics
src/party_topic_matrix.py        per-(party, topic) stance score in [-1, +1]
src/reasoning_embed.py           per-(party, topic) reasoning vectors from speech
src/compass_figures.py           heatmap, 2D compass, radar
src/gap_analysis.py              what-vs-why party-pair matrices and scatter
src/how_axis.py                  reservation co-authorship + theory-of-change lexicon
src/three_axis_figure.py         WHAT × WHY × HOW pair-level composite

# Session 4 — per-topic, reservation body text, temporal HOW
src/per_topic_axes.py            three-axis breakdown per topic
src/fetch_reservations.py        dokumentstatus XML (with full HTML body)
src/parse_reservations.py        extract 10,931 reservation paragraphs
src/embed_reservations.py        per-(party, topic) reservation reasoning vector
src/reservation_gap.py           reservation WHY vs speech WHY comparison
src/temporal_how.py              per-riksmöte lexicon + co-authorship drift
```

Outputs land in `data/processed/` (parquet + npz) and `figures/` (PNG).
Both are gitignored — regenerate end-to-end with:

```bash
.venv/bin/python -m src.fetch
.venv/bin/python -m src.build_matrix
.venv/bin/python -m src.build_speeches
.venv/bin/python -m src.embed
.venv/bin/python -m src.figures
.venv/bin/python -m src.defectors
.venv/bin/python -m src.annotated_map
.venv/bin/python -m src.temporal
.venv/bin/python -m src.drift_figures
.venv/bin/python -m src.agreement_trend
.venv/bin/python -m src.stylometry

# Phase 3 — needs sentence-transformers and ~5 min sequential
.venv/bin/python -m src.fetch_betankanden       # ~4 min
.venv/bin/python -m src.parse_betankanden
.venv/bin/python -m src.embed_recommendations   # ~2 min CPU
.venv/bin/python -m src.topics
.venv/bin/python -m src.party_topic_matrix
.venv/bin/python -m src.reasoning_embed         # ~8 min CPU
.venv/bin/python -m src.compass_figures
.venv/bin/python -m src.gap_analysis
.venv/bin/python -m src.how_axis
.venv/bin/python -m src.three_axis_figure
```

Phase-3 setup also needs `sentence-transformers` installed and downloads
`KBLab/sentence-bert-swedish-cased` (~500 MB) on first use.

Daily lab notes in `diary/` walk through the analytical decisions.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install requests numpy pandas scikit-learn scipy matplotlib pyarrow tqdm
```

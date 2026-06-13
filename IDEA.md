# Riksdagen Open Data API — Investigation Summary
 
*Probed 2026-06-12, ahead of the Swedish general election (autumn 2026). Context: exploring election-adjacent projects that apply fleet-behavior inversion and stylometry methods to parliamentary data.*
 
## The core insight
 
The Riksdag is one of the most instrumented parliaments in the world, and almost nobody treats its open data as a *behavioral* dataset. The methodological transfer is direct: a parliament is a fleet, individual voting and speech behavior are the tracks, and the hidden field to reconstruct is ideology, coalition structure, and authorship.
 
## API overview
 
Base URL: `https://data.riksdagen.se`. No authentication, no API key, JSON/XML/CSV output via `utformat` parameter. Full-year bulk dumps available under `/dataset/` (e.g. `anforande-202526.json.zip`, ~21 MB), which eliminates pagination for historical work.
 
| Endpoint | Content | Probed volume |
|---|---|---|
| `/voteringlista/` | Individual MP votes (Ja/Nej/Avstår/Frånvarande) with name, party, constituency, `votering_id`, linked `dok_id` | 565–687 vote events per riksmöte this mandate (~2,500 total × 349 MPs ≈ 870k records) |
| `/anforandelista/` | Chamber speeches with full text inline (`anforandetext`), speaker, party, debate context | Bulk year dumps confirmed |
| `/dokumentlista/?doktyp=mot` | Motions, full text per `dok_id` via `.text`/`.html` | 2,405–4,777 per year; decades of history |
| `/personlista/` | All 349 sitting MPs with stable `intressent_id`, party, constituency, birth year, photo URL | 349 confirmed |
 
Key structural properties: the stable `intressent_id` joins persons across votes, speeches, and documents. The `/voteringlista/` endpoint supports server-side grouping (e.g. `gruppering=votering_id` returns Ja/Nej/Frånvarande/Avstår tallies per vote event). Votes link to committee reports (`organ` field: FiU, AU, ...), giving coarse topical labels for free.
 
## Notable findings from the probe
 
Motion volume per riksmöte: 3,129 (2018/19), 4,777 (2021/22), 2,405 (2022/23), 2,922 (2023/24), 3,449 (2024/25), and 4,201 so far in 2025/26 with the year not yet over. The recent rise is suggestive for an LLM-authorship study, but confounded: election years historically spike (see 2021/22), so raw counts prove nothing — the stylometric signal in the text is what matters. A clean pre-ChatGPT baseline exists.
 
Abstention semantics: `Avstår` (present, abstaining) and `Frånvarande` (absent) are distinct codes. Strategic abstention is a party-discipline tactic and is itself a fingerprintable behavioral signal, not noise.
 
The 349 × ~2,500 vote matrix for the full mandate period is small by ML standards. Embedding, clustering, and drift analysis are an afternoon of compute, not an infrastructure project.
 
## The identified gap
 
Votes are not topically labeled beyond the committee of origin. The revealed-preference valkompass requires mapping vote events to policy positions, which means LLM classification of the underlying betänkanden (committee reports). This is tractable — the documents are retrievable as text — but it is the actual work of that project.
 
## Project ideas ranked against this data
 
**Riksdag stylometry** (fastest path). Fingerprint MPs from voting patterns and speech text: behavioral party membership vs. official party membership, drift over the mandate period, defection prediction. The vote matrix makes a first embedding feasible in a weekend, and the resulting "behavioral map of the Riksdag vs. the official party map" is a strong standalone visual in the Plimsoll Line register. Foundation for the two ideas below.
 
**The revealed-preference valkompass.** Election compasses ask parties what they say; this one matches voters against how parties actually voted. The gap between declared and revealed positions is itself the story. Requires the topical-labeling work described above.
 
**Regeringsbildning Monte Carlo.** Probabilistic forecasts of *feasible coalitions* (not just seat counts), combining poll distributions with a behavioral compatibility matrix derived from the stylometry work. Mediagenic and honest about uncertainty.
 
**The motion authorship detector.** Estimate the fraction of riksdag motions that are LLM-drafted, tracked over time since 2023, against the pre-2023 stylometric baseline. Must control for the election-year volume confound. Headline writes itself.
 
**Kommunprotokoll mining** (not served by this API). The data-dark layer: 290 municipalities publishing decisions as unstructured PDF protocols. An LLM extraction pipeline producing a searchable decision record, scoped to Göteborg first. Highest civic durability — the dataset outlives the election.
 
## Recommended sequencing
 
Pull the full vote matrix → build the behavioral embedding → publish the map. That single artifact validates the data pipeline, produces an immediate publishable result, and is the shared foundation for the valkompass and the coalition simulator. The motion authorship study runs independently and can proceed in parallel from the bulk text dumps.
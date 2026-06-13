<script>
  import { onMount } from 'svelte';
  import {
    pickNextTopic, pickExcerptsForTopic, updatePosterior,
    computeMatch, initialPosterior, PARTIES,
  } from './matching.js';

  import topicsRaw from '../data/topics.json';
  import vectorsRaw from '../data/vectors.json';
  import excerptsRaw from '../data/excerpts.json';
  import partiesRaw from '../data/parties.json';

  const PARTY_INFO = Object.fromEntries(partiesRaw.map(p => [p.code, p]));

  // Use up to TOTAL_QUESTIONS, keep going until we run out of topics with
  // at least 2 parties of excerpts.
  const TOTAL_QUESTIONS = 10;

  let reasoning = $state(null);
  let dataReady = $state(false);
  let phase = $state(/** @type {'intro' | 'q' | 'done'} */ ('intro'));
  let asked = $state(new Set());
  let answers = $state([]);
  let posterior = $state(initialPosterior());
  let currentTopic = $state(null);
  let currentOptions = $state([]);
  let result = $state(null);

  onMount(async () => {
    // Lazy-load the heavy reasoning vectors only when the user enters the tool.
    const base = import.meta.env.BASE_URL.replace(/\/$/, '');
    try {
      const resp = await fetch(`${base}/data/reasoning.json`);
      reasoning = await resp.json();
    } catch (e) {
      console.error('reasoning load failed', e);
      reasoning = {};
    }
    dataReady = true;
  });

  function nextQuestion() {
    // Pick a topic with at least 2 parties with excerpts.
    let topicId = null;
    let tries = 0;
    while (tries < 28) {
      topicId = pickNextTopic(topicsRaw, vectorsRaw, asked, posterior);
      if (topicId == null) break;
      const opts = pickExcerptsForTopic(topicId, vectorsRaw, excerptsRaw, posterior);
      if (opts.length >= 2) {
        currentTopic = topicsRaw.find(t => t.id === topicId);
        currentOptions = opts;
        return;
      }
      asked.add(topicId);
      asked = new Set(asked);
      tries++;
    }
    finalize();
  }

  function start() {
    phase = 'q';
    asked = new Set();
    answers = [];
    posterior = initialPosterior();
    result = null;
    nextQuestion();
  }

  function chooseOption(optionIdx) {
    const chosen = currentOptions[optionIdx];
    answers = [...answers, { topicId: currentTopic.id, party: chosen.party }];
    asked.add(currentTopic.id);
    asked = new Set(asked);
    posterior = updatePosterior(posterior, chosen.party, currentOptions);
    if (answers.length >= TOTAL_QUESTIONS) {
      finalize();
    } else {
      nextQuestion();
    }
  }

  function skipQuestion() {
    asked.add(currentTopic.id);
    asked = new Set(asked);
    if (answers.length >= TOTAL_QUESTIONS) {
      finalize();
    } else {
      nextQuestion();
    }
  }

  function finalize() {
    if (!answers.length) {
      phase = 'intro';
      return;
    }
    result = computeMatch(answers, vectorsRaw, reasoning || {}, topicsRaw);
    phase = 'done';
  }

  function reset() {
    phase = 'intro';
    asked = new Set();
    answers = [];
    posterior = initialPosterior();
    result = null;
  }

  function formatPct(x) {
    return Math.round(Math.max(0, Math.min(1, x)) * 100) + ' %';
  }
</script>

<div class="voter-tool">
  {#if phase === 'intro'}
    <div class="card">
      <h2>Tredimensionell valkompass</h2>
      <p>
        Du får läsa korta avsnitt baserade på riksdagens reservationer.
        Texterna är anonymiserade — du vet inte vilket parti som står bakom
        förrän i slutet. Välj det resonemang som övertygar dig mest.
      </p>
      <p class="meta">
        Cirka {TOTAL_QUESTIONS} frågor, 5 minuter. Allt sker i din webbläsare —
        inga svar lämnar din enhet.
      </p>
      <button onclick={start} class="btn-primary" disabled={!dataReady}>
        {dataReady ? 'Starta →' : 'Laddar …'}
      </button>
    </div>

  {:else if phase === 'q' && currentTopic && currentOptions.length}
    <div class="card">
      <p class="progress">Fråga {answers.length + 1} av {TOTAL_QUESTIONS}</p>
      <h3>{currentTopic.question || `${currentTopic.label} — vilket resonemang ligger närmast dig?`}</h3>
      <p class="instruction">
        Texterna nedan är hämtade från reservationer och betänkanden i riksdagen.
        Partinamnen är dolda — välj det resonemang som övertygar dig mest.
      </p>
      <div class="options">
        {#each currentOptions as opt, i}
          <button class="option" onclick={() => chooseOption(i)}>
            <span class="opt-text">{opt.text}</span>
          </button>
        {/each}
      </div>
      <p class="meta-row">
        <button onclick={skipQuestion} class="link">Inget passar — hoppa över</button>
        <button onclick={reset} class="link">Börja om</button>
      </p>
    </div>

  {:else if phase === 'done' && result}
    <div class="card result-card">
      <h2>Din matchning</h2>
      <p class="lede-result">
        Här är vilka partier som ligger närmast dig på de tre axlarna,
        baserat på dina {answers.length} svar.
      </p>
      <div class="three-axis">
        {#each [
          { key: 'what', label: 'VAD', explain: 'vad partiet faktiskt gör i kammaren' },
          { key: 'why',  label: 'VARFÖR', explain: 'vilka värden partiet hänvisar till' },
          { key: 'how',  label: 'HUR', explain: 'vilka medel partiet föredrar — marknad, reglering, prevention, straff' },
        ] as axis}
          {@const top = result[axis.key].top}
          {@const partyInfo = PARTY_INFO[top.party]}
          {@const evidence = result[axis.key].evidence[top.party] || []}
          <div class="axis-row" style:--accent={partyInfo.color}>
            <div class="axis-meta">
              <span class="axis-label">{axis.label}</span>
              <span class="axis-explain">{axis.explain}</span>
            </div>
            <div class="axis-result">
              <span class="axis-party" style:background={partyInfo.color}>
                {top.party}
              </span>
              <span class="axis-party-name">{partyInfo.name}</span>
              <span class="axis-sim">{formatPct(top.sim)}</span>
            </div>
            {#if evidence.length}
              <div class="axis-evidence">
                <span class="evidence-label">närmst dig på:</span>
                <span class="evidence-topics">
                  {#each evidence.slice(0, 3) as ev, i}
                    <span>{ev.label}</span>{i < Math.min(evidence.length, 3) - 1 ? ' · ' : ''}
                  {/each}
                </span>
              </div>
            {/if}
          </div>
        {/each}
      </div>
      <p class="note">
        Att olika partier dyker upp på olika axlar betyder inte att du är
        förvirrad. Det betyder att de flesta partier är inkonsekventa över de
        här tre dimensionerna — och att en vanlig valkompass döljer det.
      </p>

      <details class="all-scores">
        <summary>Visa alla partier per axel</summary>
        <div class="scores-grid">
          {#each ['what', 'why', 'how'] as axis}
            <div class="scores-col">
              <h4>{axis.toUpperCase()}</h4>
              {#each PARTIES.toSorted((a,b) => result[axis].all[b] - result[axis].all[a]) as p}
                <div class="score-row">
                  <span class="score-mark" style:background={PARTY_INFO[p].color}>{p}</span>
                  <span class="score-val">{formatPct(result[axis].all[p])}</span>
                </div>
              {/each}
            </div>
          {/each}
        </div>
      </details>

      <button onclick={reset} class="btn-secondary">Börja om</button>
    </div>

  {:else}
    <div class="card">
      <p>Förbereder verktyget …</p>
    </div>
  {/if}
</div>

<style>
  .voter-tool {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #222;
  }
  .card {
    background: #FFF;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 2rem;
    max-width: 720px;
    margin: 0 auto;
  }
  .card h2 { margin-top: 0; font-size: 1.5rem; }
  .card h3 { font-size: 1.2rem; line-height: 1.4; margin: 1rem 0 1.5rem; }
  .meta { font-size: 0.85rem; color: #666; }
  .meta-row {
    display: flex;
    gap: 1rem;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #666;
    margin: 0;
  }
  .progress {
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    margin: 0;
  }
  .instruction {
    font-size: 0.9rem;
    color: #666;
    line-height: 1.5;
    margin: 0 0 1rem;
    padding-bottom: 1rem;
    border-bottom: 1px dashed #DDD;
  }
  .options {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin: 1.5rem 0;
  }
  .option {
    background: #F5F5F2;
    border: 1px solid #DDD;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    text-align: left;
    font-size: 1rem;
    line-height: 1.5;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s, transform 0.05s;
    font-family: inherit;
    color: inherit;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .option:hover {
    border-color: #2A4A7F;
    background: #FFF;
  }
  .option:active { transform: translateY(1px); }
  .opt-text { font-size: 1rem; line-height: 1.5; }
  .opt-source {
    font-size: 0.78rem;
    color: #888;
    font-style: italic;
  }
  .btn-primary {
    background: #2A4A7F;
    color: white;
    border: 0;
    border-radius: 6px;
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    cursor: pointer;
    font-family: inherit;
  }
  .btn-primary[disabled] { background: #999; cursor: progress; }
  .btn-primary:not([disabled]):hover { background: #1A3158; }
  .btn-secondary {
    background: transparent;
    color: #2A4A7F;
    border: 1px solid #2A4A7F;
    border-radius: 6px;
    padding: 0.6rem 1.25rem;
    font-size: 0.95rem;
    cursor: pointer;
    font-family: inherit;
    margin-top: 1.5rem;
  }
  .btn-secondary:hover { background: #2A4A7F; color: white; }
  .link {
    background: none;
    border: 0;
    color: #2A4A7F;
    text-decoration: underline;
    cursor: pointer;
    font: inherit;
    padding: 0;
    font-family: inherit;
  }

  /* Result */
  .lede-result { color: #555; font-size: 0.95rem; }
  .three-axis {
    margin: 2rem 0;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .axis-row {
    --accent: #2A4A7F;
    display: grid;
    grid-template-columns: minmax(180px, 1.5fr) 2fr;
    grid-template-areas:
      "meta result"
      "evidence evidence";
    align-items: stretch;
    gap: 0.75rem 1rem;
    padding: 1.25rem;
    background: #F5F5F2;
    border-left: 4px solid var(--accent);
    border-radius: 6px;
  }
  .axis-meta { grid-area: meta; }
  .axis-result { grid-area: result; }
  .axis-evidence {
    grid-area: evidence;
    padding-top: 0.5rem;
    border-top: 1px dashed #DDD;
    font-size: 0.85rem;
    color: #555;
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
  }
  .evidence-label { font-weight: 600; color: #777; }
  .evidence-topics span { color: #2A4A7F; }
  .axis-meta { display: flex; flex-direction: column; gap: 0.25rem; }
  .axis-label {
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    color: #444;
    font-weight: 700;
  }
  .axis-explain { font-size: 0.85rem; color: #666; line-height: 1.4; }
  .axis-result {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
  }
  .axis-party {
    color: white;
    padding: 0.3rem 0.75rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.03em;
  }
  .axis-party-name { flex: 1; font-size: 0.95rem; color: #222; }
  .axis-sim { font-weight: 600; font-size: 1.05rem; color: #444; }
  .note {
    font-size: 0.9rem;
    color: #555;
    background: #FAF6E8;
    border-left: 3px solid #C5B9A3;
    padding: 1rem;
    margin: 1.5rem 0;
    line-height: 1.5;
  }

  .all-scores { margin: 1.5rem 0; }
  .all-scores summary {
    cursor: pointer;
    font-size: 0.95rem;
    color: #2A4A7F;
    padding: 0.5rem 0;
  }
  .scores-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
    margin-top: 1rem;
  }
  .scores-col h4 {
    margin: 0 0 0.5rem;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    color: #666;
  }
  .score-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
    font-size: 0.85rem;
  }
  .score-mark {
    color: white;
    padding: 0.15rem 0.4rem;
    border-radius: 3px;
    font-weight: 600;
    min-width: 30px;
    text-align: center;
  }
  .score-val { color: #444; font-weight: 500; }

  @media (max-width: 540px) {
    .card { padding: 1.25rem; }
    .axis-row { grid-template-columns: 1fr; }
    .scores-grid { grid-template-columns: 1fr; gap: 1rem; }
  }
</style>

<script>
  import { onMount } from 'svelte';
  import {
    matchWhat, matchWhy, matchHow, pickNextQuestion, PARTIES, DIMS,
  } from './matching.js';

  import partiesRaw from '../data/parties.json';
  import vectorsRaw from '../data/vectors.json';
  import topicsRaw from '../data/topics.json';
  import questionsRaw from '../data/questions.json';

  const PARTY_INFO = Object.fromEntries(partiesRaw.map(p => [p.code, p]));
  const TOPIC_INFO = Object.fromEntries(topicsRaw.map(t => [t.id, t]));

  // Number of questions per round.
  const ROUNDS = [
    {
      key: 'what',
      label: 'VAD',
      title: 'Vad bör politiken faktiskt göra?',
      intro: 'Sju påståenden om konkret politik. Ange om du instämmer eller inte.',
      target: 7,
      pool: questionsRaw.what,
    },
    {
      key: 'why',
      label: 'VARFÖR',
      title: 'Vilka värden vägleder dig?',
      intro: 'Fem påståenden om vad som är viktigt i grunden.',
      target: 5,
      pool: questionsRaw.why,
    },
    {
      key: 'how',
      label: 'HUR',
      title: 'Vilka medel föredrar du?',
      intro: 'Fem påståenden om vägen dit – inte vad utan hur.',
      target: 5,
      pool: questionsRaw.how,
    },
  ];

  const DIM_LABELS = {
    market_vs_regulation: { left: 'marknad', right: 'reglering' },
    prevention_vs_punishment: { left: 'förebyggande', right: 'straff' },
    state_vs_local: { left: 'stat', right: 'lokalt' },
    universal_vs_targeted: { left: 'generellt', right: 'riktat' },
  };

  let reasoning = $state(null);
  let dataReady = $state(false);

  /** @type {'intro' | 'q' | 'transition' | 'done'} */
  let phase = $state('intro');
  let roundIdx = $state(0);
  let answersPerAxis = $state({ what: [], why: [], how: [] });
  let asked = $state({ what: new Set(), why: new Set(), how: new Set() });
  let currentQuestion = $state(null);
  let result = $state(null);

  onMount(async () => {
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

  function pickNext() {
    const round = ROUNDS[roundIdx];
    const pool = round.pool.filter(q => !asked[round.key].has(q.id));
    if (pool.length === 0 || answersPerAxis[round.key].length >= round.target) {
      advanceRound();
      return;
    }
    const q = pickNextQuestion(pool, answersPerAxis[round.key], round.key,
                                vectorsRaw, reasoning || {});
    currentQuestion = q;
  }

  function start() {
    phase = 'q';
    roundIdx = 0;
    answersPerAxis = { what: [], why: [], how: [] };
    asked = { what: new Set(), why: new Set(), how: new Set() };
    result = null;
    pickNext();
  }

  /** @param {-1 | 0 | 1} value */
  function answer(value) {
    const round = ROUNDS[roundIdx];
    answersPerAxis[round.key].push({ question: currentQuestion, value });
    answersPerAxis = { ...answersPerAxis };
    asked[round.key].add(currentQuestion.id);
    asked = { ...asked, [round.key]: new Set(asked[round.key]) };
    pickNext();
  }

  function advanceRound() {
    if (roundIdx + 1 < ROUNDS.length) {
      roundIdx += 1;
      phase = 'transition';
    } else {
      finalize();
    }
  }

  function continueAfterTransition() {
    phase = 'q';
    pickNext();
  }

  function finalize() {
    const what = matchWhat(answersPerAxis.what, vectorsRaw);
    const why = matchWhy(answersPerAxis.why, reasoning || {});
    const how = matchHow(answersPerAxis.how, vectorsRaw);
    result = { what, why, how };
    phase = 'done';
  }

  function reset() {
    phase = 'intro';
    roundIdx = 0;
    answersPerAxis = { what: [], why: [], how: [] };
    asked = { what: new Set(), why: new Set(), how: new Set() };
    currentQuestion = null;
    result = null;
  }

  function formatPct(x) {
    return Math.round(Math.max(0, Math.min(1, x)) * 100) + ' %';
  }

  function totalAnswered() {
    return answersPerAxis.what.length + answersPerAxis.why.length + answersPerAxis.how.length;
  }
  function totalQuestions() {
    return ROUNDS.reduce((s, r) => s + r.target, 0);
  }
  function currentRound() {
    return ROUNDS[roundIdx];
  }
  function currentRoundProgress() {
    return answersPerAxis[currentRound().key].length;
  }
</script>

<div class="voter-tool">
  {#if phase === 'intro'}
    <div class="card">
      <h2>Tredimensionell valkompass</h2>
      <p>
        Tre rundor – en per axel. I varje runda får du några påståenden att
        ta ställning till. När du är klar visar verktyget vilket parti som
        ligger närmast dig på respektive axel.
      </p>
      <ul class="axis-list">
        <li><strong>VAD</strong> – vilken politik du vill se</li>
        <li><strong>VARFÖR</strong> – vilka värden som vägleder dig</li>
        <li><strong>HUR</strong> – vilka medel och mekanismer du föredrar</li>
      </ul>
      <p class="meta">
        Cirka 17 frågor, 5 minuter. Allt sker i din webbläsare –
        ingenting lämnar din enhet.
      </p>
      <button onclick={start} class="btn-primary" disabled={!dataReady}>
        {dataReady ? 'Starta →' : 'Laddar …'}
      </button>
    </div>

  {:else if phase === 'transition'}
    {@const round = currentRound()}
    <div class="card">
      <p class="round-marker">Runda {roundIdx + 1} av 3 · {round.label}</p>
      <h2>{round.title}</h2>
      <p class="round-intro">{round.intro}</p>
      <button onclick={continueAfterTransition} class="btn-primary">
        Fortsätt →
      </button>
    </div>

  {:else if phase === 'q' && currentQuestion}
    {@const round = currentRound()}
    <div class="card">
      <div class="progress-bar">
        <div class="progress-fill" style:width={`${(totalAnswered() / totalQuestions()) * 100}%`}></div>
      </div>
      <p class="round-marker">
        Runda {roundIdx + 1} av 3 ·
        <strong>{round.label}</strong>
        ·
        fråga {currentRoundProgress() + 1} av {round.target}
      </p>

      <h3 class="statement">{currentQuestion.text}</h3>

      <div class="answer-row">
        <button class="ans ans-agree" onclick={() => answer(+1)}>
          <span class="ans-icon">✓</span>
          <span>Instämmer</span>
        </button>
        <button class="ans ans-neutral" onclick={() => answer(0)}>
          <span class="ans-icon">~</span>
          <span>Vet ej / hoppa</span>
        </button>
        <button class="ans ans-disagree" onclick={() => answer(-1)}>
          <span class="ans-icon">✗</span>
          <span>Instämmer inte</span>
        </button>
      </div>

      <p class="meta-row">
        <button onclick={reset} class="link">Börja om</button>
      </p>
    </div>

  {:else if phase === 'done' && result}
    <div class="card result-card">
      <h2>Din matchning</h2>
      <p class="lede-result">
        Tre axlar, tre resultat. Att olika partier dyker upp betyder inte
        att du är förvirrad – det betyder att de flesta partier inte är
        konsekventa över alla tre dimensioner.
      </p>

      <div class="three-axis">
        {#each ['what', 'why', 'how'] as axisKey}
          {@const r = result[axisKey]}
          {@const partyInfo = PARTY_INFO[r.top.party]}
          {@const axisCfg = ROUNDS.find(rr => rr.key === axisKey)}
          <div class="axis-row" data-axis={axisKey} style:--accent={partyInfo.color}>
            <div class="axis-header">
              <span class="axis-label">{axisCfg.label}</span>
              <span class="axis-explain">
                {axisKey === 'what' ? 'vilken politik partiet faktiskt drivit' :
                 axisKey === 'why'  ? 'vilka värden partiet hänvisar till' :
                                      'vilka medel partiet föredrar'}
              </span>
            </div>
            <div class="axis-result">
              <span class="party-pill" style:background={partyInfo.color}>
                {r.top.party}
              </span>
              <span class="party-name">{partyInfo.name}</span>
              <span class="party-sim">{formatPct(r.top.sim)}</span>
            </div>
          </div>
        {/each}
      </div>

      <details class="all-scores">
        <summary>Visa alla partier per axel</summary>
        <div class="scores-grid">
          {#each ['what', 'why', 'how'] as axisKey}
            {@const axisCfg = ROUNDS.find(rr => rr.key === axisKey)}
            <div class="scores-col">
              <h4>{axisCfg.label}</h4>
              {#each PARTIES.toSorted((a,b) => result[axisKey].all[b] - result[axisKey].all[a]) as p}
                <div class="score-row">
                  <span class="score-mark" style:background={PARTY_INFO[p].color}>{p}</span>
                  <span class="score-val">{formatPct(result[axisKey].all[p])}</span>
                </div>
              {/each}
            </div>
          {/each}
        </div>
      </details>

      <details class="all-scores">
        <summary>Visa dina svar</summary>
        <div class="answer-review">
          {#each ['what', 'why', 'how'] as axisKey}
            {@const axisCfg = ROUNDS.find(rr => rr.key === axisKey)}
            <div class="review-section">
              <h4>{axisCfg.label}</h4>
              <ol>
                {#each answersPerAxis[axisKey] as a}
                  <li>
                    <span class="ans-text">{a.question.text}</span>
                    <span class={`ans-tag ans-tag-${a.value === 1 ? 'agree' : a.value === -1 ? 'disagree' : 'neutral'}`}>
                      {a.value === 1 ? 'instämmer' : a.value === -1 ? 'instämmer inte' : 'vet ej'}
                    </span>
                  </li>
                {/each}
              </ol>
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
  .axis-list {
    list-style: none;
    padding: 0;
    margin: 1rem 0;
  }
  .axis-list li {
    padding: 0.5rem 0;
    border-bottom: 1px dashed #DDD;
  }
  .meta { font-size: 0.85rem; color: #666; }

  .progress-bar {
    background: #F0EEEA;
    border-radius: 4px;
    height: 6px;
    margin-bottom: 1.5rem;
    overflow: hidden;
  }
  .progress-fill {
    background: #2A4A7F;
    height: 100%;
    transition: width 0.3s ease;
  }
  .round-marker {
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    margin: 0 0 0.5rem;
  }
  .round-marker strong {
    color: #2A4A7F;
    font-weight: 700;
  }
  .round-intro {
    font-size: 1.05rem;
    color: #444;
    margin: 1rem 0 2rem;
  }
  .statement {
    font-size: 1.35rem;
    line-height: 1.4;
    color: #222;
    margin: 0.5rem 0 2rem;
    font-weight: 500;
  }
  .answer-row {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 0.75rem;
    margin: 1.5rem 0 1rem;
  }
  .ans {
    background: #F5F5F2;
    border: 1px solid #DDD;
    border-radius: 6px;
    padding: 1.25rem 0.75rem;
    cursor: pointer;
    font-family: inherit;
    color: inherit;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.95rem;
    font-weight: 500;
    transition: border-color 0.15s, background 0.15s, transform 0.05s;
  }
  .ans:hover { background: #FFF; }
  .ans:active { transform: translateY(1px); }
  .ans-agree:hover { border-color: #1B9E77; }
  .ans-disagree:hover { border-color: #D95F02; }
  .ans-neutral:hover { border-color: #999; }
  .ans-icon {
    font-size: 1.4rem;
    font-weight: 700;
    line-height: 1;
  }
  .ans-agree .ans-icon { color: #1B9E77; }
  .ans-disagree .ans-icon { color: #D95F02; }
  .ans-neutral .ans-icon { color: #888; }

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
  }
  .meta-row {
    display: flex;
    gap: 1rem;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #666;
    margin: 1rem 0 0;
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
    --axis-bg: #F5F5F2;
    padding: 1.5rem 1.25rem;
    background: var(--axis-bg);
    border-left: 4px solid var(--accent);
    border-radius: 6px;
  }
  .axis-row[data-axis="what"]    { --axis-bg: var(--vad-soft); }
  .axis-row[data-axis="why"]     { --axis-bg: var(--varfor-soft); }
  .axis-row[data-axis="how"]     { --axis-bg: var(--hur-soft); }
  .axis-row[data-axis="what"] .axis-label    { color: var(--vad); }
  .axis-row[data-axis="why"] .axis-label     { color: var(--varfor); }
  .axis-row[data-axis="how"] .axis-label     { color: var(--hur); }
  .axis-header {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    align-items: baseline;
    margin-bottom: 0.75rem;
  }
  .axis-label {
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    color: #444;
    font-weight: 700;
  }
  .axis-explain { font-size: 0.9rem; color: #666; }
  .axis-result {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
  }
  .party-pill {
    color: white;
    padding: 0.3rem 0.75rem;
    border-radius: 4px;
    font-weight: 700;
    font-size: 1.05rem;
  }
  .party-name { flex: 1; font-size: 0.95rem; color: #222; }
  .party-sim { font-weight: 600; font-size: 1.05rem; color: #444; }

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
    font-size: 0.8rem;
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

  .answer-review { margin-top: 1rem; }
  .review-section { margin-bottom: 1.5rem; }
  .review-section h4 {
    margin: 0 0 0.5rem;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    color: #666;
  }
  .review-section ol {
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .review-section li {
    padding: 0.5rem 0;
    border-bottom: 1px solid #EEE;
    font-size: 0.9rem;
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    align-items: baseline;
  }
  .ans-text { flex: 1; }
  .ans-tag {
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    white-space: nowrap;
  }
  .ans-tag-agree { background: #D4E8DC; color: #1B6E48; }
  .ans-tag-disagree { background: #F5DAC2; color: #A04A0A; }
  .ans-tag-neutral { background: #E8E8E8; color: #666; }

  @media (max-width: 540px) {
    .card { padding: 1.25rem; }
    .answer-row {
      grid-template-columns: 1fr;
    }
    .scores-grid {
      grid-template-columns: 1fr;
      gap: 1rem;
    }
  }
</style>

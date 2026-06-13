// Matching logic for the voter tool — pure functions, browser-side.
//
// We don't ship a probabilistic model; we ship cosine/distance arithmetic
// over precomputed party vectors. The "adaptive" feel is information-
// theoretic: each question is picked to maximise expected reduction in
// uncertainty about which party best matches.

const PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"];

// --- vector ops ---
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
function norm(a) { return Math.sqrt(dot(a, a)); }
function cosine(a, b) {
  const na = norm(a), nb = norm(b);
  return na && nb ? dot(a, b) / (na * nb) : 0;
}
function meanVec(vecs) {
  if (!vecs.length) return null;
  const out = new Float32Array(vecs[0].length);
  for (const v of vecs) for (let i = 0; i < v.length; i++) out[i] += v[i];
  for (let i = 0; i < out.length; i++) out[i] /= vecs.length;
  return out;
}

/**
 * Pick the next topic to ask about.
 * Strategy: choose the topic whose party stance scores have the highest
 * remaining variance, weighted by the current party posterior. Topics where
 * the current top-likely parties most disagree are the most informative.
 *
 * @param {Array} topics  topic objects sorted by what_spread desc
 * @param {Object} vectors  party stance data
 * @param {Set<number>} asked  topic ids already asked
 * @param {Object<string, number>} posterior  current party probabilities
 * @returns {number} topic_id
 */
export function pickNextTopic(topics, vectors, asked, posterior) {
  let best = null, bestScore = -1;
  for (const topic of topics) {
    if (asked.has(topic.id)) continue;
    // Variance of stance across parties, weighted by posterior.
    let mean = 0, totalW = 0;
    for (const p of PARTIES) {
      const s = vectors[p]?.[String(topic.id)]?.stance;
      if (s == null) continue;
      const w = posterior[p];
      mean += w * s;
      totalW += w;
    }
    if (!totalW) continue;
    mean /= totalW;
    let variance = 0;
    for (const p of PARTIES) {
      const s = vectors[p]?.[String(topic.id)]?.stance;
      if (s == null) continue;
      variance += posterior[p] * (s - mean) ** 2;
    }
    variance /= totalW;
    // Tie-break with the pre-computed what_spread.
    const score = variance + 0.001 * topic.what_spread;
    if (score > bestScore) {
      bestScore = score;
      best = topic.id;
    }
  }
  return best;
}

/**
 * For a chosen topic, pick 3 excerpts from 3 different parties whose stances
 * best span the disagreement axis.
 *
 * @returns {Array<{party: string, text: string, rubrik: string}>}
 */
export function pickExcerptsForTopic(topicId, vectors, excerpts) {
  const topicExcerpts = excerpts[String(topicId)] || {};
  const partiesWithExcerpts = Object.keys(topicExcerpts);
  if (partiesWithExcerpts.length === 0) return [];

  // Score each party with excerpts by stance + lexicon vector — pick 3 that
  // are maximally apart in stance space.
  const partyVectors = partiesWithExcerpts.map(p => {
    const cell = vectors[p]?.[String(topicId)];
    const s = cell?.stance ?? 0;
    return { party: p, stance: s };
  });
  partyVectors.sort((a, b) => a.stance - b.stance);

  const picked = [];
  if (partyVectors.length >= 3) {
    picked.push(partyVectors[0].party);                           // most negative
    picked.push(partyVectors[partyVectors.length - 1].party);     // most positive
    picked.push(partyVectors[Math.floor(partyVectors.length / 2)].party); // middle
  } else {
    for (const pv of partyVectors) picked.push(pv.party);
  }
  // Dedup.
  const seen = new Set();
  const unique = picked.filter(p => !seen.has(p) && seen.add(p));

  // For each picked party, sample one excerpt (random within first few for variety).
  return unique.map(p => {
    const candidates = topicExcerpts[p] || [];
    const choice = candidates[Math.floor(Math.random() * Math.min(candidates.length, 3))];
    return {
      party: p,
      text: choice?.text || "",
      rubrik: choice?.rubrik || "",
    };
  }).filter(e => e.text);
}

/**
 * Update posterior given the voter chose a party's excerpt.
 * Light Bayesian-style update: bump chosen party, decay others.
 */
export function updatePosterior(posterior, chosenParty, options) {
  const out = { ...posterior };
  // Strong bump for the chosen party.
  for (const p of PARTIES) {
    if (p === chosenParty) {
      out[p] = out[p] * 2.0;
    } else if (options.some(o => o.party === p)) {
      // Penalise the parties whose argument was visible but not chosen.
      out[p] = out[p] * 0.7;
    } else {
      // Untouched.
    }
  }
  // Renormalise.
  let total = 0;
  for (const p of PARTIES) total += out[p];
  for (const p of PARTIES) out[p] /= total;
  return out;
}

/**
 * Compute final three-axis match.
 *
 * WHAT — voter stance per topic = +1 if they chose an Ja-leaning party,
 *        weighted by that party's stance. Average across answers.
 * WHY  — average reasoning vector of the parties whose excerpts were chosen
 *        (because the voter actually agreed with that reasoning text).
 * HOW  — average lexicon vector across the chosen parties' (topic) cells.
 *
 * For each axis, compute per-party similarity and find the closest match.
 *
 * @param {Array<{topicId: number, party: string}>} answers
 */
export function computeMatch(answers, vectors, reasoning) {
  // === WHAT ===
  // For each topic asked, voter's stance proxy is the chosen party's stance.
  // Per party, similarity = 1 - mean(|voter_stance - their_stance|)/2.
  const what = {};
  for (const p of PARTIES) {
    const diffs = [];
    for (const a of answers) {
      const t = String(a.topicId);
      const voterStance = vectors[a.party]?.[t]?.stance;
      const partyStance = vectors[p]?.[t]?.stance;
      if (voterStance == null || partyStance == null) continue;
      diffs.push(Math.abs(voterStance - partyStance));
    }
    if (!diffs.length) { what[p] = 0; continue; }
    const mad = diffs.reduce((a, b) => a + b, 0) / diffs.length;
    what[p] = 1 - mad / 2;
  }

  // === WHY ===
  // Build a voter reasoning vector from the chosen parties' reasoning vectors
  // on the topics they were chosen for. Then cosine to each party's overall
  // reasoning vector.
  const partyMeanReasoning = {};
  for (const p of PARTIES) {
    const cells = reasoning[p] || {};
    const vecs = Object.values(cells).map(c => c.v);
    partyMeanReasoning[p] = meanVec(vecs);
  }
  // Voter's reasoning vector: mean of chosen-party-on-chosen-topic reasoning vectors.
  const voterReasoningParts = [];
  for (const a of answers) {
    const cell = reasoning[a.party]?.[String(a.topicId)];
    if (cell) voterReasoningParts.push(cell.v);
  }
  const voterReasoning = meanVec(voterReasoningParts);
  const why = {};
  for (const p of PARTIES) {
    const v = partyMeanReasoning[p];
    why[p] = voterReasoning && v ? cosine(voterReasoning, v) : 0;
  }

  // === HOW ===
  // Per party, mean of lexicon scores across topics. Z-score per dimension
  // across parties so that "everyone is universalist" doesn't dominate — we
  // only care about how parties *differ* on each axis.
  const DIMS = ["market_vs_regulation", "prevention_vs_punishment",
                "state_vs_local", "universal_vs_targeted"];
  function howVec(party) {
    const out = new Float32Array(DIMS.length);
    const counts = new Float32Array(DIMS.length);
    for (const t of Object.keys(vectors[party] || {})) {
      const cell = vectors[party][t];
      for (let i = 0; i < DIMS.length; i++) {
        if (cell[DIMS[i]] != null) {
          out[i] += cell[DIMS[i]];
          counts[i] += 1;
        }
      }
    }
    for (let i = 0; i < DIMS.length; i++) {
      out[i] = counts[i] ? out[i] / counts[i] : 0;
    }
    return out;
  }
  const rawHow = Object.fromEntries(PARTIES.map(p => [p, howVec(p)]));
  // Per-dimension z-score using the across-party mean and std.
  const dimMean = new Float32Array(DIMS.length);
  const dimStd = new Float32Array(DIMS.length);
  for (let i = 0; i < DIMS.length; i++) {
    let m = 0;
    for (const p of PARTIES) m += rawHow[p][i];
    m /= PARTIES.length;
    dimMean[i] = m;
    let v = 0;
    for (const p of PARTIES) v += (rawHow[p][i] - m) ** 2;
    dimStd[i] = Math.sqrt(v / PARTIES.length) || 1;
  }
  const partyHow = {};
  for (const p of PARTIES) {
    const z = new Float32Array(DIMS.length);
    for (let i = 0; i < DIMS.length; i++) {
      z[i] = (rawHow[p][i] - dimMean[i]) / dimStd[i];
    }
    partyHow[p] = z;
  }

  // Voter HOW = mean of chosen-party-on-topic lexicon vectors, z-scored
  // using the same per-dimension statistics.
  const voterHowParts = [];
  for (const a of answers) {
    const cell = vectors[a.party]?.[String(a.topicId)];
    if (!cell) continue;
    const v = new Float32Array(DIMS.length);
    let any = false;
    for (let i = 0; i < DIMS.length; i++) {
      if (cell[DIMS[i]] != null) {
        v[i] = (cell[DIMS[i]] - dimMean[i]) / dimStd[i];
        any = true;
      }
    }
    if (any) voterHowParts.push(v);
  }
  const voterHow = meanVec(voterHowParts);
  const how = {};
  if (!voterHow) {
    for (const p of PARTIES) how[p] = 0.5;
  } else {
    // Use cosine on z-scored vectors — now differences are surfaced.
    for (const p of PARTIES) {
      const sim = cosine(Array.from(voterHow), Array.from(partyHow[p]));
      how[p] = (sim + 1) / 2;  // map [-1, 1] to [0, 1]
    }
  }

  // Rank-normalise the why scores to [0,1] for display parity with WHAT/HOW.
  const whyVals = Object.values(why);
  const minW = Math.min(...whyVals);
  const maxW = Math.max(...whyVals);
  const whyN = {};
  for (const p of PARTIES) {
    whyN[p] = maxW > minW ? (why[p] - minW) / (maxW - minW) : 0.5;
  }

  function topMatch(scoreObj) {
    let best = PARTIES[0], val = -Infinity;
    for (const p of PARTIES) if (scoreObj[p] > val) { val = scoreObj[p]; best = p; }
    return { party: best, sim: val };
  }

  return {
    what:   { all: what,  top: topMatch(what) },
    why:    { all: whyN,  top: topMatch(whyN) },
    how:    { all: how,   top: topMatch(how) },
  };
}

/** Initial uniform posterior. */
export function initialPosterior() {
  const p = {};
  for (const party of PARTIES) p[party] = 1 / PARTIES.length;
  return p;
}

export { PARTIES };

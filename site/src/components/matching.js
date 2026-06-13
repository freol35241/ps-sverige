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
 * For a chosen topic, pick 3 excerpts from 3 different parties.
 *
 * Strategy: combine "top by current posterior" with "stance diversity".
 * - Early in the questionnaire (uniform posterior): show the 3 parties whose
 *   stances diverge most on this topic (high information gain).
 * - Later (concentrated posterior): show the 3 highest-posterior parties
 *   whose stances also differ enough to be discriminative.
 *
 * @returns {Array<{party: string, text: string, rubrik: string}>}
 */
export function pickExcerptsForTopic(topicId, vectors, excerpts, posterior) {
  const topicExcerpts = excerpts[String(topicId)] || {};
  const partiesWithExcerpts = Object.keys(topicExcerpts);
  if (partiesWithExcerpts.length < 2) return [];

  // Build candidate list with stance and current posterior.
  const candidates = partiesWithExcerpts
    .map(p => ({
      party: p,
      stance: vectors[p]?.[String(topicId)]?.stance ?? 0,
      post: posterior?.[p] ?? 1 / PARTIES.length,
    }))
    .filter(c => c.stance != null);

  if (candidates.length < 2) return [];

  // Measure how concentrated the posterior is. Uniform → entropy = log(8) ≈ 2.08.
  // The closer to uniform, the more we prioritise stance diversity.
  let entropy = 0;
  for (const c of candidates) {
    if (c.post > 0) entropy -= c.post * Math.log(c.post);
  }
  const maxEntropy = Math.log(PARTIES.length);
  const uniformity = entropy / maxEntropy; // 1 = uniform, ~0 = concentrated

  // Score each party for picking.
  // - postScore: rank-normalised current posterior
  // - diversityScore: how distinct its stance is from others (variance contribution)
  const stances = candidates.map(c => c.stance);
  const meanStance = stances.reduce((a, b) => a + b, 0) / stances.length;
  const scored = candidates.map(c => ({
    ...c,
    diversity: Math.abs(c.stance - meanStance),
  }));

  // Greedy selection of 3: at each step, pick the party that maximises
  // (combined score) while ensuring its stance is far enough from already-
  // picked parties to be a real choice.
  const picked = [];
  const remaining = [...scored];
  // First pick: highest combined.
  remaining.sort((a, b) =>
    (b.post + uniformity * b.diversity * 0.3) -
    (a.post + uniformity * a.diversity * 0.3));
  picked.push(remaining.shift());

  // Subsequent picks: must be diverse from already-picked AND high score.
  const MIN_STANCE_DIFF = 0.1;
  while (picked.length < 3 && remaining.length > 0) {
    remaining.sort((a, b) => {
      const aDist = Math.min(...picked.map(p => Math.abs(a.stance - p.stance)));
      const bDist = Math.min(...picked.map(p => Math.abs(b.stance - p.stance)));
      const aScore = a.post + 0.5 * aDist + uniformity * a.diversity * 0.3;
      const bScore = b.post + 0.5 * bDist + uniformity * b.diversity * 0.3;
      return bScore - aScore;
    });
    const next = remaining.shift();
    // Skip if too close to an already-picked party (so we don't show two
    // cabinet excerpts that read the same).
    const closest = Math.min(...picked.map(p => Math.abs(next.stance - p.stance)));
    if (closest < MIN_STANCE_DIFF && picked.length < remaining.length) {
      continue;
    }
    picked.push(next);
  }

  // Convert to excerpt objects, picking the first candidate excerpt per party.
  return picked.map(p => {
    const candidates = topicExcerpts[p.party] || [];
    const choice = candidates[Math.floor(Math.random() * Math.min(candidates.length, 3))];
    return {
      party: p.party,
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
 * WHAT — voter stance per topic = the chosen party's stance on that topic.
 * WHY  — voter reasoning vector = mean of chosen-party reasoning vectors
 *        across answered topics, compared by cosine to party reasoning means.
 * HOW  — voter lexicon vector = mean of chosen-party lexicon scores,
 *        z-scored per dimension and compared by cosine.
 *
 * @param {Array<{topicId: number, party: string}>} answers
 * @param {Object} vectors
 * @param {Object} reasoning
 * @param {Array} topics  needed for the per-axis topic-level justifications
 */
export function computeMatch(answers, vectors, reasoning, topics) {
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

  // === Per-axis topic-level justifications ===
  // For each axis and each party, score how strongly each topic answer
  // contributed to that party's match. Used to surface 1-3 example topics
  // explaining the top match.
  const topicLabels = topics ? Object.fromEntries(topics.map(t => [t.id, t.label])) : {};

  function topicEvidence(party, axisFn) {
    const items = [];
    for (const a of answers) {
      const score = axisFn(a, party);
      if (score == null) continue;
      items.push({ topicId: a.topicId, label: topicLabels[a.topicId] || `Ämne ${a.topicId}`, score });
    }
    items.sort((a, b) => b.score - a.score);
    return items.slice(0, 3);
  }

  // For WHAT: per topic, contribution = 1 - |voter stance - party stance| / 2
  const whatEvidence = {};
  for (const p of PARTIES) {
    whatEvidence[p] = topicEvidence(p, (a, party) => {
      const vs = vectors[a.party]?.[String(a.topicId)]?.stance;
      const ps = vectors[party]?.[String(a.topicId)]?.stance;
      if (vs == null || ps == null) return null;
      return 1 - Math.abs(vs - ps) / 2;
    });
  }

  // For WHY: per topic, cosine between voter's chosen-party reasoning vector
  // and this party's reasoning vector on that topic.
  const whyEvidence = {};
  for (const p of PARTIES) {
    whyEvidence[p] = topicEvidence(p, (a, party) => {
      const vCell = reasoning[a.party]?.[String(a.topicId)];
      const pCell = reasoning[party]?.[String(a.topicId)];
      if (!vCell || !pCell) return null;
      return (cosine(vCell.v, pCell.v) + 1) / 2;
    });
  }

  // Top match with WHY-based tie-break. Cabinet (L, KD, M) often ties on WHAT.
  function topMatch(scoreObj, tiebreakScoreObj) {
    const ranked = PARTIES.slice().sort((a, b) => {
      const diff = scoreObj[b] - scoreObj[a];
      if (Math.abs(diff) < 0.005 && tiebreakScoreObj) {
        return tiebreakScoreObj[b] - tiebreakScoreObj[a];
      }
      return diff;
    });
    return { party: ranked[0], sim: scoreObj[ranked[0]] };
  }

  return {
    what: { all: what,  top: topMatch(what, whyN), evidence: whatEvidence },
    why:  { all: whyN,  top: topMatch(whyN, what), evidence: whyEvidence },
    how:  { all: how,   top: topMatch(how, whyN),  evidence: whyEvidence },
  };
}

/** Initial uniform posterior. */
export function initialPosterior() {
  const p = {};
  for (const party of PARTIES) p[party] = 1 / PARTIES.length;
  return p;
}

export { PARTIES };

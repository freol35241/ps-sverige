// Three-axis matching for the voter tool.
//
// The tool runs three rounds:
//   1. WHAT — policy positions; voter answers map to stance values per topic,
//             match against each party's stance pattern.
//   2. WHY  — value statements; voter's reasoning vector is built from
//             SBERT-embedded statements (added when agree, subtracted when
//             disagree), compared to each party's mean reasoning vector by
//             cosine.
//   3. HOW  — mechanism statements; voter's HOW vector accumulates per
//             lexicon dimension, z-scored, compared to each party's HOW
//             vector by cosine.
//
// Each axis produces its own top-match party. Voter answers per question are
// {value: +1 | -1 | 0}: agree, disagree, neutral (skip).

const PARTIES = ["V", "S", "MP", "C", "L", "KD", "M", "SD"];
const DIMS = [
  "market_vs_regulation",
  "prevention_vs_punishment",
  "state_vs_local",
  "universal_vs_targeted",
];

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
function addInto(target, src, scale = 1) {
  for (let i = 0; i < target.length; i++) target[i] += src[i] * scale;
}
function meanVec(vecs) {
  if (!vecs.length) return null;
  const out = new Float32Array(vecs[0].length);
  for (const v of vecs) for (let i = 0; i < v.length; i++) out[i] += v[i];
  for (let i = 0; i < out.length; i++) out[i] /= vecs.length;
  return out;
}

// =============================================================================
// WHAT axis — policy stance matching.
// =============================================================================

/**
 * Compute WHAT match per party.
 * @param {Array<{question, value}>} answers  question objects with topic_id and agree_direction
 * @param {Object} vectors  per-party stance data
 * @returns {{all: Object<string, number>, top: {party, sim}, evidence: Object}}
 */
export function matchWhat(answers, vectors) {
  // Build voter stance per topic: voter[topic] = answer.value * agree_direction
  // answer.value is +1 (agree), -1 (disagree), 0 (neutral)
  const voterStance = {};
  for (const a of answers) {
    if (a.value === 0) continue;
    const v = a.value * a.question.agree_direction;
    // If voter answers multiple questions on the same topic, average them.
    if (voterStance[a.question.topic_id] === undefined) {
      voterStance[a.question.topic_id] = { sum: v, count: 1 };
    } else {
      voterStance[a.question.topic_id].sum += v;
      voterStance[a.question.topic_id].count += 1;
    }
  }
  const voterStanceAvg = {};
  for (const t in voterStance) {
    voterStanceAvg[t] = voterStance[t].sum / voterStance[t].count;
  }

  const all = {};
  const evidence = {};
  for (const p of PARTIES) {
    let totalDiff = 0;
    let n = 0;
    const matches = [];
    for (const t in voterStanceAvg) {
      const partyStance = vectors[p]?.[t]?.stance;
      if (partyStance == null) continue;
      const diff = Math.abs(voterStanceAvg[t] - partyStance);
      totalDiff += diff;
      n++;
      matches.push({
        topic_id: parseInt(t),
        agreement: 1 - diff / 2,  // 1 = full match, 0 = opposite
      });
    }
    all[p] = n > 0 ? 1 - (totalDiff / n) / 2 : 0;
    matches.sort((x, y) => y.agreement - x.agreement);
    evidence[p] = matches.slice(0, 3);
  }

  return {
    all,
    top: topMatch(all),
    evidence,
  };
}

// =============================================================================
// WHY axis — value-frame matching via SBERT embeddings.
// =============================================================================

/**
 * Compute WHY match per party.
 * @param {Array<{question, value}>} answers  question objects with embedding[]
 * @param {Object} reasoning  per-(party, topic) reasoning vector
 */
export function matchWhy(answers, reasoning) {
  // Build voter reasoning vector: sum of (value * embedding) for non-neutral answers.
  if (!answers.length) return { all: zeros(), top: nullTop(), evidence: emptyEvidence() };
  const dim = answers[0].question.embedding.length;
  const voterVec = new Float32Array(dim);
  let nNonZero = 0;
  for (const a of answers) {
    if (a.value === 0) continue;
    addInto(voterVec, a.question.embedding, a.value);
    nNonZero++;
  }
  if (nNonZero === 0 || norm(voterVec) === 0) {
    return { all: uniform(), top: nullTop(), evidence: emptyEvidence() };
  }

  // For each party, compute mean reasoning vector across topics where they have data,
  // then cosine to voter.
  const partyMeans = {};
  for (const p of PARTIES) {
    const cells = reasoning[p] || {};
    const vecs = Object.values(cells).map(c => c.v);
    partyMeans[p] = meanVec(vecs);
  }

  const all = {};
  for (const p of PARTIES) {
    if (!partyMeans[p]) { all[p] = 0; continue; }
    const sim = cosine(Array.from(voterVec), partyMeans[p]);
    all[p] = (sim + 1) / 2;  // map [-1, 1] to [0, 1]
  }

  // Evidence: which value statements drove the match for the top party.
  // Use cosine between each answer's embedding and the top party's mean vector.
  const evidence = {};
  for (const p of PARTIES) {
    if (!partyMeans[p]) { evidence[p] = []; continue; }
    const items = [];
    for (const a of answers) {
      if (a.value === 0) continue;
      const ce = cosine(a.question.embedding, partyMeans[p]);
      // Voter's agreement direction matches party's: positive contribution.
      items.push({
        question_id: a.question.id,
        text: a.question.text,
        contribution: a.value * ce,  // higher = stronger match
      });
    }
    items.sort((x, y) => y.contribution - x.contribution);
    evidence[p] = items.slice(0, 3);
  }

  return {
    all,
    top: topMatch(all),
    evidence,
  };
}

// =============================================================================
// HOW axis — mechanism-preference matching via lexicon dimensions.
// =============================================================================

/**
 * Compute HOW match per party.
 * @param {Array<{question, value}>} answers  question objects with dimension and agree_direction
 * @param {Object} vectors  per-(party, topic) lexicon scalars
 */
export function matchHow(answers, vectors) {
  // Build voter HOW vector: 4-dim signed accumulator.
  const voterHow = new Float32Array(DIMS.length);
  const voterHowCount = new Float32Array(DIMS.length);
  for (const a of answers) {
    if (a.value === 0) continue;
    const dimIdx = DIMS.indexOf(a.question.dimension);
    if (dimIdx < 0) continue;
    voterHow[dimIdx] += a.value * a.question.agree_direction;
    voterHowCount[dimIdx] += 1;
  }
  // Normalise per dim to [-1, +1].
  for (let i = 0; i < DIMS.length; i++) {
    if (voterHowCount[i] > 0) voterHow[i] /= voterHowCount[i];
  }

  // Compute each party's mean HOW vector (raw, then z-scored).
  function partyRawHow(party) {
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
  const rawHow = Object.fromEntries(PARTIES.map(p => [p, partyRawHow(p)]));

  // Per-dim z-score using across-party mean and std (matches v1 logic).
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

  const all = {};
  for (const p of PARTIES) {
    const sim = cosine(Array.from(voterHow), Array.from(partyHow[p]));
    all[p] = (sim + 1) / 2;
  }

  // Evidence: per dim, how aligned voter and party are.
  const evidence = {};
  for (const p of PARTIES) {
    const items = [];
    for (let i = 0; i < DIMS.length; i++) {
      if (voterHowCount[i] === 0) continue;
      const dist = Math.abs(voterHow[i] - Math.sign(partyHow[p][i]));
      items.push({
        dimension: DIMS[i],
        agreement: 1 - dist / 2,
        voter_value: voterHow[i],
      });
    }
    items.sort((x, y) => y.agreement - x.agreement);
    evidence[p] = items.slice(0, 3);
  }

  return {
    all,
    top: topMatch(all),
    evidence,
  };
}

// =============================================================================
// Adaptive picker — within an axis, pick next question by information gain.
// =============================================================================

/**
 * Pick the next question from a pool that would most reduce uncertainty.
 *
 * For WHAT and HOW: score each candidate by how much it would change the
 * party-similarity ranking given the current posterior. For WHY: pick the
 * question whose embedding is most orthogonal to the current voter vector
 * (covers new ground in value space).
 *
 * Pure heuristic — not a strict information-gain calculation — but it
 * produces sensible adaptive ordering.
 *
 * @param {Array} pool  remaining candidate questions
 * @param {Array} answers  answers so far on this axis
 * @param {string} axis  "what" | "why" | "how"
 * @returns {Object} chosen question
 */
export function pickNextQuestion(pool, answers, axis, vectors, reasoning) {
  if (pool.length === 0) return null;
  if (answers.length === 0) {
    // First question: pick by general informativeness (predefined order suffices).
    return pool[0];
  }

  if (axis === "what") {
    // For each candidate, compute how much it would discriminate between the
    // current top 3 parties. Higher discrimination = better next question.
    const currentMatch = matchWhat(answers, vectors);
    const ranked = PARTIES.slice().sort((a, b) => currentMatch.all[b] - currentMatch.all[a]);
    const topK = ranked.slice(0, 4);
    let best = null, bestScore = -1;
    for (const q of pool) {
      const stances = topK.map(p => {
        const s = vectors[p]?.[q.topic_id]?.stance;
        return s != null ? s * q.agree_direction : 0;
      });
      // Discrimination = spread among top-K stances.
      const min = Math.min(...stances);
      const max = Math.max(...stances);
      const score = max - min;
      if (score > bestScore) {
        bestScore = score;
        best = q;
      }
    }
    return best || pool[0];
  }

  if (axis === "why") {
    // Pick the question whose embedding is most orthogonal to the current
    // voter reasoning vector — covers new ground.
    if (answers.length === 0) return pool[0];
    const dim = answers[0].question.embedding.length;
    const voterVec = new Float32Array(dim);
    for (const a of answers) {
      if (a.value === 0) continue;
      addInto(voterVec, a.question.embedding, a.value);
    }
    const voterArr = Array.from(voterVec);
    if (norm(voterVec) === 0) return pool[0];

    let best = null, bestScore = Infinity;
    for (const q of pool) {
      const c = Math.abs(cosine(voterArr, q.embedding));
      if (c < bestScore) {
        bestScore = c;
        best = q;
      }
    }
    return best || pool[0];
  }

  if (axis === "how") {
    // Prefer dimensions not yet covered.
    const seenDims = new Set(answers.filter(a => a.value !== 0).map(a => a.question.dimension));
    const uncovered = pool.filter(q => !seenDims.has(q.dimension));
    if (uncovered.length > 0) return uncovered[0];
    return pool[0];
  }

  return pool[0];
}

// =============================================================================
// utilities
// =============================================================================

function topMatch(scoreObj) {
  const ranked = PARTIES.slice().sort((a, b) => scoreObj[b] - scoreObj[a]);
  return { party: ranked[0], sim: scoreObj[ranked[0]] };
}

function zeros() {
  const o = {};
  for (const p of PARTIES) o[p] = 0;
  return o;
}
function uniform() {
  const o = {};
  for (const p of PARTIES) o[p] = 0.5;
  return o;
}
function nullTop() { return { party: PARTIES[0], sim: 0 }; }
function emptyEvidence() {
  const o = {};
  for (const p of PARTIES) o[p] = [];
  return o;
}

export { PARTIES, DIMS };

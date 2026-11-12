// pumpfun-top5-forecast-01 entry: a momentum-aware Monte Carlo forecaster.
//
// Idea (written from the competition's own rules, not from the starter):
//  - The house baseline bets the last mark's order holds. In the practice
//    rehearsal that bet is wrong most of the time: of the last mark's top 5,
//    only one coin (Google) was still top 5 at the last answer read, and two
//    of the five were born after the history ended. Pump.fun's top 5 churns
//    because new coins arrive and pump past the leaders within an hour.
//  - So this forecaster models each coin's future log market cap as a random
//    walk with a drift taken from its own recent velocity (exponentially
//    weighted over the last hour, capped, and decaying with a 1.5 h time
//    constant), plus a wide honest uncertainty that grows with the horizon
//    to each answer read. Coins first seen at the last mark get a positive
//    new-coin drift prior with extra variance: they are the ones that can
//    go from rank #22 to #1 in an hour (Max did exactly that in the
//    rehearsal), and they are also the ones that can die.
//  - It then runs a seeded Monte Carlo over the seven answer reads from
//    task.json, ranks the simulated caps, and turns the tallies into
//    place probabilities and size-bin probabilities. The forecast is one
//    object scored at every read, so the tallies are pooled over the reads.
//  - It deliberately leaves part of every place unnamed (more at the lower
//    places, where newcomers most often land): an unnamed share is the only
//    probability a coin born after the cut can ever score.
//  - Every time is read from task.json (reads, history window); nothing
//    about the practice or the final is hard-coded. The run is fully
//    deterministic: one fixed seed, no Date.now, no Math.random.
//
// Run: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
// (the grader runs it with /input and /output instead.)

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Coins taken by last-mark rank, before velocity / new-coin additions. */
const TOP_BY_CAP = 40;
/** Extra coins taken by recent velocity. */
const TOP_BY_VELOCITY = 15;
/** Rank cutoff for a coin first seen in the last two marks to join. */
const NEW_RANK_CUTOFF = 100;
/** Hard cap on the candidate set. */
const MAX_CANDIDATES = 60;
/** A coin's recent velocity, log10(cap in SOL) per hour, capped. */
const VEL_CAP = 1.5;
/** Momentum decay time constant, hours. */
const TAU_H = 1.5;
/** Drift prior, log10 per hour, for a coin first seen at the last mark. */
const NEW_DRIFT = 0.2;
/** Monte Carlo samples per answer read. */
const SAMPLES = 8000;
/** Fixed seed: the run is byte-for-byte deterministic. */
const SEED = 20261112;
/** Named share of each place (the rest is left unnamed, for newcomers). */
const NAMED_CAP = [0, 0.7, 0.65, 0.6, 0.55, 0.55];
/** Named share of each place's size distribution. */
const SIZE_NAMED = 0.95;
/** Smallest probability kept (rounded down to 6 decimals, like the starter). */
const MICRO = 1_000_000;

interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue: string;
  created_at: string;
  complete: boolean;
  mayhem: boolean;
}

interface Mark {
  at: string;
  ok: boolean;
  rows: Row[] | null;
}

interface Task {
  places: number;
  output: string;
  reads: string[];
  size: { edges: number[] };
}

/** log10 of a cap in SOL, as the scorer takes it: log10(lamports) - 9. */
const sizeOf = (lamports: string | number): number => Math.log10(Number(BigInt(lamports))) - 9;

/** mulberry32: a small seeded PRNG (deterministic across runs and machines). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A seeded Gaussian source (Box-Muller, spare cached, call order fixed). */
function gaussianSource(seed: number): () => number {
  const rand = mulberry32(seed);
  let spare: number | null = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0; let v = 0; let s = 0;
    do { u = rand() * 2 - 1; v = rand() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    const m = Math.sqrt(-2 * Math.log(s) / s);
    spare = v * m;
    return u * m;
  };
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8').split('\n').filter((l) => l !== '');
const marks = lines.map((l) => JSON.parse(l) as Mark).filter((m) => m.ok === true && Array.isArray(m.rows));
if (marks.length === 0) throw new Error('the history holds no ok mark');
const lastMark = marks[marks.length - 1]!;
const lastAt = new Date(lastMark.at).getTime();
const lastRows = [...lastMark.rows!].sort((a, b) => a.rank - b.rank);
if (lastRows.length < task.places) throw new Error(`the last mark ranks ${lastRows.length} coins, fewer than the ${task.places} places`);

// --- Build each coin's trajectory: log-cap at every mark it appears in. ---
const traj = new Map<string, Array<{ t: number; size: number; rank: number; venue: string; created: string }>>();
const firstSeen = new Map<string, number>();
for (let i = 0; i < marks.length; i++) {
  const mk = marks[i]!;
  const t = new Date(mk.at).getTime();
  for (const r of mk.rows!) {
    if (!traj.has(r.mint)) { traj.set(r.mint, []); firstSeen.set(r.mint, i); }
    traj.get(r.mint)!.push({ t, size: sizeOf(r.market_cap_lamports), rank: r.rank, venue: r.venue, created: r.created_at });
  }
}

// --- Recent velocity: exponentially weighted over the last hour (6 marks). ---
const INTERVAL_W = [1, 0.8, 0.6, 0.4, 0.3, 0.2];
function velocity(mint: string): number | null {
  const tr = traj.get(mint)!;
  let num = 0; let den = 0;
  for (let k = 0; k < INTERVAL_W.length && tr.length - 1 - k >= 1; k++) {
    const a = tr[tr.length - 1 - k]!;
    const b = tr[tr.length - 2 - k]!;
    const dtH = (a.t - b.t) / 3_600_000;
    if (dtH <= 0) continue;
    const w = INTERVAL_W[k]!;
    num += w * ((a.size - b.size) / dtH);
    den += w;
  }
  if (den > 0) return Math.max(-VEL_CAP, Math.min(VEL_CAP, num / den));
  if (tr.length >= 2) {
    const a = tr[tr.length - 1]!; const b = tr[0]!;
    const dtH = (a.t - b.t) / 3_600_000;
    if (dtH > 0) return Math.max(-VEL_CAP, Math.min(VEL_CAP, (a.size - b.size) / dtH));
  }
  return null;
}

// --- Candidate set: top by cap, top by velocity, new coins near the top. ---
const cand = new Map<string, { size0: number; vel: number; isNew: boolean; created: string; venue: string }>();
for (const r of lastRows.slice(0, TOP_BY_CAP)) {
  const v = velocity(r.mint);
  cand.set(r.mint, { size0: sizeOf(r.market_cap_lamports), vel: v ?? 0, isNew: firstSeen.get(r.mint) === marks.length - 1, created: r.created_at, venue: r.venue });
}
const byVel = lastRows
  .map((r) => ({ mint: r.mint, v: velocity(r.mint) }))
  .filter((x) => x.v !== null && x.v! > 0)
  .sort((a, b) => b.v! - a.v!)
  .slice(0, TOP_BY_VELOCITY);
for (const { mint, v } of byVel) {
  if (!cand.has(mint)) {
    const r = lastRows.find((x) => x.mint === mint)!;
    cand.set(mint, { size0: sizeOf(r.market_cap_lamports), vel: v!, isNew: firstSeen.get(mint) === marks.length - 1, created: r.created_at, venue: r.venue });
  }
}
for (const r of lastRows) {
  if (r.rank > NEW_RANK_CUTOFF) continue;
  if (firstSeen.get(r.mint)! >= marks.length - 2 && !cand.has(r.mint)) {
    cand.set(r.mint, { size0: sizeOf(r.market_cap_lamports), vel: NEW_DRIFT, isNew: true, created: r.created_at, venue: r.venue });
  }
}
let coins = [...cand.entries()].map(([mint, c]) => ({ mint, ...c }));
// New coins first seen at the last mark take the new-coin drift prior.
for (const c of coins) if (c.isNew && velocity(c.mint) === null) c.vel = NEW_DRIFT;
coins.sort((a, b) => b.size0 - a.size0);
coins = coins.slice(0, MAX_CANDIDATES);

// --- Monte Carlo over the answer reads: one forecast, scored at each. ---
const reads = task.reads.map((r) => new Date(r).getTime());
const edges = task.size.edges;
const nPlaces = task.places;
const nBins = edges.length + 1; // 0..81
const gauss = gaussianSource(SEED);
const placeCounts: number[][] = Array.from({ length: nPlaces + 1 }, () => new Array(coins.length).fill(0));
const sizeCounts: number[][] = Array.from({ length: nPlaces + 1 }, () => new Array(nBins).fill(0));
let totalSamples = 0;

function binOf(size: number): number {
  if (size < edges[0]!) return 0;
  for (let i = 0; i < edges.length; i++) if (size < edges[i]!) return i + 1;
  return edges.length;
}

for (const readAt of reads) {
  const h = Math.max(0, (readAt - lastAt) / 3_600_000); // hours after the last mark
  // Per-coin forecast distribution at this read.
  const mu: number[] = coins.map((c) => {
    const gain = c.vel * TAU_H * (1 - Math.exp(-h / TAU_H));
    return c.size0 + gain;
  });
  const sigma: number[] = coins.map((c, i) => {
    const gain = mu[i]! - c.size0;
    const extra = c.isNew ? 0.16 : 0;
    return Math.sqrt(0.09 + 0.12 * h + 0.0625 * gain * gain + extra);
  });
  for (let s = 0; s < SAMPLES; s++) {
    const caps = coins.map((_, i) => mu[i]! + sigma[i]! * gauss());
    const order = caps.map((cap, i) => i).sort((a, b) => {
      const d = caps[b]! - caps[a]!;
      if (d !== 0) return d;
      return coins[a]!.mint < coins[b]!.mint ? -1 : 1;
    });
    for (let k = 0; k < nPlaces; k++) {
      placeCounts[k + 1]![order[k]!]! += 1;
      sizeCounts[k + 1]![binOf(caps[order[k]!]!)]! += 1;
    }
  }
  totalSamples += SAMPLES;
}

// --- Turn tallies into the forecast. ---
const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (let place = 1; place <= nPlaces; place++) {
  const cap = NAMED_CAP[place] ?? 0.6;
  const raw = coins.map((c, i) => ({ mint: c.mint, p: placeCounts[place]![i]! / totalSamples }));
  const sum = raw.reduce((a, r) => a + r.p, 0);
  const scale = sum > 0 ? cap / sum : 0;
  for (const r of raw) {
    const p = Math.floor(r.p * scale * MICRO) / MICRO;
    if (p > 0) probabilities.push({ mint: r.mint, place, p });
  }
}
const sizes: Array<{ place: number; bin: number; p: number }> = [];
for (let place = 1; place <= nPlaces; place++) {
  const raw = sizeCounts[place]!.map((n, bin) => ({ bin, p: n / totalSamples }));
  const sum = raw.reduce((a, r) => a + r.p, 0);
  const scale = sum > 0 ? SIZE_NAMED / sum : 0;
  for (const r of raw) {
    const p = Math.floor(r.p * scale * MICRO) / MICRO;
    if (p > 0) sizes.push({ place, bin: r.bin, p });
  }
}
// top5: the most likely coin at each place in turn, distinct.
const top5: string[] = [];
const used = new Set<string>();
for (let place = 1; place <= nPlaces; place++) {
  const ranked = coins
    .map((c, i) => ({ mint: c.mint, p: placeCounts[place]![i]! / totalSamples }))
    .filter((x) => !used.has(x.mint))
    .sort((a, b) => b.p - a.p);
  const pick = ranked[0]!;
  top5.push(pick.mint);
  used.add(pick.mint);
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const out = `${JSON.stringify({ probabilities, top5, sizes })}\n`;
writeFileSync(join(OUTPUT_DIR, task.output), out);

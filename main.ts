// werewolf-agent-01 entry: a deterministic, model-free Werewolf player.
//
// No model calls, no API keys, no network beyond the env. Every decision is a
// pure function of the game's view (as /observe serves it) and MAKEFUN_TRIAL_SEED,
// so a trial's live pass and its replay play the same games action for action.
// No wall clock, no Math.random: tie-breaks draw from a SHA-256 seeded PRNG.
//
// Strategy, in short:
//   * The view's whole public log is parsed into evidence: revealed roles,
//     every vote, every speech (claims, accusations, defences), every night
//     death. A suspicion score per living seat drives the day vote.
//   * Seer: inspect the most suspicious seat each night; reveal a found wolf
//     at once, loudly, with the check in the first 120 characters (the house
//     prompt clips older statements to 120 chars). Hide villager checks.
//   * Doctor: protect a credible revealed seer first, else the villager the
//     wolves most want dead; never repeat last night's target, never self.
//   * Villager: vote the most suspicious seat; join a forming plurality rather
//     than split the vote; abstain only when every candidate looks village.
//   * Werewolf: kill the seer's throat (a credible seer claimant first, then
//     accurate accusers); copy the partner's night vote when they named
//     someone, so the kill lands; by day, bandwagon the villager the table
//     already suspects and never vote the partner.
//   * Speech is template-driven and deterministic: the claim first, one
//     reason, names and seats filled in from the view.
//
// The run loop is starter/main.ts's shape (POST /observe, answer every pending
// decision, exit once "done"), with the model call replaced by decide().

import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';

export const CONCURRENCY = 10;

type Seat = number;
type Role = 'werewolf' | 'seer' | 'doctor' | 'villager';
type DecisionKind = 'speak' | 'vote' | 'kill' | 'protect' | 'inspect';

interface LogEntry {
  seq: number;
  day: number;
  kind: string;
  died?: Seat | null;
  role?: Role;
  seat?: Seat;
  target?: Seat | null;
  text?: string;
  round?: number;
  order?: Seat[];
  tally?: { seat: Seat; votes: number }[];
  winner?: string | null;
  reason?: string;
  winners?: Seat[];
  [k: string]: unknown;
}

interface PlayerInfo {
  seat: Seat;
  name: string;
  alive: boolean;
  role: Role | null;
}

interface View {
  day: number;
  phase: string;
  you: {
    seat: Seat;
    name: string;
    role: Role;
    alive: boolean;
    partner: { seat: Seat; name: string; alive: boolean } | null;
  };
  players: PlayerInfo[];
  seerResults: { night: number; seat: Seat; werewolf: boolean }[] | null;
  lastProtect: { night: number; seat: Seat | null } | null;
  wolfVotes: { night: number; seat: Seat; target: Seat | null }[] | null;
  log: LogEntry[];
  outcome: { winner: string | null; reason: string; day: number; winners: Seat[] } | null;
  [k: string]: unknown;
}

export interface Pending {
  game: number;
  decision: DecisionKind;
  options?: Seat[];
  max_chars?: number;
  view: View;
}

interface Observed {
  games: number;
  finished: number;
  done: boolean;
  pending: Pending[];
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

/** A statement folded to one line the env's way, cut to max_chars. */
export function foldStatement(text: string): string {
  return text
    .replace(/[\t\n\v\f\r\u0085\u2028\u2029]+/g, ' ')
    .replace(/\p{Cc}/gu, '')
    .trim();
}

export function cutStatement(text: string, max: number): string {
  const folded = foldStatement(text);
  if ([...folded].length <= max) return folded;
  return [...folded].slice(0, max).join('').trim();
}

/** mulberry32 seeded from SHA-256(seed || game || salt): deterministic. */
export function rngFor(seed: string, game: number, salt: string): () => number {
  const h = createHash('sha256').update(`${seed}\0${game}\0${salt}`).digest();
  let s = (h.readUInt32LE(0) ^ h.readUInt32LE(4) ^ h.readUInt32LE(8) ^ h.readUInt32LE(12)) >>> 0;
  return () => {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Pick the max by score; ties broken by the seeded rng (lowest drawn wins). */
function argmax<T>(items: readonly T[], score: (x: T) => number, rng: () => number): T | null {
  let best: T | null = null;
  let bestScore = -Infinity;
  let bestDraw = Infinity;
  for (const it of items) {
    const sc = score(it);
    const draw = rng();
    if (sc > bestScore || (sc === bestScore && draw < bestDraw)) {
      best = it;
      bestScore = sc;
      bestDraw = draw;
    }
  }
  return best;
}

// ---------------------------------------------------------------- analysis

interface SpeechParse {
  seerClaim: boolean;
  checkedSeat: number | null;
  checkedWolf: boolean | null;
  accused: Seat[];
  defended: Seat[];
  doctorClaim: boolean;
}

const SEAT_RE = /seat\s+(\d+)/gi;
const WOLF_WORD = /\b(werewolf|wolf|wolves)\b/i;
const SUS_WORD = /\b(suspicious|sus|shady|wolfy|guilty|lying|liar)\b/i;
const VOTE_WORD = /\b(vote|exile|lynch)\b/i;
const TRUST_WORD = /\b(trust|believe|innocent|village|clear)\b/i;
const DEFEND_WORD = /\b(not convinced|don'?t buy|defend|wrong about)\b/i;

export function parseSpeech(text: string): SpeechParse {
  const seerClaim = /\bseer\b/i.test(text);
  const doctorClaim = /\bdoctor\b/i.test(text);
  let checkedSeat: number | null = null;
  let checkedWolf: boolean | null = null;
  const check = text.match(/check(?:ed|ing|s)?[^0-9]{0,80}?seat\s+(\d+)/i) ?? text.match(/seat\s+(\d+)[^.]{0,60}?check/i);
  if (check !== null) {
    checkedSeat = Number(check[1]);
    const after = text.slice(text.indexOf(check[0]));
    if (/\bnot\b[^.]{0,30}?\b(werewolf|wolf)\b/i.test(after) || /\b(villager|innocent|not a wolf)\b/i.test(after)) checkedWolf = false;
    else if (WOLF_WORD.test(after)) checkedWolf = true;
  }
  const accused: Seat[] = [];
  const defended: Seat[] = [];
  for (const clause of text.split(/[,.;:!?]/)) {
    const seats = [...clause.matchAll(SEAT_RE)].map((m) => Number(m[1]));
    if (seats.length === 0) continue;
    if (WOLF_WORD.test(clause) || SUS_WORD.test(clause) || VOTE_WORD.test(clause)) {
      for (const s of seats) if (!accused.includes(s)) accused.push(s);
    }
    if (TRUST_WORD.test(clause) || DEFEND_WORD.test(clause)) {
      for (const s of seats) if (!defended.includes(s)) defended.push(s);
    }
  }
  return { seerClaim, checkedSeat, checkedWolf, accused, defended, doctorClaim };
}

export interface Analysis {
  me: Seat;
  role: Role;
  alive: boolean;
  partner: Seat | null;
  day: number;
  living: Seat[];
  nameOf: Map<Seat, string>;
  revealed: Map<Seat, Role>;
  knownWolf: Set<Seat>;
  knownVillage: Set<Seat>;
  suspicion: Map<Seat, number>;
  heat: Map<Seat, number>;
  threat: Map<Seat, number>;
  speechCount: Map<Seat, number>;
  claimedSeer: Map<Seat, { target: Seat | null; saidWolf: boolean | null }>;
  claimedDoctor: Set<Seat>;
  credibleSeer: Seat | null;
  myChecks: { seat: Seat; werewolf: boolean }[];
  votesToday: { seat: Seat; target: Seat | null }[];
  tallyToday: Map<Seat, number>;
  mySpeechesToday: number;
  lastProtectSeat: Seat | null;
  partnerVoteTonight: Seat | null;
  partnerAbstainedTonight: boolean;
}

export function analyze(view: View): Analysis {
  const me = view.you.seat;
  const role = view.you.role;
  const nameOf = new Map<Seat, string>();
  const revealed = new Map<Seat, Role>();
  const living: Seat[] = [];
  for (const p of view.players) {
    nameOf.set(p.seat, p.name);
    if (p.role !== null) revealed.set(p.seat, p.role);
    if (p.alive) living.push(p.seat);
  }
  for (const e of view.log) {
    if (e.kind === 'dawn' && e.died != null && e.role !== undefined) revealed.set(e.died, e.role as Role);
    if (e.kind === 'exile' && e.seat != null && e.role !== undefined) revealed.set(e.seat, e.role as Role);
  }
  const knownWolf = new Set<Seat>();
  const knownVillage = new Set<Seat>();
  for (const p of view.players) {
    if (p.role === 'werewolf') knownWolf.add(p.seat);
    else if (p.role !== null) knownVillage.add(p.seat);
  }
  const myChecks = (view.seerResults ?? []).filter((r) => r.werewolf !== null);
  for (const r of myChecks) (r.werewolf ? knownWolf : knownVillage).add(r.seat);

  const suspicion = new Map<Seat, number>();
  const heat = new Map<Seat, number>();
  const threat = new Map<Seat, number>();
  const speechCount = new Map<Seat, number>();
  for (const s of living) {
    if (s === me) continue;
    suspicion.set(s, 0);
    heat.set(s, 0);
    threat.set(s, 0);
    speechCount.set(s, 0);
  }
  const bump = (m: Map<Seat, number>, s: Seat, n: number) => {
    if (m.has(s)) m.set(s, (m.get(s) ?? 0) + n);
  };

  const claimedSeer = new Map<Seat, { target: Seat | null; saidWolf: boolean | null }>();
  const claimedDoctor = new Set<Seat>();
  const accusedByDead: { accuser: Seat; accused: Seat }[] = [];
  const votesByDay = new Map<number, { seat: Seat; target: Seat | null }[]>();
  const votesToday: { seat: Seat; target: Seat | null }[] = [];
  let mySpeechesToday = 0;

  for (const e of view.log) {
    if (e.kind === 'speech' && typeof e.text === 'string' && e.seat !== undefined) {
      const s = e.seat as Seat;
      bump(speechCount, s, 1);
      if (s === me && e.day === view.day) mySpeechesToday += 1;
      const parsed = parseSpeech(e.text);
      if (parsed.seerClaim) claimedSeer.set(s, { target: parsed.checkedSeat, saidWolf: parsed.checkedWolf });
      if (parsed.doctorClaim) claimedDoctor.add(s);
      for (const t of parsed.accused) {
        bump(heat, t, 1);
        if (revealed.get(t) === 'werewolf') bump(suspicion, s, -2);
        else if (revealed.has(t) && revealed.get(t) !== 'werewolf') bump(suspicion, s, 2);
        if (!revealed.get(s) && !revealed.has(s)) {
          // Accusations made by a player later night-killed are gold: wolves
          // silence the people who suspect them. Resolved after the loop.
          accusedByDead.push({ accuser: s, accused: t });
        }
      }
      for (const t of parsed.defended) {
        if (revealed.get(t) === 'werewolf') bump(suspicion, s, 2);
        else if (revealed.has(t) && revealed.get(t) !== 'werewolf') bump(suspicion, s, -1);
      }
    } else if (e.kind === 'vote' && e.seat !== undefined) {
      const v = { seat: e.seat as Seat, target: (e.target ?? null) as Seat | null };
      const list = votesByDay.get(e.day) ?? [];
      list.push(v);
      votesByDay.set(e.day, list);
      if (e.day === view.day) {
        votesToday.push(v);
        if (v.target !== null) bump(heat, v.target, 2);
      }
      if (v.target !== null) {
        const rt = revealed.get(v.target);
        if (rt === 'werewolf') bump(suspicion, v.seat, -3);
        else if (rt !== undefined) bump(suspicion, v.seat, 3);
      }
    }
  }
  // The night-killed accuse from beyond the grave: if the accuser was killed
  // at night (revealed non-wolf by a dawn entry), their accusations count.
  const nightKilled = new Set<Seat>();
  for (const e of view.log) {
    if (e.kind === 'dawn' && e.died != null && revealed.get(e.died) !== 'werewolf') nightKilled.add(e.died);
  }
  for (const a of accusedByDead) {
    if (nightKilled.has(a.accuser)) bump(suspicion, a.accused, 2);
  }
  // Wolves vote together: sharing a target with a known wolf is a tell.
  for (const [, votes] of votesByDay) {
    for (const v of votes) {
      if (v.target === null || !knownWolf.has(v.seat)) continue;
      for (const w of votes) {
        if (w.seat !== v.seat && w.target === v.target) bump(suspicion, w.seat, 1);
      }
    }
  }
  // Seer claims: a claim whose check proved right is credible; a claim whose
  // check proved wrong, or one made while I am the seer, is a wolf's lie.
  let credibleSeer: Seat | null = null;
  for (const [s, c] of claimedSeer) {
    if (s === me) continue;
    if (c.target !== null && c.saidWolf === true && revealed.get(c.target) === 'werewolf') {
      bump(suspicion, s, -4);
      bump(suspicion, c.target, 0);
      if (credibleSeer === null) credibleSeer = s;
    } else if (c.target !== null && c.saidWolf === true && revealed.has(c.target) && revealed.get(c.target) !== 'werewolf') {
      bump(suspicion, s, 4);
    } else if (c.target !== null && c.saidWolf === false && revealed.get(c.target) === 'werewolf') {
      bump(suspicion, s, 4);
    }
    if (role === 'seer') bump(suspicion, s, 5);
    if (c.target !== null && c.saidWolf === true && !knownVillage.has(c.target)) bump(suspicion, c.target, 3);
  }
  // A claimant naming a wolf the village then exiled is the real seer until
  // proven otherwise; protect/follow them.
  if (credibleSeer === null) {
    for (const [s, c] of claimedSeer) {
      if (s !== me && c.target !== null && c.saidWolf === true && !knownVillage.has(c.target) && !revealed.has(c.target)) {
        credibleSeer = s;
        break;
      }
    }
  }
  // Threat: how much the wolves want a seat dead (for the doctor's protect and
  // the wolves' kill). Seer claimants, accurate accusers, vocal players.
  for (const s of living) {
    if (s === me) continue;
    if (claimedSeer.has(s)) bump(threat, s, 6);
    if (claimedDoctor.has(s)) bump(threat, s, 4);
    if (s === credibleSeer) bump(threat, s, 4);
    bump(threat, s, Math.min(speechCount.get(s) ?? 0, 4));
    const votedWolf = [...votesByDay.values()].flat().some((v) => v.seat === s && v.target !== null && revealed.get(v.target) === 'werewolf');
    if (votedWolf) bump(threat, s, 2);
  }

  const tallyToday = new Map<Seat, number>();
  for (const v of votesToday) {
    if (v.target !== null) tallyToday.set(v.target, (tallyToday.get(v.target) ?? 0) + 1);
  }

  let partnerVoteTonight: Seat | null = null;
  let partnerAbstainedTonight = false;
  const partner = view.you.partner?.seat ?? null;
  if (partner !== null && view.wolfVotes) {
    for (const v of view.wolfVotes) {
      if (v.night === view.day && v.seat === partner) {
        if (v.target === null) partnerAbstainedTonight = true;
        else partnerVoteTonight = v.target;
      }
    }
  }
  const lastProtectSeat = view.lastProtect?.seat ?? null;

  return {
    me,
    role,
    alive: view.you.alive,
    partner,
    day: view.day,
    living,
    nameOf,
    revealed,
    knownWolf,
    knownVillage,
    suspicion,
    heat,
    threat,
    speechCount,
    claimedSeer,
    claimedDoctor,
    credibleSeer,
    myChecks,
    votesToday,
    tallyToday,
    mySpeechesToday,
    lastProtectSeat,
    partnerVoteTonight,
    partnerAbstainedTonight,
  };
}

const who = (a: Analysis, s: Seat): string => `seat ${s} (${a.nameOf.get(s) ?? '?'})`;
const sus = (a: Analysis, s: Seat): number => a.suspicion.get(s) ?? 0;

// ---------------------------------------------------------------- decisions

function decideInspect(p: Pending, a: Analysis, rng: () => number): Record<string, unknown> {
  const done = new Set(a.myChecks.map((r) => r.seat));
  const cands = a.living.filter((s) => s !== a.me && !done.has(s) && !a.knownWolf.has(s) && !a.knownVillage.has(s));
  const pool = cands.length > 0 ? cands : a.living.filter((s) => s !== a.me && !done.has(s));
  const pick = argmax(pool, (s) => sus(a, s) + (a.claimedSeer.has(s) ? 2 : 0), rng);
  return { game: p.game, decision: 'inspect', target: pick ?? null };
}

function decideProtect(p: Pending, a: Analysis, rng: () => number): Record<string, unknown> {
  const options = (p.options ?? []).filter((s) => s !== a.lastProtectSeat);
  const pool = options.filter((s) => !a.knownWolf.has(s));
  const use = pool.length > 0 ? pool : options;
  // A credible seer is the wolves' next victim: guard them first.
  if (a.credibleSeer !== null && use.includes(a.credibleSeer)) {
    return { game: p.game, decision: 'protect', target: a.credibleSeer };
  }
  const pick = argmax(use, (s) => (a.threat.get(s) ?? 0) + (a.claimedSeer.has(s) ? 5 : 0), rng);
  return { game: p.game, decision: 'protect', target: pick ?? null };
}

function decideKill(p: Pending, a: Analysis, rng: () => number): Record<string, unknown> {
  const options = p.options ?? [];
  // The partner already named someone: agree, and the kill lands.
  if (a.partnerVoteTonight !== null && options.includes(a.partnerVoteTonight)) {
    return { game: p.game, decision: 'kill', target: a.partnerVoteTonight };
  }
  const pick = argmax(
    options,
    (s) => {
      let sc = a.threat.get(s) ?? 0;
      if (a.claimedSeer.has(s)) sc += 10;
      if (s === a.credibleSeer) sc += 6;
      if (a.claimedDoctor.has(s)) sc += 4;
      // Don't kill the seat the village is about to exile itself.
      sc -= (a.heat.get(s) ?? 0) * 2;
      sc -= (a.tallyToday.get(s) ?? 0) * 2;
      return sc;
    },
    rng,
  );
  return { game: p.game, decision: 'kill', target: pick ?? null };
}

function decideVote(p: Pending, a: Analysis, rng: () => number): Record<string, unknown> {
  const options = (p.options ?? []).filter((s) => s !== a.me);
  if (options.length === 0) return { game: p.game, decision: 'vote', target: null };
  if (a.role === 'werewolf') {
    // Never the partner. Bandwagon the villager the table already suspects.
    const pool = options.filter((s) => s !== a.partner);
    const use = pool.length > 0 ? pool : options;
    const leader = argmax(use, (s) => (a.tallyToday.get(s) ?? 0) * 3 + (a.heat.get(s) ?? 0), rng);
    if (leader !== null && ((a.tallyToday.get(leader) ?? 0) > 0 || (a.heat.get(leader) ?? 0) > 0)) {
      return { game: p.game, decision: 'vote', target: leader };
    }
    const pick = argmax(use, (s) => (a.threat.get(s) ?? 0) + (a.heat.get(s) ?? 0), rng);
    return { game: p.game, decision: 'vote', target: pick ?? null };
  }
  // Village side: a known wolf is the vote, full stop.
  const wolves = options.filter((s) => a.knownWolf.has(s));
  if (wolves.length > 0) {
    const pick = argmax(wolves, (s) => (a.tallyToday.get(s) ?? 0) + sus(a, s), rng);
    return { game: p.game, decision: 'vote', target: pick };
  }
  const top = argmax(options, (s) => sus(a, s), rng);
  const topScore = top === null ? -Infinity : sus(a, top);
  if (top !== null && topScore >= 2) return { game: p.game, decision: 'vote', target: top };
  // No strong read: join the forming plurality when its target is not a
  // known villager, rather than split the vote into a tie (no exile).
  const leader = argmax(options, (s) => a.tallyToday.get(s) ?? 0, rng);
  if (leader !== null && (a.tallyToday.get(leader) ?? 0) > 0 && !a.knownVillage.has(leader) && sus(a, leader) >= 0) {
    return { game: p.game, decision: 'vote', target: leader };
  }
  if (top !== null && topScore >= 1) return { game: p.game, decision: 'vote', target: top };
  return { game: p.game, decision: 'vote', target: null };
}

function reasonFor(a: Analysis, s: Seat): string {
  // One evidence phrase for suspecting seat s, from the log.
  const parts: string[] = [];
  const votedVillager = aRevealedVote(a, s, false);
  if (votedVillager !== null) parts.push(`they voted to exile ${who(a, votedVillager)}, who was a villager`);
  const accusedVillager = aAccused(a, s, false);
  if (accusedVillager !== null) parts.push(`they pushed the case on ${who(a, accusedVillager)}, a villager`);
  const defendedWolf = aDefended(a, s, true);
  if (defendedWolf !== null) parts.push(`they defended ${who(a, defendedWolf)}, who was a werewolf`);
  if (a.claimedSeer.has(s)) {
    const c = a.claimedSeer.get(s)!;
    if (c.target !== null && c.saidWolf === true && a.revealed.get(c.target) !== 'werewolf' && a.revealed.has(c.target)) {
      parts.push(`their seer claim about ${who(a, c.target)} was wrong`);
    }
  }
  return parts.length > 0 ? parts[0]! : 'their votes and words do not add up';
}

// Helpers that re-scan the game's log (the analysis already holds the maps;
// these read the raw log via the module-level reference set by decide()).
let CURRENT_LOG: LogEntry[] = [];
function aRevealedVote(a: Analysis, voter: Seat, wolf: boolean): Seat | null {
  for (const e of CURRENT_LOG) {
    if (e.kind === 'vote' && e.seat === voter && e.target != null) {
      const r = a.revealed.get(e.target);
      if (wolf ? r === 'werewolf' : r !== undefined && r !== 'werewolf') return e.target;
    }
  }
  return null;
}
function aAccused(a: Analysis, s: Seat, wolf: boolean): Seat | null {
  for (const e of CURRENT_LOG) {
    if (e.kind === 'speech' && e.seat === s && typeof e.text === 'string') {
      const parsed = parseSpeech(e.text);
      for (const t of parsed.accused) {
        const r = a.revealed.get(t);
        if (wolf ? r === 'werewolf' : r !== undefined && r !== 'werewolf') return t;
      }
    }
  }
  return null;
}
function aDefended(a: Analysis, s: Seat, wolf: boolean): Seat | null {
  for (const e of CURRENT_LOG) {
    if (e.kind === 'speech' && e.seat === s && typeof e.text === 'string') {
      const parsed = parseSpeech(e.text);
      for (const t of parsed.defended) {
        const r = a.revealed.get(t);
        if (wolf ? r === 'werewolf' : r !== undefined && r !== 'werewolf') return t;
      }
    }
  }
  return null;
}

function decideSpeak(p: Pending, a: Analysis, rng: () => number): Record<string, unknown> {
  const max = p.max_chars ?? 500;
  const round = a.mySpeechesToday + 1;
  const text = speakText(p, a, rng, round);
  if (text === null) return { game: p.game, decision: 'speak', pass: true };
  return { game: p.game, decision: 'speak', text: cutStatement(text, max) };
}

function speakText(p: Pending, a: Analysis, rng: () => number, round: number): string | null {
  void p;
  void rng;
  const livingOthers = a.living.filter((s) => s !== a.me);
  if (livingOthers.length === 0) return null;
  const topSuspect = argmax(livingOthers, (s) => sus(a, s), rng);
  const wolvesKnown = livingOthers.filter((s) => a.knownWolf.has(s));
  const pluralLeader = argmax(livingOthers, (s) => a.tallyToday.get(s) ?? 0, rng);

  if (a.role === 'seer') {
    if (wolvesKnown.length > 0) {
      const w = wolvesKnown[0]!;
      if (round === 1) {
        return `I am the seer. I checked ${who(a, w)} last night and they are a werewolf. Vote ${who(a, w)} today.`;
      }
      const w2 = wolvesKnown[1];
      if (w2 !== undefined) return `My checks so far: ${who(a, w)} is a werewolf and ${who(a, w2)} is a werewolf. Vote ${who(a, w)} first.`;
      return `I am the seer and my check stands: ${who(a, w)} is a werewolf. Do not let the vote drift — vote ${who(a, w)}.`;
    }
    // Only villager checks: stay hidden, steer with soft evidence.
    if (topSuspect !== null && sus(a, topSuspect) >= 2) {
      return `I find ${who(a, topSuspect)} the most suspicious here: ${reasonFor(a, topSuspect)}. I say we vote ${who(a, topSuspect)}.`;
    }
    return `No hard read yet. I trust ${who(a, livingOthers[0]!)} for now — their votes have tracked the village. Let's hear more before voting.`;
  }

  if (a.role === 'werewolf') {
    // Blend in: accuse a villager the table already doubts, or agree with it.
    const target = argmax(
      livingOthers.filter((s) => s !== a.partner),
      (s) => (a.heat.get(s) ?? 0) * 2 + (a.tallyToday.get(s) ?? 0) * 3 + sus(a, s),
      rng,
    );
    if (a.partner !== null && pluralLeader === a.partner) {
      if (target !== null) return `I am not convinced by the case on ${who(a, a.partner)} — their votes have tracked the village so far. I would rather look at ${who(a, target)}.`;
      return `I am not convinced by the case on ${who(a, a.partner)} — their votes have tracked the village so far.`;
    }
    if (target !== null && ((a.heat.get(target) ?? 0) > 0 || (a.tallyToday.get(target) ?? 0) > 0)) {
      return `${who(a, target)} looks off to me: ${reasonFor(a, target)}. I am voting ${who(a, target)}.`;
    }
    if (topSuspect !== null && topSuspect !== a.partner && sus(a, topSuspect) >= 2) {
      return `I find ${who(a, topSuspect)} the most suspicious here: ${reasonFor(a, topSuspect)}.`;
    }
    if (round === 1) {
      const q = livingOthers.filter((s) => s !== a.partner)[0]!;
      return `Nothing solid yet. I want to hear more from ${who(a, q)} before I decide my vote.`;
    }
    return `I am still listening. Nothing I have heard so far convinces me to vote yet.`;
  }

  // Village side: doctor or villager.
  if (wolvesKnown.length > 0) {
    // A house seer (or my own hidden knowledge) named a wolf: push it.
    const w = wolvesKnown[0]!;
    return `${who(a, w)} is a werewolf — the seer's check says so. Vote ${who(a, w)} today.`;
  }
  if (a.credibleSeer !== null) {
    const c = a.claimedSeer.get(a.credibleSeer);
    if (c?.target !== null && c.saidWolf === true && !a.knownVillage.has(c.target) && a.living.includes(c.target)) {
      return `${who(a, a.credibleSeer)} claims seer and checked ${who(a, c.target)} as a werewolf. That matches what I have seen — vote ${who(a, c.target)}.`;
    }
  }
  if (topSuspect !== null && sus(a, topSuspect) >= 2) {
    if (round === 1) return `I find ${who(a, topSuspect)} the most suspicious here: ${reasonFor(a, topSuspect)}. I say we vote ${who(a, topSuspect)}.`;
    return `The case on ${who(a, topSuspect)} still stands: ${reasonFor(a, topSuspect)}. Vote ${who(a, topSuspect)}.`;
  }
  if (pluralLeader !== null && (a.tallyToday.get(pluralLeader) ?? 0) > 0) {
    return `I can get behind the votes on ${who(a, pluralLeader)} — nothing I have seen clears them.`;
  }
  const trusted = argmax(livingOthers, (s) => -sus(a, s), rng);
  if (round === 1) {
    return `No strong read yet. I trust ${who(a, trusted ?? livingOthers[0]!)} for now — their votes have tracked the village. Let's hear more before voting.`;
  }
  return `Still no strong read. I am holding my vote until someone makes a real case.`;
}

/** The pass / abstention for a decision. */
export function fallbackFor(p: Pick<Pending, 'game' | 'decision'>): Record<string, unknown> {
  return p.decision === 'speak' ? { game: p.game, decision: 'speak', pass: true } : { game: p.game, decision: p.decision, target: null };
}

/** One decision, from its view alone: pure function of (view, seed). */
export function decide(p: Pending, seed: string): Record<string, unknown> {
  const view = p.view;
  if (!view.you.alive) return fallbackFor(p);
  CURRENT_LOG = view.log;
  const a = analyze(view);
  const salt = `${p.decision}:${view.day}:${view.log.length}`;
  const rng = rngFor(seed, p.game, salt);
  switch (p.decision) {
    case 'speak':
      return decideSpeak(p, a, rng);
    case 'vote':
      return decideVote(p, a, rng);
    case 'kill':
      return decideKill(p, a, rng);
    case 'protect':
      return decideProtect(p, a, rng);
    case 'inspect':
      return decideInspect(p, a, rng);
  }
}

// ---------------------------------------------------------------- run loop

export interface RunSummary {
  decisions: number;
  acted: number;
  fallbacks: number;
}

export interface RunOptions {
  envUrl?: string;
  seed?: string;
  log?: (line: string) => void;
}

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

export async function run(o: RunOptions = {}): Promise<RunSummary> {
  const env = process.env;
  const envUrl = (o.envUrl ?? env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  const seed = o.seed ?? env.MAKEFUN_TRIAL_SEED ?? 'werewolf-agent-01';
  const log = o.log ?? ((line: string) => process.stdout.write(`${line}\n`));
  if (envUrl === '') throw new Error('MAKEFUN_ENV_URL is needed');
  const summary: RunSummary = { decisions: 0, acted: 0, fallbacks: 0 };

  const envPost = async (path: string, body: unknown): Promise<{ status: number; body: unknown }> => {
    const res = await fetch(`${envUrl}${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
    return { status: res.status, body: parsed };
  };

  // A decision is known by its game, its kind and how far its game's log has
  // come: the same decision is listed again on every /observe until answered.
  const keyOf = (p: Pending): string => `${p.game}:${p.decision}:${p.view.log.length}`;
  const started = new Set<string>();
  const errors: string[] = [];
  for (;;) {
    const r = await envPost('/observe', {});
    if (r.status !== 200 || !isRecord(r.body)) throw new Error(`POST /observe answered ${r.status}`);
    const obs = r.body as unknown as Observed;
    if (obs.done) break;
    for (const p of obs.pending) {
      const key = keyOf(p);
      if (started.has(key)) continue;
      started.add(key);
      summary.decisions += 1;
      let act: Record<string, unknown>;
      try {
        act = decide(p, seed);
      } catch (err) {
        errors.push(`game ${p.game}: decide failed: ${(err as Error).message}`);
        break;
      }
      let posted = await envPost('/act', act);
      if (posted.status !== 200) {
        const fb = fallbackFor(p);
        if (JSON.stringify(act) !== JSON.stringify(fb)) {
          posted = await envPost('/act', fb);
          summary.fallbacks += 1;
        }
      }
      if (posted.status === 200) summary.acted += 1;
      else errors.push(`game ${p.game}: /act refused ${posted.status}`);
      if (errors.length > 0) break;
    }
    if (errors.length > 0) break;
    if (obs.pending.length === 0) await sleep(50);
    else if (obs.pending.every((p) => started.has(keyOf(p)))) await sleep(250); // listed, answered, not yet moved on
  }
  if (errors.length > 0) throw new Error(`the env failed: ${errors.join('; ')}`);
  log(`werewolf-agent: ${summary.decisions} decisions, ${summary.acted} acted, ${summary.fallbacks} fell back`);
  return summary;
}

const isMain = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  try {
    await run();
    process.exitCode = 0;
  } catch (err) {
    process.stderr.write(`werewolf-agent: ${(err as Error).stack ?? String(err)}\n`);
    process.exitCode = 1;
  }
}

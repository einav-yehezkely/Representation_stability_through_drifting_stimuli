// Seeded randomization. One explicitly seeded generator family is used for every experimental
// random decision. Stimulus selection is deterministic, so randomness only determines which group
// supplies each sample and on which side each group's face appears.
//
// Algorithm: sfc32 (Small Fast Counter, 32-bit), version 1. Each named stream is seeded from
// (sessionSeed, streamName) through splitmix32, so the streams are independent of each other.

export const RNG_ALGORITHM = "sfc32";
export const RNG_ALGORITHM_VERSION = "1";

function fnv1a32(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function splitmix32(state) {
  let s = state >>> 0;
  return () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
    return (z ^ (z >>> 16)) >>> 0;
  };
}

export class SeededRng {
  constructor(sessionSeed, streamName) {
    if (!Number.isInteger(sessionSeed) || sessionSeed < 0 || sessionSeed > 0xffffffff) {
      throw new Error("sessionSeed must be an unsigned 32-bit integer");
    }
    this.sessionSeed = sessionSeed;
    this.streamName = streamName;
    const init = splitmix32((sessionSeed ^ fnv1a32(streamName)) >>> 0);
    this.a = init();
    this.b = init();
    this.c = init();
    this.d = init();
    this.draws = 0;
    for (let i = 0; i < 12; i += 1) this.nextUint32(); // warm-up, not counted
    this.draws = 0;
  }

  nextUint32() {
    const t = (((this.a + this.b) >>> 0) + this.d) >>> 0;
    this.d = (this.d + 1) >>> 0;
    this.a = this.b ^ (this.b >>> 9);
    this.b = (this.c + (this.c << 3)) >>> 0;
    this.c = ((this.c << 21) | (this.c >>> 11)) >>> 0;
    this.c = (this.c + t) >>> 0;
    this.draws += 1;
    return t;
  }

  /** Uniform float in [0, 1). */
  next() {
    return this.nextUint32() / 4294967296;
  }

  /** Uniform integer in [0, n). */
  nextInt(n) {
    return Math.floor(this.next() * n);
  }

  state() {
    return { stream: this.streamName, draws: this.draws };
  }
}

export function createRngFactory(sessionSeed) {
  return (streamName) => new SeededRng(sessionSeed, streamName);
}

/** New random session seed (uint32) from the platform CSPRNG; the value is recorded. */
export function generateSessionSeed(cryptoImpl = globalThis.crypto) {
  const buffer = new Uint32Array(1);
  cryptoImpl.getRandomValues(buffer);
  return buffer[0];
}

/** In-place Fisher–Yates shuffle driven by `rng`. */
export function shuffle(array, rng) {
  for (let i = array.length - 1; i > 0; i -= 1) {
    const j = rng.nextInt(i + 1);
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

/**
 * Balanced schedule of groups: floor(N/2) A and floor(N/2) B; for odd N the extra group is chosen
 * by the RNG. Shuffled with Fisher–Yates. |#A - #B| <= 1.
 */
export function createBalancedSchedule(numberOfTrials, rng) {
  const half = Math.floor(numberOfTrials / 2);
  const groups = [...Array(half).fill("A"), ...Array(half).fill("B")];
  if (numberOfTrials % 2 === 1) groups.push(rng.nextInt(2) === 0 ? "A" : "B");
  return shuffle(groups, rng);
}

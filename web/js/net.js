// model/temporal_gnn.py in the browser: typed-array inference for the trained
// temporal graph network. Same arithmetic as the torch model, same weights.
import { NODE_DIM, EDGE_DIM } from "./sceneGraph.js";

export const HIDDEN = 64;
export const EDGE_EMB = 24;
export const N_TRACK_OUT = 11;
export const MAX_TURN = 0.2;
export const TURN_QUANTILE = 0.9;
export const MAX_BIAS = 0.03;
export const MAX_FLANK = 1.0;
export const MAX_WOBBLE = 1.2;

const sigmoid = (v) => 1.0 / (1.0 + Math.exp(-v));

// A torch Linear: W is (out, in) row-major, bias length out.
function linear(outBuf, inBuf, rows, W, b, inDim, outDim, inStride, outStride) {
  for (let r = 0; r < rows; r++) {
    const ib = r * (inStride || inDim), ob = r * (outStride || outDim);
    for (let o = 0; o < outDim; o++) {
      let acc = b ? b[o] : 0.0;
      const wb = o * inDim;
      for (let i = 0; i < inDim; i++) acc += W[wb + i] * inBuf[ib + i];
      outBuf[ob + o] = acc;
    }
  }
}

function reluInPlace(buf, n) {
  for (let i = 0; i < n; i++) if (buf[i] < 0) buf[i] = 0;
}

// torch LayerNorm over the last dimension, eps 1e-5.
function layerNorm(buf, rows, dim, w, b) {
  for (let r = 0; r < rows; r++) {
    const base = r * dim;
    let mean = 0.0;
    for (let i = 0; i < dim; i++) mean += buf[base + i];
    mean /= dim;
    let varr = 0.0;
    for (let i = 0; i < dim; i++) {
      const d = buf[base + i] - mean;
      varr += d * d;
    }
    varr /= dim;
    const inv = 1.0 / Math.sqrt(varr + 1e-5);
    for (let i = 0; i < dim; i++) buf[base + i] = (buf[base + i] - mean) * inv * w[i] + b[i];
  }
}

export class TemporalGraphNet {
  constructor(manifest, data) {
    this.config = manifest.config;
    this.useEdges = this.config.use_edges !== false;
    this.useMemory = this.config.use_memory !== false;
    this.layersN = this.config.layers === undefined ? 2 : this.config.layers;
    this.hidden = this.config.hidden === undefined ? HIDDEN : this.config.hidden;
    this.t = {};
    for (const [name, meta] of Object.entries(manifest.tensors)) {
      this.t[name] = data.subarray(meta.offset, meta.offset + meta.size);
    }
    this.params = Object.values(manifest.tensors).reduce((a, m) => a + m.size, 0);
    this._buf = {};
  }

  // Scratch buffers, grown on demand and reused every tick.
  _scratch(name, size) {
    let b = this._buf[name];
    if (b === undefined || b.length < size) {
      b = new Float32Array(size);
      this._buf[name] = b;
    }
    return b;
  }

  // Returns { z (N x 2H), e (E x EDGE_EMB), newMem (N x H), gates (E) }.
  forward(g, mem) {
    const H = this.hidden, N = g.n, E = g.e, t = this.t;
    const h = this._scratch("h", N * H);
    const tmp = this._scratch("tmp", N * H);

    // node_in: Linear -> ReLU -> Linear
    linear(tmp, g.x, N, t["node_in.0.weight"], t["node_in.0.bias"], NODE_DIM, H);
    reluInPlace(tmp, N * H);
    linear(h, tmp, N, t["node_in.2.weight"], t["node_in.2.bias"], H, H);

    if (this.useMemory) {
      // mem_in: Linear without bias, added to h
      const mi = this._scratch("mi", N * H);
      linear(mi, mem, N, t["mem_in.weight"], null, H, H);
      for (let i = 0; i < N * H; i++) h[i] += mi[i];
    }
    layerNorm(h, N, H, t["in_norm.weight"], t["in_norm.bias"]);

    // Edge embeddings: relu(etype_emb[etype] + eattr_in(eattr))
    const e = this._scratch("e", Math.max(1, E) * EDGE_EMB);
    if (E) {
      linear(e, g.eattr, E, t["eattr_in.weight"], t["eattr_in.bias"], EDGE_DIM, EDGE_EMB);
      const emb = t["etype_emb.weight"];
      for (let i = 0; i < E; i++) {
        const base = i * EDGE_EMB, eb = g.etype[i] * EDGE_EMB;
        for (let k = 0; k < EDGE_EMB; k++) {
          const v = e[base + k] + emb[eb + k];
          e[base + k] = v < 0 ? 0 : v;
        }
      }
    }

    let gates = null;
    if (this.useEdges && E) {
      const ZDIM = 2 * H + EDGE_EMB;              // 152
      const z = this._scratch("z", E * ZDIM);
      const msgTmp = this._scratch("msgTmp", E * H);
      const msg = this._scratch("msg", E * H);
      const gateBuf = this._scratch("gate", E);
      const agg = this._scratch("agg", N * H);
      const deg = this._scratch("deg", N);
      const updIn = this._scratch("updIn", N * 2 * H);
      const updTmp = this._scratch("updTmp", N * H);
      const updOut = this._scratch("updOut", N * H);

      for (let L = 0; L < this.layersN; L++) {
        const p = "layers." + L + ".";
        for (let i = 0; i < E; i++) {
          const zb = i * ZDIM, sb = g.src[i] * H, db = g.dst[i] * H, eb = i * EDGE_EMB;
          for (let k = 0; k < H; k++) z[zb + k] = h[sb + k];
          for (let k = 0; k < H; k++) z[zb + H + k] = h[db + k];
          for (let k = 0; k < EDGE_EMB; k++) z[zb + 2 * H + k] = e[eb + k];
        }
        // gate = sigmoid(Linear(z)); m = msg(z) * gate
        linear(gateBuf, z, E, t[p + "gate.weight"], t[p + "gate.bias"], ZDIM, 1);
        for (let i = 0; i < E; i++) gateBuf[i] = sigmoid(gateBuf[i]);
        linear(msgTmp, z, E, t[p + "msg.0.weight"], t[p + "msg.0.bias"], ZDIM, H);
        reluInPlace(msgTmp, E * H);
        linear(msg, msgTmp, E, t[p + "msg.2.weight"], t[p + "msg.2.bias"], H, H);
        for (let i = 0; i < E; i++) {
          const gv = gateBuf[i], mb = i * H;
          for (let k = 0; k < H; k++) msg[mb + k] *= gv;
        }
        // agg = index_add(dst, m) / sqrt(max(deg, 1))
        agg.fill(0.0, 0, N * H);
        deg.fill(0.0, 0, N);
        for (let i = 0; i < E; i++) {
          const db = g.dst[i] * H, mb = i * H;
          for (let k = 0; k < H; k++) agg[db + k] += msg[mb + k];
          deg[g.dst[i]] += 1.0;
        }
        for (let n = 0; n < N; n++) {
          const s = 1.0 / Math.sqrt(Math.max(deg[n], 1.0));
          const nb = n * H;
          for (let k = 0; k < H; k++) agg[nb + k] *= s;
        }
        // h = norm(h + upd(cat([h, agg])))
        for (let n = 0; n < N; n++) {
          const ub = n * 2 * H, nb = n * H;
          for (let k = 0; k < H; k++) updIn[ub + k] = h[nb + k];
          for (let k = 0; k < H; k++) updIn[ub + H + k] = agg[nb + k];
        }
        linear(updTmp, updIn, N, t[p + "upd.0.weight"], t[p + "upd.0.bias"], 2 * H, H);
        reluInPlace(updTmp, N * H);
        linear(updOut, updTmp, N, t[p + "upd.2.weight"], t[p + "upd.2.bias"], H, H);
        for (let i = 0; i < N * H; i++) h[i] += updOut[i];
        layerNorm(h, N, H, t[p + "norm.weight"], t[p + "norm.bias"]);
        if (L === this.layersN - 1) {
          gates = gateBuf.subarray(0, E);
        }
      }
    }

    // GRUCell memory update
    const newMem = this._scratch("newMem", N * H);
    if (this.useMemory) {
      const gi = this._scratch("gi", N * 3 * H);
      const gh = this._scratch("gh", N * 3 * H);
      linear(gi, h, N, t["memory.weight_ih"], t["memory.bias_ih"], H, 3 * H);
      linear(gh, mem, N, t["memory.weight_hh"], t["memory.bias_hh"], H, 3 * H);
      for (let n = 0; n < N; n++) {
        const b3 = n * 3 * H, b1 = n * H;
        for (let k = 0; k < H; k++) {
          const r = sigmoid(gi[b3 + k] + gh[b3 + k]);
          const zz = sigmoid(gi[b3 + H + k] + gh[b3 + H + k]);
          const nn = Math.tanh(gi[b3 + 2 * H + k] + r * gh[b3 + 2 * H + k]);
          newMem[b1 + k] = (1.0 - zz) * nn + zz * mem[b1 + k];
        }
      }
    } else {
      newMem.fill(0.0, 0, N * H);
    }

    // z = cat([h, new_mem])
    const zOut = this._scratch("zOut", N * 2 * H);
    for (let n = 0; n < N; n++) {
      const zb = n * 2 * H, nb = n * H;
      for (let k = 0; k < H; k++) zOut[zb + k] = h[nb + k];
      for (let k = 0; k < H; k++) zOut[zb + H + k] = newMem[nb + k];
    }
    return { z: zOut, e: e, newMem: newMem, gates: gates, N: N, E: E, H: H };
  }

  // track_head on the given rows -> decoded estimates, one object per row.
  decodeTracks(fw, rows) {
    const H = this.hidden, t = this.t, R = rows.length;
    if (!R) return [];
    const inBuf = this._scratch("thIn", R * 2 * H);
    for (let r = 0; r < R; r++) {
      const src = rows[r] * 2 * H, dstB = r * 2 * H;
      for (let k = 0; k < 2 * H; k++) inBuf[dstB + k] = fw.z[src + k];
    }
    const tmp = this._scratch("thTmp", R * H);
    const raw = this._scratch("thRaw", R * N_TRACK_OUT);
    linear(tmp, inBuf, R, t["track_head.0.weight"], t["track_head.0.bias"], 2 * H, H);
    reluInPlace(tmp, R * H);
    linear(raw, tmp, R, t["track_head.2.weight"], t["track_head.2.bias"], H, N_TRACK_OUT);
    const out = [];
    for (let r = 0; r < R; r++) {
      const b = r * N_TRACK_OUT;
      const sideSign = raw[b + 3] > 0 ? 1.0 : (raw[b + 3] < 0 ? -1.0 : 0.0);
      out.push({
        pursuit: sigmoid(raw[b]),
        turn_rate: sigmoid(raw[b + 1]) * MAX_TURN,
        lead: sigmoid(raw[b + 2]),
        side: sideSign * (raw[b + 10] > 0.0 ? 1.0 : 0.0),
        guided: sigmoid(raw[b + 4]),
        threat: sigmoid(raw[b + 5]),
        turn_hi: sigmoid(raw[b + 6]) * MAX_TURN,
        bias: Math.tanh(raw[b + 7]) * MAX_BIAS,
        flank: sigmoid(raw[b + 8]) * MAX_FLANK,
        wobble: sigmoid(raw[b + 9]) * MAX_WOBBLE,
      });
    }
    return out;
  }

  // Symmetric logit per LINK pair; rows index the forward edge, row+1 its reverse.
  linkProbs(fw, g, rows) {
    const H = this.hidden, t = this.t, R = rows.length;
    if (!R) return [];
    const IN = 4 * H + 2 * EDGE_EMB;             // 304
    const inBuf = this._scratch("lkIn", 2 * R * IN);
    for (let r = 0; r < R; r++) {
      const row = rows[r];
      const a = g.src[row] * 2 * H, b = g.dst[row] * 2 * H;
      const ef = row * EDGE_EMB, er = (row + 1) * EDGE_EMB;
      let o = (2 * r) * IN;
      for (let k = 0; k < 2 * H; k++) inBuf[o + k] = fw.z[a + k];
      for (let k = 0; k < 2 * H; k++) inBuf[o + 2 * H + k] = fw.z[b + k];
      for (let k = 0; k < EDGE_EMB; k++) inBuf[o + 4 * H + k] = fw.e[ef + k];
      for (let k = 0; k < EDGE_EMB; k++) inBuf[o + 4 * H + EDGE_EMB + k] = fw.e[er + k];
      o = (2 * r + 1) * IN;
      for (let k = 0; k < 2 * H; k++) inBuf[o + k] = fw.z[b + k];
      for (let k = 0; k < 2 * H; k++) inBuf[o + 2 * H + k] = fw.z[a + k];
      for (let k = 0; k < EDGE_EMB; k++) inBuf[o + 4 * H + k] = fw.e[er + k];
      for (let k = 0; k < EDGE_EMB; k++) inBuf[o + 4 * H + EDGE_EMB + k] = fw.e[ef + k];
    }
    const tmp = this._scratch("lkTmp", 2 * R * H);
    const raw = this._scratch("lkRaw", 2 * R);
    linear(tmp, inBuf, 2 * R, t["link_head.0.weight"], t["link_head.0.bias"], IN, H);
    reluInPlace(tmp, 2 * R * H);
    linear(raw, tmp, 2 * R, t["link_head.2.weight"], t["link_head.2.bias"], H, 1);
    const out = [];
    for (let r = 0; r < R; r++) out.push(sigmoid(0.5 * (raw[2 * r] + raw[2 * r + 1])));
    return out;
  }
}

// Runtime wrapper: per-node memory for one match, keyed by node key (TGN-style).
export class GraphBrain {
  constructor(model) {
    this.model = model;
    this.memory = new Map();
    this.zero = new Float32Array(model.hidden);
  }

  reset() { this.memory = new Map(); }

  step(g) {
    const H = this.model.hidden, N = g.n;
    if (N === 0) return { rows: [], links: [], gates: null };
    const mem = new Float32Array(N * H);
    for (let i = 0; i < N; i++) {
      const prev = this.memory.get(g.keys[i]);
      if (prev) mem.set(prev, i * H);
    }
    const fw = this.model.forward(g, mem);
    // Nodes absent from this snapshot are deleted from memory (TGN node deletion).
    const next = new Map();
    for (let i = 0; i < N; i++) {
      next.set(g.keys[i], fw.newMem.slice(i * H, (i + 1) * H));
    }
    this.memory = next;

    const rows = this.model.decodeTracks(fw, g.track_rows);
    const links = [];
    if (g.link_rows.length) {
      const probs = this.model.linkProbs(fw, g, g.link_rows);
      const tidOf = new Map();
      for (let i = 0; i < g.track_rows.length; i++) tidOf.set(g.track_rows[i], g.track_tids[i]);
      for (let i = 0; i < g.link_rows.length; i++) {
        const r = g.link_rows[i];
        links.push([tidOf.get(g.src[r]), tidOf.get(g.dst[r]), probs[i]]);
      }
    }
    return { rows: rows, links: links, gates: fw.gates };
  }
}

// Half-precision weights, expanded once at load. The web build ships float16
// (234 KB instead of 458 KB); the weights are O(1) so the cost is ~2e-4 relative.
export function halfToFloat(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sign = (h & 0x8000) ? -1 : 1;
    const exp = (h & 0x7C00) >> 10;
    const frac = h & 0x03FF;
    if (exp === 0) out[i] = sign * 6.103515625e-5 * (frac / 1024);
    else if (exp === 0x1F) out[i] = frac ? NaN : sign * Infinity;
    else out[i] = sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
  }
  return out;
}

export async function loadBrain(base) {
  const manifest = await fetch(base + ".json").then((r) => r.json());
  const dir = base.replace(/[^/]*$/, "");
  if (manifest.b64) {
    // One base64 text file: the only binary-safe form some hosts will serve.
    const text = await fetch(dir + manifest.b64).then((r) => r.text());
    const bin = atob(text.trim());
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new TemporalGraphNet(manifest, halfToFloat(new Uint16Array(bytes.buffer, 0, bytes.length / 2)));
  }
  if (manifest.chunks) {
    const parts = await Promise.all(
      manifest.chunks.map((c) => fetch(dir + c).then((r) => r.arrayBuffer())));
    const total = parts.reduce((a, p) => a + p.byteLength, 0);
    const bytes = new Uint8Array(total);
    let at = 0;
    for (const p of parts) { bytes.set(new Uint8Array(p), at); at += p.byteLength; }
    const u16 = new Uint16Array(bytes.buffer, 0, total / 2);
    return new TemporalGraphNet(manifest, halfToFloat(u16));
  }
  const blob = await fetch(base + ".bin").then((r) => r.arrayBuffer());
  return new TemporalGraphNet(manifest, new Float32Array(blob));
}

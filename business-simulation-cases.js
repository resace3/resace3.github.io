/**
 * Business Simulation Cases — interactive explorer.
 *
 * Every case mirrors the data-generating process of one notebook in
 * github.com/resace3/Simulation_Cases_book. Nothing is fetched or stored:
 * each sample is drawn in the browser from a seeded generator, so moving a
 * control re-runs the whole case end to end.
 */
(() => {
  const root = document.querySelector("[data-sim-cases]");
  if (!root) return;

  const REPO = "https://github.com/resace3/Simulation_Cases_book/blob/main/";

  /* ------------------------------------------------------------------ *
   * Seeded random generator with the samplers the notebooks rely on.
   * ------------------------------------------------------------------ */

  function makeRng(seed) {
    let state = seed >>> 0;

    function next() {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }

    const rng = {
      next,
      uniform(a = 0, b = 1) {
        return a + (b - a) * next();
      },
      randint(lo, hi) {
        return lo + Math.floor(next() * (hi - lo));
      },
      normal(mu = 0, sd = 1) {
        let u = 0;
        let v = 0;
        while (u === 0) u = next();
        while (v === 0) v = next();
        return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      },
      exponential(scale = 1) {
        return -scale * Math.log(1 - next());
      },
      bernoulli(p) {
        return next() < p ? 1 : 0;
      },
      poisson(lam) {
        if (lam <= 0) return 0;
        if (lam < 30) {
          const limit = Math.exp(-lam);
          let k = 0;
          let product = 1;
          do {
            k += 1;
            product *= next();
          } while (product > limit);
          return k - 1;
        }
        return Math.max(0, Math.round(rng.normal(lam, Math.sqrt(lam))));
      },
      gamma(shape, scale = 1) {
        if (shape < 1) {
          return rng.gamma(shape + 1, scale) * Math.pow(next(), 1 / shape);
        }
        const d = shape - 1 / 3;
        const c = 1 / Math.sqrt(9 * d);
        for (;;) {
          let x;
          let v;
          do {
            x = rng.normal(0, 1);
            v = 1 + c * x;
          } while (v <= 0);
          v = v * v * v;
          const u = next();
          if (u < 1 - 0.0331 * x * x * x * x) return d * v * scale;
          if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
        }
      },
      beta(a, b) {
        const x = rng.gamma(a, 1);
        const y = rng.gamma(b, 1);
        return x + y === 0 ? 0 : x / (x + y);
      },
      choice(items, probs) {
        if (!probs) return items[Math.floor(next() * items.length)];
        const r = next();
        let acc = 0;
        for (let i = 0; i < items.length; i += 1) {
          acc += probs[i];
          if (r < acc) return items[i];
        }
        return items[items.length - 1];
      },
    };

    return rng;
  }

  /* ------------------------------------------------------------------ *
   * Statistics
   * ------------------------------------------------------------------ */

  const sum = (values) => values.reduce((acc, v) => acc + v, 0);
  const mean = (values) => (values.length ? sum(values) / values.length : 0);

  function sd(values) {
    if (values.length < 2) return 0;
    const m = mean(values);
    return Math.sqrt(sum(values.map((v) => (v - m) ** 2)) / (values.length - 1));
  }

  function quantile(values, q) {
    if (!values.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  const median = (values) => quantile(values, 0.5);

  function erf(x) {
    const sign = x < 0 ? -1 : 1;
    const ax = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * ax);
    const y =
      1 -
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
        t *
        Math.exp(-ax * ax);
    return sign * y;
  }

  const normalCdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

  /** Welch's two-sample t test. p uses the normal approximation. */
  function welchTest(a, b) {
    const ma = mean(a);
    const mb = mean(b);
    const va = sd(a) ** 2;
    const vb = sd(b) ** 2;
    const se = Math.sqrt(va / a.length + vb / b.length);
    const diff = mb - ma;
    const t = se === 0 ? 0 : diff / se;
    return {
      meanA: ma,
      meanB: mb,
      diff,
      se,
      t,
      p: 2 * (1 - normalCdf(Math.abs(t))),
      ciLow: diff - 1.96 * se,
      ciHigh: diff + 1.96 * se,
    };
  }

  /** Gaussian elimination with partial pivoting. */
  function solveLinear(A, b) {
    const n = b.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col += 1) {
      let pivot = col;
      for (let r = col + 1; r < n; r += 1) {
        if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
      }
      if (Math.abs(M[pivot][col]) < 1e-12) return null;
      [M[col], M[pivot]] = [M[pivot], M[col]];
      for (let r = 0; r < n; r += 1) {
        if (r === col) continue;
        const factor = M[r][col] / M[col][col];
        if (factor === 0) continue;
        for (let c = col; c <= n; c += 1) M[r][c] -= factor * M[col][c];
      }
    }
    return M.map((row, i) => row[n] / M[i][i]);
  }

  /** Ordinary least squares with an intercept prepended. */
  function ols(X, y) {
    const k = X[0].length + 1;
    const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
    const Xty = new Array(k).fill(0);

    for (let i = 0; i < X.length; i += 1) {
      const row = [1, ...X[i]];
      for (let a = 0; a < k; a += 1) {
        Xty[a] += row[a] * y[i];
        for (let b = 0; b < k; b += 1) XtX[a][b] += row[a] * row[b];
      }
    }

    // A constant column (a lever held at 0% or 100%) makes X'X singular: the
    // effect is not identifiable and callers should say so rather than print a
    // number that looks like an estimate.
    const solved = solveLinear(XtX, Xty);
    const coef = solved || new Array(k).fill(0);
    const fitted = X.map((row) => coef[0] + row.reduce((acc, v, j) => acc + v * coef[j + 1], 0));
    const ybar = mean(y);
    const ssRes = sum(y.map((v, i) => (v - fitted[i]) ** 2));
    const ssTot = sum(y.map((v) => (v - ybar) ** 2));
    return {
      coef,
      fitted,
      singular: !solved,
      r2: ssTot === 0 ? 0 : 1 - ssRes / ssTot,
      rmse: Math.sqrt(ssRes / y.length),
    };
  }

  function standardize(X) {
    const k = X[0].length;
    const mus = [];
    const sds = [];
    for (let j = 0; j < k; j += 1) {
      const col = X.map((row) => row[j]);
      const m = mean(col);
      const s = sd(col) || 1;
      mus.push(m);
      sds.push(s);
    }
    return {
      Z: X.map((row) => row.map((v, j) => (v - mus[j]) / sds[j])),
      mus,
      sds,
    };
  }

  const sigmoid = (z) => 1 / (1 + Math.exp(-Math.max(-35, Math.min(35, z))));

  /** Logistic regression by full-batch gradient descent on standardized inputs. */
  function logisticFit(X, y, iterations = 160, lr = 0.6) {
    const { Z } = standardize(X);
    const k = Z[0].length;
    const n = Z.length;
    const w = new Array(k).fill(0);
    let b = 0;

    for (let step = 0; step < iterations; step += 1) {
      const gw = new Array(k).fill(0);
      let gb = 0;
      for (let i = 0; i < n; i += 1) {
        let z = b;
        for (let j = 0; j < k; j += 1) z += w[j] * Z[i][j];
        const err = sigmoid(z) - y[i];
        gb += err;
        for (let j = 0; j < k; j += 1) gw[j] += err * Z[i][j];
      }
      b -= (lr * gb) / n;
      for (let j = 0; j < k; j += 1) w[j] -= (lr * gw[j]) / n;
    }

    const scores = Z.map((row) => sigmoid(b + row.reduce((acc, v, j) => acc + v * w[j], 0)));
    return { scores, weights: w, bias: b };
  }

  /** Rank-based ROC AUC. */
  function auc(scores, labels) {
    const order = scores.map((s, i) => [s, labels[i]]).sort((a, b) => a[0] - b[0]);
    let rank = 1;
    let positiveRankSum = 0;
    let positives = 0;
    let i = 0;
    while (i < order.length) {
      let j = i;
      while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j += 1;
      const avgRank = (rank + (rank + (j - i))) / 2;
      for (let t = i; t <= j; t += 1) {
        if (order[t][1] === 1) {
          positiveRankSum += avgRank;
          positives += 1;
        }
      }
      rank += j - i + 1;
      i = j + 1;
    }
    const negatives = order.length - positives;
    if (!positives || !negatives) return 0.5;
    return (positiveRankSum - (positives * (positives + 1)) / 2) / (positives * negatives);
  }

  function confusion(scores, labels, threshold) {
    let tp = 0;
    let fp = 0;
    let tn = 0;
    let fn = 0;
    for (let i = 0; i < scores.length; i += 1) {
      const flagged = scores[i] >= threshold;
      if (labels[i] === 1) {
        if (flagged) tp += 1;
        else fn += 1;
      } else if (flagged) fp += 1;
      else tn += 1;
    }
    return {
      tp,
      fp,
      tn,
      fn,
      precision: tp + fp ? tp / (tp + fp) : 0,
      recall: tp + fn ? tp / (tp + fn) : 0,
      accuracy: (tp + tn) / Math.max(1, scores.length),
      flagged: tp + fp,
    };
  }

  /** Lloyd's algorithm on standardized coordinates. */
  function kmeans(points, k, rng, iterations = 24) {
    const centers = [];
    const used = new Set();
    while (centers.length < k && used.size < points.length) {
      const idx = rng.randint(0, points.length);
      if (used.has(idx)) continue;
      used.add(idx);
      centers.push([...points[idx]]);
    }
    const assign = new Array(points.length).fill(0);

    for (let step = 0; step < iterations; step += 1) {
      let moved = false;
      for (let i = 0; i < points.length; i += 1) {
        let best = 0;
        let bestDist = Infinity;
        for (let c = 0; c < centers.length; c += 1) {
          let dist = 0;
          for (let d = 0; d < points[i].length; d += 1) dist += (points[i][d] - centers[c][d]) ** 2;
          if (dist < bestDist) {
            bestDist = dist;
            best = c;
          }
        }
        if (assign[i] !== best) moved = true;
        assign[i] = best;
      }
      const sums = centers.map(() => new Array(points[0].length).fill(0));
      const counts = new Array(centers.length).fill(0);
      for (let i = 0; i < points.length; i += 1) {
        counts[assign[i]] += 1;
        for (let d = 0; d < points[i].length; d += 1) sums[assign[i]][d] += points[i][d];
      }
      for (let c = 0; c < centers.length; c += 1) {
        if (!counts[c]) continue;
        for (let d = 0; d < centers[c].length; d += 1) centers[c][d] = sums[c][d] / counts[c];
      }
      if (!moved && step > 0) break;
    }

    const sizes = new Array(k).fill(0);
    assign.forEach((c) => {
      sizes[c] += 1;
    });
    return { assign, centers, sizes };
  }

  /** Isolation Forest: average path length over random axis-aligned splits. */
  function isolationForest(points, rng, treeCount = 70, sampleSize = 192) {
    const dims = points[0].length;
    const limit = Math.ceil(Math.log2(Math.max(2, sampleSize)));

    function build(rows, depth) {
      if (depth >= limit || rows.length <= 1) return { size: rows.length };
      const d = rng.randint(0, dims);
      let lo = Infinity;
      let hi = -Infinity;
      for (const row of rows) {
        if (row[d] < lo) lo = row[d];
        if (row[d] > hi) hi = row[d];
      }
      if (lo === hi) return { size: rows.length };
      const split = rng.uniform(lo, hi);
      const left = [];
      const right = [];
      for (const row of rows) (row[d] < split ? left : right).push(row);
      if (!left.length || !right.length) return { size: rows.length };
      return { d, split, left: build(left, depth + 1), right: build(right, depth + 1) };
    }

    const harmonic = (n) => (n <= 1 ? 0 : 2 * (Math.log(n - 1) + 0.5772156649) - (2 * (n - 1)) / n);

    function pathLength(node, row, depth) {
      if (node.size !== undefined) return depth + harmonic(node.size);
      return pathLength(row[node.d] < node.split ? node.left : node.right, row, depth + 1);
    }

    const trees = [];
    for (let t = 0; t < treeCount; t += 1) {
      const sample = [];
      const take = Math.min(sampleSize, points.length);
      for (let i = 0; i < take; i += 1) sample.push(points[rng.randint(0, points.length)]);
      trees.push(build(sample, 0));
    }

    const norm = harmonic(Math.min(sampleSize, points.length)) || 1;
    return points.map((row) => {
      const avg = mean(trees.map((tree) => pathLength(tree, row, 0)));
      return Math.pow(2, -avg / norm);
    });
  }

  /**
   * One hidden layer, tanh activation, sigmoid output.
   * Each epoch descends on a bounded mini-batch so that dragging a slider
   * stays interactive even at the top of the parameter ranges.
   */
  const MLP_BATCH = 384;

  function trainMlp(X, y, hidden, epochs, rng) {
    const { Z } = standardize(X);
    const n = Z.length;
    const k = Z[0].length;
    const split = Math.floor(n * 0.75);
    const trainIdx = [];
    const testIdx = [];
    for (let i = 0; i < n; i += 1) (i < split ? trainIdx : testIdx).push(i);

    const scale = Math.sqrt(2 / k);
    const W1 = Array.from({ length: hidden }, () =>
      Array.from({ length: k }, () => rng.normal(0, scale))
    );
    const b1 = new Array(hidden).fill(0);
    const W2 = Array.from({ length: hidden }, () => rng.normal(0, 0.5));
    let b2 = 0;
    const lr = 0.35;
    const losses = [];

    const forward = (row) => {
      const h = new Array(hidden);
      for (let j = 0; j < hidden; j += 1) {
        let z = b1[j];
        for (let f = 0; f < k; f += 1) z += W1[j][f] * row[f];
        h[j] = Math.tanh(z);
      }
      let out = b2;
      for (let j = 0; j < hidden; j += 1) out += W2[j] * h[j];
      return { h, p: sigmoid(out) };
    };

    const batchSize = Math.min(MLP_BATCH, trainIdx.length);

    for (let epoch = 0; epoch < epochs; epoch += 1) {
      const gW1 = W1.map((row) => new Array(k).fill(0));
      const gb1 = new Array(hidden).fill(0);
      const gW2 = new Array(hidden).fill(0);
      let gb2 = 0;
      let loss = 0;

      const batch =
        batchSize === trainIdx.length
          ? trainIdx
          : Array.from({ length: batchSize }, () => trainIdx[rng.randint(0, trainIdx.length)]);

      for (const i of batch) {
        const { h, p } = forward(Z[i]);
        const err = p - y[i];
        loss += -(y[i] * Math.log(p + 1e-9) + (1 - y[i]) * Math.log(1 - p + 1e-9));
        gb2 += err;
        for (let j = 0; j < hidden; j += 1) {
          gW2[j] += err * h[j];
          const dh = err * W2[j] * (1 - h[j] * h[j]);
          gb1[j] += dh;
          for (let f = 0; f < k; f += 1) gW1[j][f] += dh * Z[i][f];
        }
      }

      const m = batch.length;
      b2 -= (lr * gb2) / m;
      for (let j = 0; j < hidden; j += 1) {
        W2[j] -= (lr * gW2[j]) / m;
        b1[j] -= (lr * gb1[j]) / m;
        for (let f = 0; f < k; f += 1) W1[j][f] -= (lr * gW1[j][f]) / m;
      }
      if (epoch % Math.max(1, Math.floor(epochs / 40)) === 0) losses.push(loss / m);
    }

    const scoreOf = (i) => forward(Z[i]).p;
    const trainScores = trainIdx.map(scoreOf);
    const testScores = testIdx.map(scoreOf);
    const trainLabels = trainIdx.map((i) => y[i]);
    const testLabels = testIdx.map((i) => y[i]);

    return {
      losses,
      trainAccuracy: confusion(trainScores, trainLabels, 0.5).accuracy,
      testAccuracy: confusion(testScores, testLabels, 0.5).accuracy,
      testAuc: auc(testScores, testLabels),
      testConfusion: confusion(testScores, testLabels, 0.5),
      testSize: testIdx.length,
    };
  }

  /* ------------------------------------------------------------------ *
   * Formatting
   * ------------------------------------------------------------------ */

  const fmtInt = (v) => Math.round(v).toLocaleString("en-US");
  const fmtNum = (v, digits = 2) =>
    Number(v).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  const fmtPct = (v, digits = 1) => `${(v * 100).toFixed(digits)}%`;
  const fmtMoney = (v, digits = 0) =>
    `$${Number(v).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;

  const escapeHtml = (value) =>
    String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /* ------------------------------------------------------------------ *
   * Chart primitives (inline SVG, no dependencies)
   * ------------------------------------------------------------------ */

  // Sized for a full-width card so the SVG renders near 1:1 and its type lands
  // at its nominal size rather than being scaled up.
  const CW = 780;
  const CH = 300;
  const PAD = { top: 16, right: 18, bottom: 44, left: 60 };
  const PW = CW - PAD.left - PAD.right;
  const PH = CH - PAD.top - PAD.bottom;

  // One blue-forward ramp shared by every chart. Red is reserved for the
  // "bad" series (fraud, churn) and never used as a neutral category colour.
  const PALETTE = ["#2563eb", "#0ea5e9", "#6366f1", "#0f9f8a", "#e0334b", "#7c3aed"];

  let gradientSeq = 0;

  function lighten(hex, amount = 0.42) {
    const n = parseInt(hex.slice(1), 16);
    const mix = (channel) => Math.round(channel + (255 - channel) * amount);
    return `rgb(${mix((n >> 16) & 255)}, ${mix((n >> 8) & 255)}, ${mix(n & 255)})`;
  }

  /** Vertical light-to-base gradient per colour, so marks match the glass UI.
   *  Ordinal ramps pass a smaller `amount` so the gradient doesn't wash out the
   *  light end of the ramp into the plate behind it. */
  function gradientDefs(colors, amount = 0.42) {
    const unique = [...new Set(colors)];
    gradientSeq += 1;
    const ids = new Map();
    const defs = unique
      .map((color) => {
        const id = `sg${gradientSeq}-${color.replace("#", "")}`;
        ids.set(color, id);
        return `<linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${lighten(color, amount)}" /><stop offset="100%" stop-color="${color}" /></linearGradient>`;
      })
      .join("");
    return { markup: `<defs>${defs}</defs>`, fill: (color) => `url(#${ids.get(color)})` };
  }

  /** Keep the first and last x-axis labels inside the viewBox. */
  const anchorFor = (fraction) => (fraction === 0 ? "start" : fraction === 1 ? "end" : "middle");

  const tickX = (fraction) => {
    const x = PAD.left + PW * fraction;
    return fraction === 0 ? x - 14 : fraction === 1 ? x + 14 : x;
  };

  function niceTicks(min, max, count = 5) {
    if (min === max) {
      return [min - 1, min, min + 1];
    }
    const span = max - min;
    const rawStep = span / count;
    const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const normalized = rawStep / magnitude;
    const step = (normalized >= 5 ? 10 : normalized >= 2 ? 5 : normalized >= 1 ? 2 : 1) * magnitude;
    const start = Math.floor(min / step) * step;
    const ticks = [];
    for (let v = start; v <= max + step * 0.5; v += step) ticks.push(Number(v.toFixed(10)));
    return ticks;
  }

  function frame(yMin, yMax, ticks, yTitle, tickFormat) {
    const scale = (v) => PAD.top + PH * (1 - (v - yMin) / (yMax - yMin || 1));
    let svg = "";
    ticks.forEach((tick) => {
      const y = scale(tick);
      if (y < PAD.top - 1 || y > PAD.top + PH + 1) return;
      svg += `<line class="sim-grid-line" x1="${PAD.left}" y1="${y.toFixed(1)}" x2="${PAD.left + PW}" y2="${y.toFixed(1)}" />`;
      svg += `<text class="sim-axis-text" x="${PAD.left - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end">${escapeHtml(tickFormat(tick))}</text>`;
    });
    svg += `<line class="sim-axis-line" x1="${PAD.left}" y1="${PAD.top + PH}" x2="${PAD.left + PW}" y2="${PAD.top + PH}" />`;
    if (yTitle) {
      svg += `<text class="sim-axis-title" transform="translate(14 ${PAD.top + PH / 2}) rotate(-90)" text-anchor="middle">${escapeHtml(yTitle)}</text>`;
    }
    return { svg, scale };
  }

  const wrap = (inner) =>
    `<svg viewBox="0 0 ${CW} ${CH}" role="img" preserveAspectRatio="xMidYMid meet">${inner}</svg>`;

  /** `colors` (one per bar) is for ordinal categories only — an ordered scale
   *  earns a colour ramp; unordered categories share the primary blue. */
  function barChart({ labels, values, color = PALETTE[0], colors, yTitle, tickFormat = (v) => fmtNum(v, 0), valueFormat }) {
    const max = Math.max(...values, 0);
    const min = Math.min(...values, 0);
    const ticks = niceTicks(min, max || 1, 4);
    const yMin = Math.min(ticks[0], min);
    const yMax = Math.max(ticks[ticks.length - 1], max);
    const base = frame(yMin, yMax, ticks, yTitle, tickFormat);
    const slot = PW / labels.length;
    // Bar-to-band ratio around 0.66 — thinner than this and categorical bars
    // read as isolated pillars floating in the plate.
    const width = Math.min(132, slot * 0.66);
    const barColor = (i) => (colors ? colors[i % colors.length] : color);
    const grad = gradientDefs(labels.map((_, i) => barColor(i)), colors ? 0.16 : 0.42);
    let svg = grad.markup + base.svg;
    const zero = base.scale(Math.max(yMin, Math.min(0, yMax)));

    labels.forEach((label, i) => {
      const x = PAD.left + slot * i + (slot - width) / 2;
      const y = base.scale(values[i]);
      const top = Math.min(y, zero);
      const height = Math.max(1, Math.abs(zero - y));
      svg += `<rect class="sim-bar" x="${x.toFixed(1)}" y="${top.toFixed(1)}" width="${width.toFixed(1)}" height="${height.toFixed(1)}" rx="6" fill="${grad.fill(barColor(i))}" />`;
      if (valueFormat) {
        svg += `<text class="sim-value-text" x="${(x + width / 2).toFixed(1)}" y="${(top - 6).toFixed(1)}" text-anchor="middle">${escapeHtml(valueFormat(values[i]))}</text>`;
      }
      svg += `<text class="sim-axis-text" x="${(x + width / 2).toFixed(1)}" y="${PAD.top + PH + 20}" text-anchor="middle">${escapeHtml(label)}</text>`;
    });

    return wrap(svg);
  }

  function groupedBarChart({ labels, series, yTitle, tickFormat = (v) => fmtNum(v, 0), valueFormat }) {
    const all = series.flatMap((s) => s.values);
    const max = Math.max(...all, 0);
    const min = Math.min(...all, 0);
    const ticks = niceTicks(min, max || 1, 4);
    const yMin = Math.min(ticks[0], min);
    const yMax = Math.max(ticks[ticks.length - 1], max);
    const base = frame(yMin, yMax, ticks, yTitle, tickFormat);
    const slot = PW / labels.length;
    const groupWidth = Math.min(230, slot * 0.7);
    const width = groupWidth / series.length;
    const zero = base.scale(Math.max(yMin, Math.min(0, yMax)));
    const grad = gradientDefs(series.map((s, k) => s.color || PALETTE[k]));
    let svg = grad.markup + base.svg;

    labels.forEach((label, i) => {
      const startX = PAD.left + slot * i + (slot - groupWidth) / 2;
      series.forEach((s, k) => {
        const x = startX + width * k;
        const y = base.scale(s.values[i]);
        const top = Math.min(y, zero);
        const height = Math.max(1, Math.abs(zero - y));
        svg += `<rect class="sim-bar" x="${x.toFixed(1)}" y="${top.toFixed(1)}" width="${(width * 0.86).toFixed(1)}" height="${height.toFixed(1)}" rx="6" fill="${grad.fill(s.color || PALETTE[k])}" />`;
        if (valueFormat) {
          svg += `<text class="sim-value-text" x="${(x + width * 0.43).toFixed(1)}" y="${(top - 6).toFixed(1)}" text-anchor="middle">${escapeHtml(valueFormat(s.values[i]))}</text>`;
        }
      });
      svg += `<text class="sim-axis-text" x="${(startX + groupWidth / 2).toFixed(1)}" y="${PAD.top + PH + 20}" text-anchor="middle">${escapeHtml(label)}</text>`;
    });

    return wrap(svg);
  }

  function histogram({ values, bins = 26, color = PALETTE[0], xTitle, yTitle = "Count" }) {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const width = (max - min) / bins || 1;
    const counts = new Array(bins).fill(0);
    values.forEach((v) => {
      const idx = Math.min(bins - 1, Math.max(0, Math.floor((v - min) / width)));
      counts[idx] += 1;
    });

    const ticks = niceTicks(0, Math.max(...counts), 4);
    const base = frame(0, ticks[ticks.length - 1], ticks, yTitle, (v) => fmtInt(v));
    const slot = PW / bins;
    const grad = gradientDefs([color]);
    let svg = grad.markup + base.svg;

    counts.forEach((count, i) => {
      const x = PAD.left + slot * i;
      const y = base.scale(count);
      svg += `<rect class="sim-bar" x="${(x + 0.6).toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(1, slot - 1.4).toFixed(1)}" height="${(PAD.top + PH - y).toFixed(1)}" rx="3" fill="${grad.fill(color)}" />`;
    });

    [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
      svg += `<text class="sim-axis-text" x="${tickX(f).toFixed(1)}" y="${PAD.top + PH + 20}" text-anchor="${anchorFor(f)}">${escapeHtml(fmtNum(min + (max - min) * f, max - min > 20 ? 0 : 2))}</text>`;
    });
    if (xTitle) {
      svg += `<text class="sim-axis-title" x="${PAD.left + PW / 2}" y="${CH - 6}" text-anchor="middle">${escapeHtml(xTitle)}</text>`;
    }

    return wrap(svg);
  }

  function lineChart({ series, xTitle, yTitle, tickFormat = (v) => fmtNum(v, 0), xFormat = (v) => fmtNum(v, 0) }) {
    const allY = series.flatMap((s) => s.points.map((p) => p.y));
    const allX = series.flatMap((s) => s.points.map((p) => p.x));
    const yTicks = niceTicks(Math.min(...allY), Math.max(...allY), 4);
    const yMin = yTicks[0];
    const yMax = yTicks[yTicks.length - 1];
    const xMin = Math.min(...allX);
    const xMax = Math.max(...allX);
    const base = frame(yMin, yMax, yTicks, yTitle, tickFormat);
    const sx = (v) => PAD.left + PW * ((v - xMin) / (xMax - xMin || 1));
    let svg = base.svg;

    series.forEach((s, i) => {
      const color = s.color || PALETTE[i];
      const d = s.points
        .map((p, idx) => `${idx === 0 ? "M" : "L"}${sx(p.x).toFixed(1)} ${base.scale(p.y).toFixed(1)}`)
        .join(" ");
      svg += `<path class="sim-line" d="${d}" fill="none" stroke="${color}" stroke-width="${s.width || 2.4}" stroke-linejoin="round" stroke-linecap="round" ${s.dashed ? 'stroke-dasharray="5 4"' : ""} opacity="${s.opacity || 1}" />`;
      if (s.marker) {
        const p = s.marker;
        svg += `<circle cx="${sx(p.x).toFixed(1)}" cy="${base.scale(p.y).toFixed(1)}" r="5" fill="${color}" stroke="#fff" stroke-width="2" />`;
      }
    });

    [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
      svg += `<text class="sim-axis-text" x="${tickX(f).toFixed(1)}" y="${PAD.top + PH + 20}" text-anchor="${anchorFor(f)}">${escapeHtml(xFormat(xMin + (xMax - xMin) * f))}</text>`;
    });
    if (xTitle) {
      svg += `<text class="sim-axis-title" x="${PAD.left + PW / 2}" y="${CH - 6}" text-anchor="middle">${escapeHtml(xTitle)}</text>`;
    }

    return wrap(svg);
  }

  function scatterChart({ groups, xTitle, yTitle, maxPoints = 700, line }) {
    const allX = groups.flatMap((g) => g.points.map((p) => p.x));
    const allY = groups.flatMap((g) => g.points.map((p) => p.y));
    const yTicks = niceTicks(Math.min(...allY), Math.max(...allY), 4);
    const base = frame(yTicks[0], yTicks[yTicks.length - 1], yTicks, yTitle, (v) => fmtNum(v, Math.abs(v) > 20 ? 0 : 1));
    const xMin = Math.min(...allX);
    const xMax = Math.max(...allX);
    const sx = (v) => PAD.left + PW * ((v - xMin) / (xMax - xMin || 1));
    let svg = base.svg;
    const perGroup = Math.max(40, Math.floor(maxPoints / groups.length));

    groups.forEach((g, i) => {
      const color = g.color || PALETTE[i];
      const stride = Math.max(1, Math.ceil(g.points.length / perGroup));
      for (let idx = 0; idx < g.points.length; idx += stride) {
        const p = g.points[idx];
        svg += `<circle cx="${sx(p.x).toFixed(1)}" cy="${base.scale(p.y).toFixed(1)}" r="${g.radius || 2.6}" fill="${color}" opacity="${g.opacity || 0.5}" />`;
      }
    });

    if (line) {
      const d = line.points
        .map((p, idx) => `${idx === 0 ? "M" : "L"}${sx(p.x).toFixed(1)} ${base.scale(p.y).toFixed(1)}`)
        .join(" ");
      svg += `<path d="${d}" fill="none" stroke="${line.color || "#131a2b"}" stroke-width="2.4" stroke-linecap="round" />`;
    }

    [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
      svg += `<text class="sim-axis-text" x="${tickX(f).toFixed(1)}" y="${PAD.top + PH + 20}" text-anchor="${anchorFor(f)}">${escapeHtml(fmtNum(xMin + (xMax - xMin) * f, xMax - xMin > 20 ? 0 : 2))}</text>`;
    });
    if (xTitle) {
      svg += `<text class="sim-axis-title" x="${PAD.left + PW / 2}" y="${CH - 6}" text-anchor="middle">${escapeHtml(xTitle)}</text>`;
    }

    return wrap(svg);
  }

  const legend = (items) =>
    `<div class="sim-legend">${items
      .map((item) => `<span><i style="background:${item.color}"></i>${escapeHtml(item.label)}</span>`)
      .join("")}</div>`;

  /* ------------------------------------------------------------------ *
   * Cases — one per notebook
   * ------------------------------------------------------------------ */

  const CASES = [
    {
      id: "ch1",
      chapter: 1,
      notebook: "Chapter 1.ipynb",
      track: "Foundations",
      company: "RetailCo",
      title: "Data Science in 1995 vs 2025",
      focus: "Small hand-collected retail table",
      methods: ["Descriptive stats", "Poisson demand", "Categorical summaries"],
      scenario:
        "A retail analyst in 1995 works from a narrow transaction table: five products, four regions, daily unit counts and a paper feedback slip. The case asks what decisions that table can and cannot support.",
      controls: [
        { key: "rows", label: "Daily records", min: 100, max: 1200, step: 50, value: 500 },
        { key: "lam", label: "Mean units per day", min: 4, max: 45, step: 1, value: 20 },
      ],
      run(p, rng) {
        const products = ["Product_1", "Product_2", "Product_3", "Product_4", "Product_5"];
        const regions = ["North", "South", "East", "West"];
        const prices = [5.99, 6.99, 7.99, 8.99];
        const feedbackLevels = ["Poor", "Average", "Good", "Excellent"];

        const rows = [];
        for (let i = 0; i < p.rows; i += 1) {
          const units = rng.poisson(p.lam);
          const price = rng.choice(prices);
          rows.push({
            product: rng.choice(products),
            region: rng.choice(regions),
            units,
            price,
            revenue: units * price,
            feedback: rng.choice(feedbackLevels),
          });
        }

        const byRegion = regions.map((r) => mean(rows.filter((row) => row.region === r).map((row) => row.units)));
        const feedbackCounts = feedbackLevels.map((f) => rows.filter((row) => row.feedback === f).length);
        const revenue = sum(rows.map((row) => row.revenue));
        const bestRegionIdx = byRegion.indexOf(Math.max(...byRegion));

        return {
          metrics: [
            { label: "Total revenue", value: fmtMoney(revenue), hint: `${fmtInt(p.rows)} daily records` },
            { label: "Mean units per record", value: fmtNum(mean(rows.map((r) => r.units)), 1) },
            { label: "Strongest region", value: regions[bestRegionIdx], hint: `${fmtNum(byRegion[bestRegionIdx], 1)} units/day`, tone: "accent" },
            {
              label: "Excellent feedback",
              value: fmtPct(feedbackCounts[3] / p.rows),
              hint: "Free-text era: no sentiment scoring",
            },
          ],
          charts: [
            {
              title: "Average units sold by region",
              note: "Region is drawn uniformly, so gaps here are sampling noise — the 1995 analyst had no way to tell.",
              svg: barChart({ labels: regions, values: byRegion, color: PALETTE[0], yTitle: "Units / record", valueFormat: (v) => fmtNum(v, 1) }),
            },
            {
              title: "Customer feedback distribution",
              note: "Poor to Excellent is ordered, so the bars carry a light-to-dark ramp rather than four unrelated colours.",
              svg: barChart({
                labels: feedbackLevels,
                values: feedbackCounts,
                colors: ["#8fb0ea", "#5f8ade", "#3f66c4", "#274a93"],
                yTitle: "Records",
                valueFormat: fmtInt,
              }),
            },
          ],
          insight: `Every region and feedback level is sampled uniformly, so the "best" region shifts each time you draw a new sample. Increase daily records and the four bars flatten toward ${fmtNum(p.lam, 0)} units — the chapter's point about how thin 1995 data really was.`,
          table: {
            columns: ["Product", "Region", "Units", "Price", "Feedback"],
            rows: rows.slice(0, 6).map((r) => [r.product, r.region, fmtInt(r.units), fmtMoney(r.price, 2), r.feedback]),
          },
        };
      },
    },

    {
      id: "ch2",
      chapter: 2,
      notebook: "Chapter 2.ipynb",
      track: "Foundations",
      company: "ShopSmart",
      title: "Cleaning and Analyzing Customer Purchase Data",
      focus: "Messy pipeline, churn flags, high-value customers",
      methods: ["Missing data", "Feature engineering", "Segmentation"],
      scenario:
        "ShopSmart's pipeline is messy: incomplete demographics, unstandardized text and missing email responses. The task is to clean, engineer features and answer who is high-value and who is about to churn.",
      controls: [
        { key: "customers", label: "Customers", min: 200, max: 4000, step: 100, value: 1000 },
        { key: "spendScale", label: "Spend scale (exponential)", min: 150, max: 900, step: 25, value: 500, hint: "np.random.exponential(scale=…)" },
        { key: "churnWindow", label: "Churn window (days)", min: 60, max: 330, step: 10, value: 180 },
      ],
      run(p, rng) {
        const spend = [];
        const orders = [];
        const daysAgo = [];
        const ages = [];
        let missingEmail = 0;
        const rows = [];

        for (let i = 0; i < p.customers; i += 1) {
          const totalSpent = rng.exponential(p.spendScale);
          const numOrders = rng.poisson(5);
          const last = rng.randint(1, 365);
          const age = rng.randint(18, 70);
          const emailDraw = rng.next();
          const responded = emailDraw < 0.4 ? 0 : emailDraw < 0.9 ? 1 : null;
          if (responded === null) missingEmail += 1;
          spend.push(totalSpent);
          orders.push(numOrders);
          daysAgo.push(last);
          ages.push(age);
          if (rows.length < 6) {
            rows.push([
              String(i + 1),
              String(age),
              fmtMoney(totalSpent, 2),
              String(numOrders),
              String(last),
              responded === null ? "missing" : String(responded),
            ]);
          }
        }

        const churned = daysAgo.map((d) => (d > p.churnWindow ? 1 : 0));
        const churnRate = mean(churned);
        const highValueCut = quantile(spend, 0.8);
        const churnedSpend = spend.filter((_, i) => churned[i] === 1);
        const retainedSpend = spend.filter((_, i) => churned[i] === 0);

        return {
          metrics: [
            { label: "Churn flag rate", value: fmtPct(churnRate), hint: `No order in ${p.churnWindow}+ days`, tone: churnRate > 0.5 ? "bad" : undefined },
            { label: "Median spend", value: fmtMoney(median(spend)), hint: `Mean ${fmtMoney(mean(spend))}` },
            { label: "High-value threshold", value: fmtMoney(highValueCut), hint: "Top 20% of spend", tone: "accent" },
            { label: "Missing email response", value: fmtPct(missingEmail / p.customers), hint: "Filled with 0 before modelling" },
          ],
          charts: [
            {
              title: "Total spend distribution",
              note: "Exponential spend produces the long right tail that makes the mean a poor summary of a typical customer.",
              svg: histogram({ values: spend, bins: 30, color: PALETTE[0], xTitle: "Total spent ($)" }),
            },
            {
              title: "Mean spend by churn status",
              note: "Churn is defined purely from recency, so spend barely separates the groups — the notebook's caution against reading a boxplot as causal.",
              svg: groupedBarChart({
                labels: ["Retained", "Churned"],
                series: [{ name: "Mean spend", color: PALETTE[0], values: [mean(retainedSpend), mean(churnedSpend)] }],
                yTitle: "Mean spend ($)",
                valueFormat: (v) => fmtMoney(v),
              }),
            },
          ],
          insight: `The churn flag is a recency rule, not a model: widening the window to ${p.churnWindow} days sets churn at ${fmtPct(churnRate)} directly, because <code>last_order_days_ago</code> is uniform over the year. Spend is independent of it, which is why the two bars stay level however you tune the sample.`,
          table: {
            columns: ["ID", "Age", "Total spent", "Orders", "Days ago", "Email"],
            rows,
          },
        };
      },
    },

    {
      id: "ch3",
      chapter: 3,
      notebook: "Chapter 3.ipynb",
      track: "Machine Learning",
      company: "PriceWise",
      title: "Optimizing Product Pricing with Machine Learning",
      focus: "Price regression and product clustering",
      methods: ["Linear regression", "K-means", "Model evaluation"],
      scenario:
        "PriceWise wants a model that suggests a list price from product features, plus behavioural clusters for A/B testing and promotion targeting.",
      controls: [
        { key: "products", label: "Products", min: 200, max: 2500, step: 100, value: 1000 },
        { key: "priceNoise", label: "Price noise (sd)", min: 3, max: 30, step: 1, value: 15 },
        { key: "signal", label: "Feature signal strength", min: 0, max: 1, step: 0.05, value: 0, hint: "0 reproduces the notebook, where price is drawn independently" },
        { key: "clusters", label: "K-means clusters", min: 2, max: 6, step: 1, value: 4 },
      ],
      run(p, rng) {
        const categories = ["Electronics", "Home", "Apparel", "Toys"];
        const X = [];
        const price = [];
        const rows = [];

        for (let i = 0; i < p.products; i += 1) {
          const catIdx = rng.randint(0, 4);
          const rating = Math.min(5, Math.max(1, rng.normal(4, 0.5)));
          const reviews = rng.poisson(50);
          const units = rng.poisson(30);
          const discount = rng.randint(0, 2);
          const daysInStock = rng.randint(1, 365);
          const signalTerm =
            p.signal * (6 * (rating - 4) + 0.18 * (reviews - 50) - 0.22 * (units - 30) - 9 * discount);
          const listPrice = rng.normal(50, p.priceNoise) + signalTerm;
          X.push([rating, reviews, units, discount, daysInStock, catIdx]);
          price.push(listPrice);
          if (rows.length < 6) {
            rows.push([
              categories[catIdx],
              fmtNum(rating, 2),
              String(reviews),
              String(units),
              discount ? "yes" : "no",
              fmtMoney(listPrice, 2),
            ]);
          }
        }

        const fit = ols(X, price);
        const scaled = standardize(X.map((r) => [r[0], r[1], r[2], r[4]]));
        const clustering = kmeans(scaled.Z, p.clusters, rng);

        const scatterPoints = price.map((actual, i) => ({ x: actual, y: fit.fitted[i] }));
        const lo = Math.min(...price);
        const hi = Math.max(...price);

        return {
          metrics: [
            { label: "R² (held-in)", value: fmtNum(fit.r2, 3), tone: fit.r2 > 0.3 ? "good" : "bad", hint: p.signal === 0 ? "Price is independent noise" : "Signal injected" },
            { label: "RMSE", value: fmtMoney(fit.rmse, 2), hint: `Price sd ${fmtMoney(sd(price), 2)}` },
            { label: "Largest cluster", value: fmtInt(Math.max(...clustering.sizes)), hint: `of ${fmtInt(p.products)} products` },
            { label: "Cluster balance", value: fmtNum(Math.min(...clustering.sizes) / Math.max(...clustering.sizes), 2), hint: "smallest ÷ largest", tone: "accent" },
          ],
          charts: [
            {
              title: "Predicted vs actual price",
              note: "A perfect model sits on the diagonal. Raise feature signal strength above zero to watch the cloud collapse onto it.",
              svg: scatterChart({
                groups: [{ name: "Products", color: PALETTE[0], points: scatterPoints, opacity: 0.4 }],
                line: { points: [{ x: lo, y: lo }, { x: hi, y: hi }], color: "#131a2b" },
                xTitle: "Actual price ($)",
                yTitle: "Predicted ($)",
              }),
              legendItems: [{ label: "Products", color: PALETTE[0] }, { label: "Perfect prediction", color: "#131a2b" }],
            },
            {
              title: "K-means cluster sizes",
              svg: barChart({
                labels: clustering.sizes.map((_, i) => `Cluster ${i + 1}`),
                values: clustering.sizes,
                color: PALETTE[0],
                yTitle: "Products",
                valueFormat: fmtInt,
              }),
            },
          ],
          insight:
            p.signal === 0
              ? `Exactly as written, the notebook draws <code>price</code> from its own normal distribution, independent of every feature — so R² sits near ${fmtNum(fit.r2, 3)} no matter how many products you simulate. That is a useful negative result: the workflow runs cleanly and still learns nothing. Push feature signal strength up to make the features genuinely predictive.`
              : `With signal at ${fmtNum(p.signal, 2)}, rating, reviews, units and discount now move price, and R² climbs to ${fmtNum(fit.r2, 3)}. Adding products tightens the estimate; adding price noise buries the signal again.`,
          table: {
            columns: ["Category", "Rating", "Reviews", "Units sold", "Discount", "Price"],
            rows,
          },
        };
      },
    },

    {
      id: "ch4",
      chapter: 4,
      notebook: "Chapter 4.ipynb",
      track: "Statistics",
      company: "PromoImpact",
      title: "Evaluating a Marketing Campaign",
      focus: "Did the promotion actually lift spending?",
      methods: ["Hypothesis testing", "Confidence intervals", "Regression"],
      scenario:
        "PromoImpact sent a targeted email promotion to half of its customer base. Management wants to know whether the lift is real, how large it is, and whether the campaign should continue.",
      controls: [
        { key: "customers", label: "Customers", min: 100, max: 5000, step: 100, value: 1000 },
        { key: "lift", label: "True promo effect ($)", min: 0, max: 60, step: 1, value: 20 },
        { key: "noise", label: "Spending noise (sd)", min: 5, max: 80, step: 1, value: 25 },
      ],
      run(p, rng) {
        const control = [];
        const promo = [];
        const rows = [];

        for (let i = 0; i < p.customers; i += 1) {
          const age = Math.round(rng.normal(40, 12));
          const income = Math.round(rng.normal(60000, 15000) / 100) * 100;
          const prior = rng.poisson(5);
          const group = rng.randint(0, 2);
          const spending = 50 + prior * 10 + group * p.lift + rng.normal(0, p.noise);
          (group === 1 ? promo : control).push(spending);
          if (rows.length < 6) {
            rows.push([String(age), fmtMoney(income), String(prior), group ? "promo" : "control", fmtMoney(spending, 2)]);
          }
        }

        const test = welchTest(control, promo);
        const significant = test.p < 0.05;

        return {
          metrics: [
            { label: "Control mean", value: fmtMoney(test.meanA, 2), hint: `n = ${fmtInt(control.length)}` },
            { label: "Promo mean", value: fmtMoney(test.meanB, 2), hint: `n = ${fmtInt(promo.length)}` },
            { label: "Estimated lift", value: fmtMoney(test.diff, 2), hint: `95% CI ${fmtMoney(test.ciLow, 2)} to ${fmtMoney(test.ciHigh, 2)}`, tone: "accent" },
            {
              label: "Two-sided p",
              value: test.p < 0.001 ? "< 0.001" : fmtNum(test.p, 3),
              hint: significant ? "Reject the null at 5%" : "Cannot reject the null",
              tone: significant ? "good" : "bad",
            },
          ],
          charts: [
            {
              title: "Mean spending by group",
              note: "Bars show group means; the estimated lift and its interval are in the metrics above.",
              svg: groupedBarChart({
                labels: ["Control", "Promo"],
                series: [{ name: "Mean spend", color: PALETTE[0], values: [test.meanA, test.meanB] }],
                yTitle: "Mean spend ($)",
                valueFormat: (v) => fmtMoney(v, 2),
              }),
            },
            {
              title: "Spending distribution, promo group",
              svg: histogram({ values: promo, bins: 28, color: PALETTE[0], xTitle: "Spending ($)" }),
            },
          ],
          insight: `The true effect is fixed at ${fmtMoney(p.lift)}; the test recovers ${fmtMoney(test.diff, 2)}. Drop the sample toward 100 customers or raise noise past 60 and a real effect stops being detectable — the difference between "no effect" and "not enough data" that the chapter is built around.`,
          table: {
            columns: ["Age", "Income", "Prior purchases", "Group", "Spending"],
            rows,
          },
        };
      },
    },

    {
      id: "ch5",
      chapter: 5,
      notebook: "Chapter 5.ipynb",
      track: "Analytics Tools",
      company: "PayGuard",
      title: "Detecting Transaction Fraud",
      focus: "Risk scoring and an analyst review queue",
      methods: ["Feature engineering", "Risk scoring", "Precision / recall"],
      scenario:
        "PayGuard sees a spike in fraud but has no automatic triage. The case builds an early-warning score and then asks where to set the review threshold given a limited analyst team.",
      controls: [
        { key: "transactions", label: "Transactions", min: 500, max: 9000, step: 250, value: 5000 },
        { key: "amountScale", label: "Amount scale ($)", min: 50, max: 600, step: 10, value: 200 },
        { key: "threshold", label: "Review threshold", min: 0.05, max: 0.9, step: 0.01, value: 0.35, format: (v) => fmtNum(v, 2) },
      ],
      run(p, rng) {
        const scores = [];
        const labels = [];
        const amounts = [];
        const rows = [];

        for (let i = 0; i < p.transactions; i += 1) {
          const amount = rng.exponential(p.amountScale);
          const deviceTrust = rng.uniform(0, 1);
          const distance = rng.exponential(50);
          const repeat = rng.next() < 0.4 ? 1 : 0;
          const hour = rng.randint(0, 24);
          const previousFraud = rng.next() < 0.1 ? 1 : 0;

          const risk = Math.min(
            0.95,
            0.05 +
              0.2 * (amount > 500 ? 1 : 0) +
              0.3 * (deviceTrust < 0.2 ? 1 : 0) +
              0.2 * (distance > 100 ? 1 : 0) +
              0.15 * (repeat === 0 ? 1 : 0) +
              0.3 * previousFraud
          );
          const fraud = rng.bernoulli(risk);

          scores.push(risk);
          labels.push(fraud);
          amounts.push(amount);
          if (rows.length < 6) {
            rows.push([
              fmtMoney(amount, 2),
              fmtNum(deviceTrust, 2),
              `${fmtNum(distance, 0)} km`,
              repeat ? "yes" : "no",
              String(hour),
              fmtNum(risk, 2),
              fraud ? "fraud" : "clean",
            ]);
          }
        }

        const cm = confusion(scores, labels, p.threshold);
        const buckets = ["<100", "100–300", "300–500", "500–1k", "1k+"];
        const bucketOf = (a) => (a < 100 ? 0 : a < 300 ? 1 : a < 500 ? 2 : a < 1000 ? 3 : 4);
        const bucketRates = buckets.map((_, b) => {
          const idx = amounts.map((a, i) => (bucketOf(a) === b ? i : -1)).filter((i) => i >= 0);
          return idx.length ? mean(idx.map((i) => labels[i])) : 0;
        });

        const sweep = [];
        for (let t = 0.05; t <= 0.9; t += 0.025) {
          const c = confusion(scores, labels, t);
          sweep.push({ t, precision: c.precision, recall: c.recall });
        }

        return {
          metrics: [
            { label: "Fraud rate", value: fmtPct(mean(labels)), hint: `${fmtInt(sum(labels))} fraudulent` },
            { label: "Sent to review", value: fmtPct(cm.flagged / p.transactions), hint: `${fmtInt(cm.flagged)} transactions`, tone: "accent" },
            { label: "Precision", value: fmtPct(cm.precision), hint: "Share of reviewed that are fraud", tone: cm.precision > 0.5 ? "good" : undefined },
            { label: "Recall", value: fmtPct(cm.recall), hint: "Share of fraud that is caught", tone: cm.recall < 0.5 ? "bad" : "good" },
          ],
          charts: [
            {
              title: "Precision and recall across thresholds",
              note: `The marker sits at your current threshold of ${fmtNum(p.threshold, 2)}.`,
              svg: lineChart({
                series: [
                  { name: "Precision", color: PALETTE[0], points: sweep.map((s) => ({ x: s.t, y: s.precision })), marker: { x: p.threshold, y: cm.precision } },
                  { name: "Recall", color: PALETTE[1], points: sweep.map((s) => ({ x: s.t, y: s.recall })), marker: { x: p.threshold, y: cm.recall } },
                ],
                xTitle: "Review threshold",
                yTitle: "Rate",
                tickFormat: (v) => fmtPct(v, 0),
                xFormat: (v) => fmtNum(v, 2),
              }),
              legendItems: [{ label: "Precision", color: PALETTE[0] }, { label: "Recall", color: PALETTE[1] }],
            },
            {
              title: "Fraud rate by transaction size",
              svg: barChart({ labels: buckets, values: bucketRates, color: PALETTE[4], yTitle: "Fraud rate", tickFormat: (v) => fmtPct(v, 0), valueFormat: (v) => fmtPct(v, 0) }),
            },
          ],
          insight: `The score here <em>is</em> the generating probability, so this is the best any model could do. Even then, moving the threshold trades precision against recall one-for-one: at ${fmtNum(p.threshold, 2)} the queue holds ${fmtInt(cm.flagged)} transactions and misses ${fmtInt(cm.fn)} frauds. That trade — not model accuracy — is the prescriptive decision.`,
          table: {
            columns: ["Amount", "Device trust", "Distance", "Repeat", "Hour", "Risk", "Label"],
            rows,
          },
        };
      },
    },

    {
      id: "ch6",
      chapter: 6,
      notebook: "Chapter 6.ipynb",
      track: "Machine Learning",
      company: "RetailIQ",
      title: "Demand Forecasting and Dynamic Pricing",
      focus: "Product × day panel with promotions",
      methods: ["Panel simulation", "Poisson demand", "Promo evaluation"],
      scenario:
        "RetailIQ needs a demand model across a product catalogue, segments for targeting, and a pricing policy it can evaluate before rollout.",
      controls: [
        { key: "products", label: "Products", min: 10, max: 100, step: 5, value: 60 },
        { key: "days", label: "Days", min: 20, max: 150, step: 10, value: 90 },
        { key: "promoRate", label: "Promotion rate", min: 0, max: 0.6, step: 0.02, value: 0.18, format: (v) => fmtPct(v, 0) },
        { key: "priceCoef", label: "Price sensitivity", min: 0, max: 0.03, step: 0.001, value: 0.01, format: (v) => fmtNum(v, 3) },
      ],
      run(p, rng) {
        const categories = ["Electronics", "Home", "Beauty", "Apparel", "Sports"];
        const catProbs = [0.25, 0.2, 0.2, 0.2, 0.15];
        const catalogue = [];
        for (let i = 0; i < p.products; i += 1) {
          catalogue.push({
            category: rng.choice(categories, catProbs),
            basePrice: rng.uniform(10, 200),
            quality: rng.uniform(0.4, 0.95),
            tightness: rng.beta(2, 5),
          });
        }

        const dailyUnits = new Array(p.days).fill(0);
        const promoUnits = [];
        const plainUnits = [];
        const byCategory = new Map(categories.map((c) => [c, 0]));
        let revenue = 0;
        let totalUnits = 0;
        const rows = [];

        for (let i = 0; i < p.products; i += 1) {
          const product = catalogue[i];
          for (let d = 0; d < p.days; d += 1) {
            const promo = rng.bernoulli(p.promoRate);
            const adSpend = rng.gamma(2, 20);
            const price = Math.max(5, product.basePrice * (1 - 0.2 * promo) * (1 + 0.05 * rng.normal(0, 1)));
            const eta =
              1.2 * product.quality +
              0.0009 * adSpend +
              0.25 * promo -
              p.priceCoef * price -
              0.35 * product.tightness;
            const mu = Math.min(300, Math.max(0.1, Math.exp(2.1 + eta)));
            const units = rng.poisson(mu);

            dailyUnits[d] += units;
            totalUnits += units;
            revenue += units * price;
            byCategory.set(product.category, byCategory.get(product.category) + units);
            (promo ? promoUnits : plainUnits).push(units);

            if (rows.length < 6 && d < 2) {
              rows.push([product.category, fmtMoney(product.basePrice, 2), fmtMoney(price, 2), promo ? "yes" : "no", fmtMoney(adSpend, 0), String(units)]);
            }
          }
        }

        const promoLift = mean(plainUnits) ? mean(promoUnits) / mean(plainUnits) - 1 : 0;
        const naiveMae = mean(dailyUnits.slice(1).map((v, i) => Math.abs(v - dailyUnits[i])));

        return {
          metrics: [
            { label: "Units sold", value: fmtInt(totalUnits), hint: `${fmtInt(p.products * p.days)} product-days` },
            { label: "Revenue", value: fmtMoney(revenue), hint: `${fmtMoney(revenue / Math.max(1, p.days))} / day` },
            { label: "Promotion lift", value: fmtPct(promoLift), hint: "Mean units, promo vs not", tone: promoLift > 0 ? "good" : "bad" },
            { label: "Naive forecast MAE", value: fmtNum(naiveMae, 1), hint: "Yesterday-as-forecast baseline", tone: "accent" },
          ],
          charts: [
            {
              title: "Total units per day",
              note: "Demand is Poisson around an exponential mean, so the series is noisy but stationary — a hard baseline for any forecaster to beat.",
              svg: lineChart({
                series: [{ name: "Units", color: PALETTE[0], points: dailyUnits.map((v, i) => ({ x: i + 1, y: v })) }],
                xTitle: "Day",
                yTitle: "Units",
              }),
            },
            {
              title: "Units by category",
              svg: barChart({
                labels: categories,
                values: categories.map((c) => byCategory.get(c)),
                color: PALETTE[0],
                yTitle: "Units",
                valueFormat: fmtInt,
              }),
            },
          ],
          insight: `Price enters demand through <code>exp(2.1 + η)</code>, so raising price sensitivity to ${fmtNum(p.priceCoef, 3)} compounds: a $200 product loses far more volume than a $20 one. That multiplicative structure is what makes a single across-the-board discount the wrong policy.`,
          table: {
            columns: ["Category", "Base price", "Realized price", "Promo", "Ad spend", "Units"],
            rows,
          },
        };
      },
    },

    {
      id: "ch7",
      chapter: 7,
      notebook: "Chapter 7.ipynb",
      track: "NLP",
      company: "ShopSmart",
      title: "Sentiment Analysis of Customer Reviews",
      focus: "Noisy text, ambiguous labels",
      methods: ["Text cleaning", "Lexicon classifier", "Class balance"],
      scenario:
        "ShopSmart collects thousands of reviews a day. A sentiment pipeline should route complaints to support quickly — but the training labels themselves are partly ambiguous.",
      controls: [
        { key: "reviews", label: "Reviews", min: 40, max: 800, step: 20, value: 120 },
        { key: "positiveBias", label: "Positive share", min: 0.2, max: 0.9, step: 0.05, value: 0.6, format: (v) => fmtPct(v, 0) },
        { key: "labelNoise", label: "Ambiguous label rate", min: 0, max: 0.5, step: 0.05, value: 0.15, format: (v) => fmtPct(v, 0) },
      ],
      run(p, rng) {
        const positive = [
          "This product is excellent, I love it!",
          "Very satisfied with the quality and service.",
          "Exceeded my expectations, totally worth the price.",
          "Great customer support and fast delivery.",
          "Amazing build quality. Highly recommend.",
        ];
        const negative = [
          "This is terrible. It broke after one use.",
          "Very disappointed. Not worth the money.",
          "Customer service was unhelpful and rude.",
          "Shipping was delayed and product was damaged.",
          "Worst purchase I've ever made.",
        ];
        const neutral = ["It's okay, nothing special.", "Average quality for the price.", "The product is fine, but delivery took too long."];

        const positiveWords = ["excellent", "love", "satisfied", "quality", "exceeded", "worth", "great", "fast", "amazing", "recommend", "fine"];
        const negativeWords = ["terrible", "broke", "disappointed", "not worth", "unhelpful", "rude", "delayed", "damaged", "worst", "too long"];

        const texts = [];
        const labels = [];
        const rows = [];

        for (let i = 0; i < p.reviews; i += 1) {
          let text;
          let label;
          const draw = rng.next();
          if (draw < p.positiveBias) {
            text = rng.choice(positive);
            label = 1;
          } else if (draw < p.positiveBias + (1 - p.positiveBias) * 0.7) {
            text = rng.choice(negative);
            label = 0;
          } else {
            text = rng.choice(neutral);
            label = rng.randint(0, 2);
          }
          if (rng.next() < p.labelNoise) label = 1 - label;
          if (rng.next() < 0.3) text = text.toUpperCase();
          if (rng.next() < 0.2) text = `  ${text}  `;
          texts.push(text);
          labels.push(label);
        }

        const predictions = texts.map((text) => {
          const lower = text.toLowerCase();
          const pos = positiveWords.filter((w) => lower.includes(w)).length;
          const neg = negativeWords.filter((w) => lower.includes(w)).length;
          return pos >= neg ? 1 : 0;
        });

        predictions.slice(0, 6).forEach((pred, i) => {
          const clean = texts[i].trim();
          rows.push([clean.length > 46 ? `${clean.slice(0, 46)}…` : clean, labels[i] ? "positive" : "negative", pred ? "positive" : "negative", labels[i] === pred ? "✓" : "✗"]);
        });

        let tp = 0;
        let fp = 0;
        let fn = 0;
        let correct = 0;
        predictions.forEach((pred, i) => {
          if (pred === labels[i]) correct += 1;
          if (pred === 1 && labels[i] === 1) tp += 1;
          if (pred === 1 && labels[i] === 0) fp += 1;
          if (pred === 0 && labels[i] === 1) fn += 1;
        });
        const accuracy = correct / p.reviews;
        const positiveShare = mean(labels);

        return {
          metrics: [
            { label: "Lexicon accuracy", value: fmtPct(accuracy), tone: accuracy > 0.8 ? "good" : accuracy < 0.65 ? "bad" : undefined },
            { label: "Precision (positive)", value: fmtPct(tp + fp ? tp / (tp + fp) : 0) },
            { label: "Recall (positive)", value: fmtPct(tp + fn ? tp / (tp + fn) : 0) },
            { label: "Labelled positive", value: fmtPct(positiveShare), hint: "Class balance in the sample", tone: "accent" },
          ],
          charts: [
            {
              title: "Predicted vs actual sentiment",
              svg: groupedBarChart({
                labels: ["Negative", "Positive"],
                series: [
                  { name: "Actual", color: PALETTE[0], values: [labels.filter((l) => l === 0).length, labels.filter((l) => l === 1).length] },
                  { name: "Predicted", color: PALETTE[1], values: [predictions.filter((l) => l === 0).length, predictions.filter((l) => l === 1).length] },
                ],
                yTitle: "Reviews",
                valueFormat: fmtInt,
              }),
              legendItems: [{ label: "Actual label", color: PALETTE[0] }, { label: "Lexicon prediction", color: PALETTE[1] }],
            },
          ],
          insight: `Neutral reviews get a coin-flip label and ${fmtPct(p.labelNoise, 0)} of all labels are then flipped outright. Accuracy lands at ${fmtPct(accuracy)} not because the classifier is weak but because the ceiling itself has moved — the chapter's warning about grading a model against noisy ground truth.`,
          table: {
            columns: ["Review", "Label", "Predicted", ""],
            rows,
          },
        };
      },
    },

    {
      id: "ch8",
      chapter: 8,
      notebook: "Chapter 8.ipynb",
      track: "Optimization",
      company: "OptiLogix",
      title: "Capacity, Pricing, and Planning",
      focus: "Three optimizations, solved exactly",
      methods: ["Linear programming", "Knapsack", "Nonlinear pricing"],
      scenario:
        "OptiLogix must allocate shipments from two warehouses to three markets, pick promotional bundles inside a fixed budget, and set a single price. Every answer here is computed exactly, not sampled.",
      controls: [
        { key: "cap1", label: "Warehouse W1 capacity", min: 300, max: 1500, step: 50, value: 900 },
        { key: "cap2", label: "Warehouse W2 capacity", min: 300, max: 1200, step: 50, value: 700 },
        { key: "budget", label: "Promo budget (k)", min: 25, max: 300, step: 5, value: 150 },
        { key: "elasticity", label: "Price elasticity", min: 0.005, max: 0.04, step: 0.001, value: 0.015, format: (v) => fmtNum(v, 3) },
      ],
      run(p) {
        const marketNames = ["North", "Central", "South"];
        const demands = [600, 500, 400];
        const costW1 = [6.0, 7.0, 9.0];
        const costW2 = [8.0, 6.0, 5.5];
        const totalDemand = sum(demands);

        // Two sources reduce to: choose x_i shipped from W1 to market i.
        // Cost = Σ d_i·c2_i + Σ x_i·(c1_i − c2_i), subject to capacity bounds.
        const feasible = p.cap1 + p.cap2 >= totalDemand;
        const deltas = marketNames.map((_, i) => costW1[i] - costW2[i]);
        const mustShipFromW1 = Math.max(0, totalDemand - p.cap2);
        const order = deltas.map((d, i) => ({ d, i })).sort((a, b) => a.d - b.d);
        const x = new Array(3).fill(0);
        let capacityLeft = Math.min(p.cap1, totalDemand);

        order.forEach(({ d, i }) => {
          if (d < 0 && capacityLeft > 0) {
            const take = Math.min(demands[i], capacityLeft);
            x[i] = take;
            capacityLeft -= take;
          }
        });
        let shipped = sum(x);
        order.forEach(({ i }) => {
          if (shipped >= mustShipFromW1) return;
          const room = Math.min(demands[i] - x[i], p.cap1 - shipped);
          const add = Math.min(room, mustShipFromW1 - shipped);
          if (add > 0) {
            x[i] += add;
            shipped += add;
          }
        });

        const shippingCost = marketNames.reduce(
          (acc, _, i) => acc + x[i] * costW1[i] + (demands[i] - x[i]) * costW2[i],
          0
        );

        // Promo bundles: exact search over all 64 subsets.
        const bundles = [
          { name: "A", cost: 40, uplift: 60 },
          { name: "B", cost: 55, uplift: 85 },
          { name: "C", cost: 35, uplift: 45 },
          { name: "D", cost: 60, uplift: 90 },
          { name: "E", cost: 25, uplift: 35 },
          { name: "F", cost: 45, uplift: 55 },
        ];
        let bestUplift = 0;
        let bestSet = [];
        for (let mask = 0; mask < 1 << bundles.length; mask += 1) {
          let cost = 0;
          let uplift = 0;
          const picked = [];
          for (let b = 0; b < bundles.length; b += 1) {
            if (mask & (1 << b)) {
              cost += bundles[b].cost;
              uplift += bundles[b].uplift;
              picked.push(bundles[b].name);
            }
          }
          if (cost <= p.budget && uplift > bestUplift) {
            bestUplift = uplift;
            bestSet = picked;
          }
        }

        // Nonlinear pricing: demand(p) = 1800·exp(−elasticity·(p − 80)).
        const marginalCost = 35;
        const baseline = 1800;
        const refPrice = 80;
        const demandAt = (price) => baseline * Math.exp(-p.elasticity * (price - refPrice));
        const profitAt = (price) => (price - marginalCost) * demandAt(price);
        const curve = [];
        let bestPrice = 40;
        let bestProfit = -Infinity;
        for (let price = 40; price <= 120.001; price += 0.5) {
          const profit = profitAt(price);
          curve.push({ x: price, y: profit });
          if (profit > bestProfit) {
            bestProfit = profit;
            bestPrice = price;
          }
        }

        return {
          metrics: [
            {
              label: "Min shipping cost",
              value: feasible ? fmtMoney(shippingCost) : "Infeasible",
              hint: feasible ? `${fmtInt(totalDemand)} units delivered` : "Capacity below total demand",
              tone: feasible ? "accent" : "bad",
            },
            { label: "Promo bundles chosen", value: bestSet.length ? bestSet.join(", ") : "none", hint: `Uplift ${fmtInt(bestUplift)}k` },
            { label: "Optimal price", value: fmtMoney(bestPrice, 2), hint: `Unconstrained optimum ${fmtMoney(marginalCost + 1 / p.elasticity, 2)}`, tone: "accent" },
            { label: "Profit at optimum", value: fmtMoney(bestProfit), hint: `${fmtInt(demandAt(bestPrice))} units` },
          ],
          charts: [
            {
              title: "Profit across the price range",
              note: "Exponential demand gives a single interior optimum at marginal cost + 1/elasticity, clipped to the $40–$120 band.",
              svg: lineChart({
                series: [{ name: "Profit", color: PALETTE[0], points: curve, marker: { x: bestPrice, y: bestProfit } }],
                xTitle: "Price ($)",
                yTitle: "Profit ($)",
                tickFormat: (v) => `${fmtNum(v / 1000, 0)}k`,
                xFormat: (v) => fmtNum(v, 0),
              }),
            },
            {
              title: "Shipment allocation by market",
              svg: groupedBarChart({
                labels: marketNames,
                series: [
                  { name: "From W1", color: PALETTE[0], values: x },
                  { name: "From W2", color: PALETTE[2], values: marketNames.map((_, i) => demands[i] - x[i]) },
                ],
                yTitle: "Units",
                valueFormat: fmtInt,
              }),
              legendItems: [{ label: "From W1", color: PALETTE[0] }, { label: "From W2", color: PALETTE[2] }],
            },
          ],
          insight: feasible
            ? `W1 is cheaper into North, W2 into Central and South, so the solver fills those first and uses the leftover only when capacity forces it. Cut W2 below ${fmtInt(totalDemand - p.cap1)} units and W1 must cover the gap at a worse rate — the shadow price of capacity made visible.`
            : `Combined capacity is ${fmtInt(p.cap1 + p.cap2)} against demand of ${fmtInt(totalDemand)}. No allocation satisfies every market; the LP would report infeasible rather than return a partial plan.`,
          table: {
            columns: ["Market", "Demand", "From W1", "From W2", "Cost"],
            rows: marketNames.map((name, i) => [
              name,
              fmtInt(demands[i]),
              fmtInt(x[i]),
              fmtInt(demands[i] - x[i]),
              fmtMoney(x[i] * costW1[i] + (demands[i] - x[i]) * costW2[i], 2),
            ]),
          },
        };
      },
    },

    {
      id: "ch9",
      chapter: 9,
      notebook: "Chapter_9.ipynb",
      track: "Forecasting",
      company: "Time series lab",
      title: "Predictive Analytics and Forecasting",
      focus: "Trend, seasonality, and three baselines",
      methods: ["Moving average", "Exponential smoothing", "Error metrics"],
      scenario:
        "A series with trend, weekly seasonality and noise is simulated so that moving averages, exponential smoothing and ARIMA can be compared on a known ground truth.",
      controls: [
        { key: "periods", label: "Periods", min: 60, max: 400, step: 20, value: 180 },
        { key: "trend", label: "Trend per period", min: 0, max: 1.5, step: 0.05, value: 0.3, format: (v) => fmtNum(v, 2) },
        { key: "season", label: "Seasonal amplitude", min: 0, max: 40, step: 1, value: 12 },
        { key: "noise", label: "Noise (sd)", min: 1, max: 30, step: 1, value: 8 },
        { key: "alpha", label: "Smoothing α", min: 0.05, max: 0.95, step: 0.05, value: 0.3, format: (v) => fmtNum(v, 2) },
      ],
      run(p, rng) {
        const series = [];
        for (let t = 0; t < p.periods; t += 1) {
          series.push(100 + p.trend * t + p.season * Math.sin((2 * Math.PI * t) / 7) + rng.normal(0, p.noise));
        }

        const window = 7;
        const movingAverage = series.map((_, i) =>
          i < window ? series[i] : mean(series.slice(i - window, i))
        );

        const ses = [series[0]];
        for (let t = 1; t < series.length; t += 1) {
          ses.push(p.alpha * series[t - 1] + (1 - p.alpha) * ses[t - 1]);
        }

        const errorsOf = (fitted) =>
          mean(series.slice(window).map((v, i) => Math.abs(v - fitted[i + window])));
        const naive = series.map((_, i) => (i === 0 ? series[0] : series[i - 1]));

        const maeNaive = errorsOf(naive);
        const maeMa = errorsOf(movingAverage);
        const maeSes = errorsOf(ses);
        const best = [
          ["Naive", maeNaive],
          ["Moving average", maeMa],
          ["Exp. smoothing", maeSes],
        ].sort((a, b) => a[1] - b[1])[0];

        return {
          metrics: [
            { label: "Naive MAE", value: fmtNum(maeNaive, 2), hint: "Yesterday as forecast" },
            { label: "Moving average MAE", value: fmtNum(maeMa, 2), hint: `${window}-period window` },
            { label: "Smoothing MAE", value: fmtNum(maeSes, 2), hint: `α = ${fmtNum(p.alpha, 2)}` },
            { label: "Best baseline", value: best[0], hint: `MAE ${fmtNum(best[1], 2)}`, tone: "accent" },
          ],
          charts: [
            {
              title: "Series and fitted baselines",
              note: "A 7-period moving average smooths the weekly cycle away; exponential smoothing tracks it but lags.",
              svg: lineChart({
                series: [
                  { name: "Actual", color: "#98a3ba", points: series.map((y, x) => ({ x, y })), width: 1.4 },
                  { name: "Moving average", color: PALETTE[0], points: movingAverage.map((y, x) => ({ x, y })) },
                  { name: "Exp. smoothing", color: PALETTE[1], points: ses.map((y, x) => ({ x, y })) },
                ],
                xTitle: "Period",
                yTitle: "Value",
              }),
              legendItems: [
                { label: "Actual", color: "#98a3ba" },
                { label: "Moving average", color: PALETTE[0] },
                { label: "Exponential smoothing", color: PALETTE[1] },
              ],
            },
          ],
          insight: `The ranking flips with the shape of the series rather than with model sophistication. Around a flat level, averaging wins — the naive forecast carries a full period of noise into every prediction. Push the trend past about ${fmtNum(p.noise / 4, 1)} per period and the order inverts: a 7-period moving average trails a trend by roughly half its window, while naive is never more than one step behind.`,
          table: {
            columns: ["Period", "Actual", "Moving avg", "Smoothing"],
            rows: [10, 20, 30, 40, 50, 60]
              .filter((i) => i < series.length)
              .map((i) => [String(i), fmtNum(series[i], 2), fmtNum(movingAverage[i], 2), fmtNum(ses[i], 2)]),
          },
        };
      },
    },

    {
      id: "ch10",
      chapter: 10,
      notebook: "Chapter 10.ipynb",
      track: "Anomaly Detection",
      company: "SecureBank",
      title: "Isolation Forest for Transaction Fraud",
      focus: "Rare, unusual transactions",
      methods: ["Isolation Forest", "Contamination tuning", "Anomaly scores"],
      scenario:
        "SecureBank has far more normal transactions than fraudulent ones. Rather than train on labels, the case isolates outliers: unusually large amounts from unusually old, international accounts.",
      controls: [
        { key: "normal", label: "Normal transactions", min: 200, max: 2000, step: 50, value: 950 },
        { key: "fraud", label: "Fraudulent transactions", min: 5, max: 250, step: 5, value: 50 },
        { key: "contamination", label: "Contamination", min: 0.01, max: 0.25, step: 0.01, value: 0.05, format: (v) => fmtPct(v, 0) },
      ],
      run(p, rng) {
        const points = [];
        const labels = [];
        const rows = [];

        for (let i = 0; i < p.normal; i += 1) {
          const amount = rng.gamma(2, 100);
          const time = rng.uniform(0, 24);
          const age = rng.normal(35, 10);
          const international = rng.next() < 0.1 ? 1 : 0;
          points.push([amount, time, age, international]);
          labels.push(0);
        }
        for (let i = 0; i < p.fraud; i += 1) {
          const amount = rng.gamma(9, 300);
          const time = rng.uniform(0, 24);
          const age = rng.normal(50, 15);
          const international = rng.next() < 0.7 ? 1 : 0;
          points.push([amount, time, age, international]);
          labels.push(1);
        }

        const scores = isolationForest(points, rng);
        const cutoff = quantile(scores, 1 - p.contamination);
        const cm = confusion(scores, labels, cutoff);

        points.slice(0, 6).forEach((point, i) => {
          rows.push([
            fmtMoney(point[0], 2),
            `${fmtNum(point[1], 1)}h`,
            fmtNum(point[2], 0),
            point[3] ? "yes" : "no",
            fmtNum(scores[i], 3),
            scores[i] >= cutoff ? "flagged" : "—",
          ]);
        });

        const normalPoints = points.filter((_, i) => labels[i] === 0).map((pt) => ({ x: pt[0], y: pt[2] }));
        const fraudPoints = points.filter((_, i) => labels[i] === 1).map((pt) => ({ x: pt[0], y: pt[2] }));

        return {
          metrics: [
            { label: "Fraud in sample", value: fmtPct(mean(labels)), hint: `${fmtInt(p.fraud)} of ${fmtInt(points.length)}` },
            { label: "Detection rate", value: fmtPct(cm.recall), hint: `${fmtInt(cm.tp)} of ${fmtInt(p.fraud)} caught`, tone: cm.recall > 0.6 ? "good" : "bad" },
            { label: "Precision", value: fmtPct(cm.precision), hint: `${fmtInt(cm.fp)} false alarms`, tone: "accent" },
            { label: "AUC", value: fmtNum(auc(scores, labels), 3), hint: "Threshold-free separation" },
          ],
          charts: [
            {
              title: "Transaction amount vs customer age",
              note: "Fraud is drawn from a heavier gamma and an older age distribution — separable on amount, overlapping on age.",
              svg: scatterChart({
                groups: [
                  { name: "Normal", color: PALETTE[0], points: normalPoints, opacity: 0.35 },
                  { name: "Fraud", color: PALETTE[4], points: fraudPoints, opacity: 0.75, radius: 3.2 },
                ],
                xTitle: "Amount ($)",
                yTitle: "Customer age",
              }),
              legendItems: [{ label: "Normal", color: PALETTE[0] }, { label: "Fraud", color: PALETTE[4] }],
            },
            {
              title: "Anomaly score distribution",
              svg: histogram({ values: scores, bins: 30, color: PALETTE[0], xTitle: "Isolation Forest score" }),
            },
          ],
          insight: `Contamination is an assumption, not an estimate: setting it to ${fmtPct(p.contamination, 0)} flags exactly that share regardless of how much fraud exists. When true fraud (${fmtPct(mean(labels))}) exceeds the assumption, recall is capped before the model does anything.`,
          table: {
            columns: ["Amount", "Time", "Age", "International", "Score", "Decision"],
            rows,
          },
        };
      },
    },

    {
      id: "ch11",
      chapter: 11,
      notebook: "Chapter 11.ipynb",
      track: "Deep Learning",
      company: "StreamNow",
      title: "Neural Network Churn Prediction",
      focus: "Ten behavioural features, imbalanced classes",
      methods: ["Neural network", "Train/test split", "AUC"],
      scenario:
        "StreamNow wants to know which subscribers are about to cancel. Ten behavioural features are simulated with a known class imbalance, then a small network is trained and evaluated.",
      controls: [
        { key: "samples", label: "Customers", min: 300, max: 2500, step: 100, value: 1000 },
        { key: "churnRate", label: "Churn rate", min: 0.1, max: 0.5, step: 0.05, value: 0.3, format: (v) => fmtPct(v, 0) },
        { key: "separation", label: "Signal separation", min: 0.2, max: 2.5, step: 0.1, value: 1.1, format: (v) => fmtNum(v, 1) },
        { key: "hidden", label: "Hidden units", min: 2, max: 20, step: 1, value: 8 },
        { key: "epochs", label: "Training epochs", min: 40, max: 320, step: 20, value: 200 },
      ],
      run(p, rng) {
        const names = [
          "Tenure", "MonthlyCharges", "SupportTickets", "ContractLength", "InternetUsage",
          "NumLogins", "StreamingHours", "PhoneUsage", "DeviceChanges", "RegionScore",
        ];
        const informative = 6;
        const X = [];
        const y = [];
        const rows = [];

        for (let i = 0; i < p.samples; i += 1) {
          const churn = rng.bernoulli(p.churnRate);
          const row = [];
          for (let f = 0; f < 8; f += 1) {
            const shift = f < informative ? (churn ? p.separation : -p.separation) * (f % 2 === 0 ? 1 : -0.7) : 0;
            row.push(rng.normal(shift, 1));
          }
          // Two redundant features: linear combinations of the informative ones.
          row.push(0.7 * row[0] + 0.3 * row[2] + rng.normal(0, 0.1));
          row.push(-0.5 * row[1] + 0.5 * row[4] + rng.normal(0, 0.1));
          X.push(row);
          y.push(churn);
          if (rows.length < 6) {
            rows.push([...row.slice(0, 5).map((v) => fmtNum(v, 2)), churn ? "churn" : "stay"]);
          }
        }

        const model = trainMlp(X, y, p.hidden, p.epochs, rng);
        const gap = model.trainAccuracy - model.testAccuracy;

        return {
          metrics: [
            { label: "Test accuracy", value: fmtPct(model.testAccuracy), hint: `${fmtInt(model.testSize)} held-out rows`, tone: model.testAccuracy > 0.8 ? "good" : undefined },
            { label: "Test AUC", value: fmtNum(model.testAuc, 3), tone: "accent" },
            { label: "Train − test gap", value: fmtPct(gap), hint: gap > 0.08 ? "Overfitting" : "Generalizing", tone: gap > 0.08 ? "bad" : "good" },
            { label: "Majority baseline", value: fmtPct(Math.max(1 - mean(y), mean(y))), hint: "Accuracy from always guessing the common class" },
          ],
          charts: [
            {
              title: "Training loss",
              note: "Binary cross-entropy on each epoch's mini-batch, sampled across training.",
              svg: lineChart({
                series: [{ name: "Loss", color: PALETTE[0], points: model.losses.map((v, i) => ({ x: i, y: v })) }],
                xTitle: "Checkpoint",
                yTitle: "Loss",
                tickFormat: (v) => fmtNum(v, 2),
              }),
            },
            {
              title: "Held-out predictions",
              svg: groupedBarChart({
                labels: ["Predicted stay", "Predicted churn"],
                series: [
                  { name: "Actually stayed", color: PALETTE[0], values: [model.testConfusion.tn, model.testConfusion.fp] },
                  { name: "Actually churned", color: PALETTE[4], values: [model.testConfusion.fn, model.testConfusion.tp] },
                ],
                yTitle: "Customers",
                valueFormat: fmtInt,
              }),
              legendItems: [{ label: "Actually stayed", color: PALETTE[0] }, { label: "Actually churned", color: PALETTE[4] }],
            },
          ],
          insight: `Accuracy alone is misleading here: at a ${fmtPct(p.churnRate, 0)} churn rate, predicting "stay" for everyone already scores ${fmtPct(Math.max(1 - mean(y), mean(y)))}. AUC of ${fmtNum(model.testAuc, 3)} is the number that says whether the network learned anything. Drop signal separation toward 0.2 and watch accuracy hold while AUC collapses.`,
          table: {
            columns: [...names.slice(0, 5), "Churn"],
            rows,
          },
        };
      },
    },

    {
      id: "ch12",
      chapter: 12,
      notebook: "Chapter 12.ipynb",
      track: "Decision Science",
      company: "Subscription app",
      title: "Threshold Selection and ROI",
      focus: "Turning a probability into a business decision",
      methods: ["Logistic regression", "Cost matrix", "Operating point"],
      scenario:
        "A subscription funnel is simulated where an unobserved preference drives both exposure and conversion. A model scores each user; the real question is where to set the offer threshold given what an offer costs and a conversion is worth.",
      controls: [
        { key: "users", label: "Users", min: 1000, max: 12000, step: 500, value: 5000 },
        { key: "exposureEffect", label: "Exposure effect", min: 0, max: 1.2, step: 0.05, value: 0.3, format: (v) => fmtNum(v, 2) },
        { key: "value", label: "Value per conversion ($)", min: 10, max: 250, step: 5, value: 60 },
        { key: "cost", label: "Cost per offer ($)", min: 1, max: 60, step: 1, value: 8 },
        { key: "threshold", label: "Offer threshold", min: 0.02, max: 0.7, step: 0.01, value: 0.15, format: (v) => fmtNum(v, 2) },
      ],
      run(p, rng) {
        const X = [];
        const y = [];
        const rows = [];

        for (let i = 0; i < p.users; i += 1) {
          const pref = rng.normal(0, 1);
          const exposure = rng.bernoulli(sigmoid(0.5 * pref + 0.2 * rng.normal(0, 1)));
          const timeOnApp = Math.max(0, rng.gamma(2, 3) + 1.5 * exposure);
          const priceSensitivity = rng.beta(2, 5);
          const pastPurchases = rng.poisson(Math.exp(0.3 * pref));
          const logit =
            -2 +
            0.8 * pref +
            p.exposureEffect * exposure +
            0.15 * Math.log1p(timeOnApp) -
            1.2 * priceSensitivity +
            0.05 * pastPurchases +
            0.2 * rng.normal(0, 1);
          const convert = rng.bernoulli(sigmoid(logit));
          X.push([pref, exposure, timeOnApp, priceSensitivity, pastPurchases]);
          y.push(convert);
          if (rows.length < 6) {
            rows.push([
              fmtNum(pref, 2),
              exposure ? "yes" : "no",
              fmtNum(timeOnApp, 1),
              fmtNum(priceSensitivity, 2),
              String(pastPurchases),
              convert ? "converted" : "—",
            ]);
          }
        }

        const model = logisticFit(X, y);
        const sweep = [];
        let bestThreshold = 0.02;
        let bestRoi = -Infinity;
        for (let t = 0.02; t <= 0.7; t += 0.02) {
          const cm = confusion(model.scores, y, t);
          const roi = cm.tp * p.value - cm.flagged * p.cost;
          sweep.push({ x: t, y: roi });
          if (roi > bestRoi) {
            bestRoi = roi;
            bestThreshold = t;
          }
        }
        const current = confusion(model.scores, y, p.threshold);
        const currentRoi = current.tp * p.value - current.flagged * p.cost;

        return {
          metrics: [
            { label: "Conversion rate", value: fmtPct(mean(y)), hint: `${fmtInt(sum(y))} conversions` },
            { label: "Model AUC", value: fmtNum(auc(model.scores, y), 3), tone: "accent" },
            { label: "Net value at your threshold", value: fmtMoney(currentRoi), hint: `${fmtInt(current.flagged)} offers sent`, tone: currentRoi > 0 ? "good" : "bad" },
            { label: "Best threshold", value: fmtNum(bestThreshold, 2), hint: `Would return ${fmtMoney(bestRoi)}` },
          ],
          charts: [
            {
              title: "Net value across offer thresholds",
              note: "Every conversion earns the offer value; every offer sent costs, converted or not. The peak is the operating point.",
              svg: lineChart({
                series: [{ name: "Net value", color: PALETTE[0], points: sweep, marker: { x: p.threshold, y: currentRoi } }],
                xTitle: "Offer threshold",
                yTitle: "Net value ($)",
                tickFormat: (v) => (Math.abs(v) >= 1000 ? `${fmtNum(v / 1000, 0)}k` : fmtNum(v, 0)),
                xFormat: (v) => fmtNum(v, 2),
              }),
            },
          ],
          insight: `The optimal threshold moves with the economics, not the model. At ${fmtMoney(p.value)} per conversion and ${fmtMoney(p.cost)} per offer the peak sits near ${fmtNum(bestThreshold, 2)}; make offers cheaper and it slides toward zero — blanket everyone. Accuracy never enters the calculation.`,
          table: {
            columns: ["Preference", "Exposed", "Time on app", "Price sens.", "Past buys", "Outcome"],
            rows,
          },
        };
      },
    },

    {
      id: "ch13",
      chapter: 13,
      notebook: "Chapter 13.ipynb",
      track: "Responsible AI",
      company: "Automated hiring",
      title: "Bias, Fairness, and Accountability",
      focus: "Bias encoded in the label itself",
      methods: ["Demographic parity", "Equal opportunity", "Explainability"],
      scenario:
        "An automated hiring screen is trained on historical decisions that already favoured one group. The label carries the bias, so a model that fits the data perfectly reproduces the discrimination.",
      controls: [
        { key: "applicants", label: "Applicants", min: 300, max: 4000, step: 100, value: 1000 },
        { key: "bias", label: "Explicit bias coefficient", min: 0, max: 2, step: 0.1, value: 0.8, format: (v) => fmtNum(v, 1) },
        { key: "dropGroup", label: "Remove group from model inputs", type: "toggle", value: true, hint: "Fairness through unawareness — does hiding the attribute help?" },
      ],
      run(p, rng) {
        const X = [];
        const y = [];
        const groups = [];
        const rows = [];

        for (let i = 0; i < p.applicants; i += 1) {
          const group = rng.bernoulli(0.5);
          const experience = rng.normal(5 + group, 2);
          const education = rng.bernoulli(0.6);
          const testScore = rng.normal(70 + 5 * group, 10);
          const logit = 0.3 * experience + 0.5 * education + 0.04 * testScore + p.bias * group;
          const hired = rng.bernoulli(sigmoid(logit - 4.5));
          X.push(p.dropGroup ? [experience, education, testScore] : [experience, education, testScore, group]);
          y.push(hired);
          groups.push(group);
          if (rows.length < 6) {
            rows.push([group ? "B" : "A", fmtNum(experience, 1), education ? "degree" : "none", fmtNum(testScore, 0), hired ? "hired" : "—"]);
          }
        }

        const model = logisticFit(X, y);
        const predictions = model.scores.map((s) => (s >= 0.5 ? 1 : 0));

        const rateFor = (g) => {
          const idx = groups.map((v, i) => (v === g ? i : -1)).filter((i) => i >= 0);
          return idx.length ? mean(idx.map((i) => predictions[i])) : 0;
        };
        const tprFor = (g) => {
          const idx = groups.map((v, i) => (v === g && y[i] === 1 ? i : -1)).filter((i) => i >= 0);
          return idx.length ? mean(idx.map((i) => predictions[i])) : 0;
        };

        const rateA = rateFor(0);
        const rateB = rateFor(1);
        const parityGap = rateB - rateA;
        const opportunityGap = tprFor(1) - tprFor(0);

        return {
          metrics: [
            { label: "Selection rate, group A", value: fmtPct(rateA) },
            { label: "Selection rate, group B", value: fmtPct(rateB) },
            {
              label: "Demographic parity gap",
              value: fmtPct(parityGap),
              hint: Math.abs(parityGap) < 0.05 ? "Within 5 points" : "Outside a common 5-point rule",
              tone: Math.abs(parityGap) < 0.05 ? "good" : "bad",
            },
            { label: "Equal opportunity gap", value: fmtPct(opportunityGap), hint: "Difference in true positive rate", tone: "accent" },
          ],
          charts: [
            {
              title: "Selection and true positive rate by group",
              svg: groupedBarChart({
                labels: ["Group A", "Group B"],
                series: [
                  { name: "Selection rate", color: PALETTE[0], values: [rateA, rateB] },
                  { name: "True positive rate", color: PALETTE[3], values: [tprFor(0), tprFor(1)] },
                ],
                yTitle: "Rate",
                tickFormat: (v) => fmtPct(v, 0),
                valueFormat: (v) => fmtPct(v, 0),
              }),
              legendItems: [{ label: "Selection rate", color: PALETTE[0] }, { label: "True positive rate", color: PALETTE[3] }],
            },
          ],
          insight: p.dropGroup
            ? `Group membership is hidden from the model, yet the parity gap is still ${fmtPct(parityGap)}. Experience is drawn as <code>normal(5 + group, 2)</code> and test score as <code>normal(70 + 5·group, 10)</code>, so both act as proxies. Zeroing the explicit bias coefficient does not close the gap either — that removes only the direct term and leaves the shifted features untouched. Bias lives in the data here, not just in a coefficient.`
            : `Group is supplied directly and the model uses it, giving a parity gap of ${fmtPct(parityGap)}. Toggle it off and the gap barely moves: experience and test score are drawn with their own group shifts, so they carry the same information. Removing the protected attribute is not a fairness fix.`,
          table: {
            columns: ["Group", "Experience", "Education", "Test score", "Outcome"],
            rows,
          },
        };
      },
    },

    {
      id: "ch14",
      chapter: 14,
      notebook: "Chapter 14.ipynb",
      track: "AI Analytics",
      company: "Retention team",
      title: "From Reporting to AI-Driven Decisions",
      focus: "What prediction adds over a dashboard",
      methods: ["Logistic regression", "Targeting", "ROI framework"],
      scenario:
        "A company already reports its churn rate every month. The case asks what changes when a model predicts churn per customer, and whether acting on those predictions pays for itself.",
      controls: [
        { key: "customers", label: "Customers", min: 300, max: 5000, step: 100, value: 1000 },
        { key: "interventionCost", label: "Cost per intervention ($)", min: 1, max: 60, step: 1, value: 12 },
        { key: "saveRate", label: "Intervention success rate", min: 0.05, max: 0.7, step: 0.05, value: 0.25, format: (v) => fmtPct(v, 0) },
        { key: "margin", label: "Margin per retained ($)", min: 20, max: 400, step: 10, value: 120 },
        { key: "threshold", label: "Targeting threshold", min: 0.05, max: 0.85, step: 0.01, value: 0.35, format: (v) => fmtNum(v, 2) },
      ],
      run(p, rng) {
        const X = [];
        const y = [];
        const rows = [];

        for (let i = 0; i < p.customers; i += 1) {
          const tenure = rng.exponential(24);
          const monthlySpend = rng.normal(70, 20);
          const supportCalls = rng.poisson(2);
          const logit = -0.04 * tenure + 0.03 * supportCalls - 0.01 * monthlySpend + 1.2;
          const churn = rng.bernoulli(sigmoid(logit));
          X.push([tenure, monthlySpend, supportCalls]);
          y.push(churn);
          if (rows.length < 6) {
            rows.push([fmtNum(tenure, 1), fmtMoney(monthlySpend, 2), String(supportCalls), churn ? "churned" : "stayed"]);
          }
        }

        const model = logisticFit(X, y);
        const cm = confusion(model.scores, y, p.threshold);
        const saves = cm.tp * p.saveRate;
        const roi = saves * p.margin - cm.flagged * p.interventionCost;

        const blanketRoi = sum(y) * p.saveRate * p.margin - p.customers * p.interventionCost;

        const sweep = [];
        for (let t = 0.05; t <= 0.85; t += 0.025) {
          const c = confusion(model.scores, y, t);
          sweep.push({ x: t, y: c.tp * p.saveRate * p.margin - c.flagged * p.interventionCost });
        }

        return {
          metrics: [
            { label: "Churn rate", value: fmtPct(mean(y)), hint: "What the dashboard reports" },
            { label: "Customers targeted", value: fmtInt(cm.flagged), hint: `${fmtPct(cm.flagged / p.customers)} of the base`, tone: "accent" },
            { label: "Expected saves", value: fmtNum(saves, 1), hint: `${fmtPct(cm.precision)} of those targeted would churn` },
            { label: "Net return", value: fmtMoney(roi), hint: `Blanket campaign: ${fmtMoney(blanketRoi)}`, tone: roi > 0 ? "good" : "bad" },
          ],
          charts: [
            {
              title: "Net return across targeting thresholds",
              svg: lineChart({
                series: [{ name: "Net return", color: PALETTE[0], points: sweep, marker: { x: p.threshold, y: roi } }],
                xTitle: "Targeting threshold",
                yTitle: "Net return ($)",
                tickFormat: (v) => (Math.abs(v) >= 1000 ? `${fmtNum(v / 1000, 0)}k` : fmtNum(v, 0)),
                xFormat: (v) => fmtNum(v, 2),
              }),
            },
          ],
          insight: `The dashboard tells you ${fmtPct(mean(y))} of customers churn. The model tells you <em>which</em>, which is only worth something when interventions are scarce: at ${fmtMoney(p.interventionCost)} each, targeting returns ${fmtMoney(roi)} against ${fmtMoney(blanketRoi)} for contacting everyone. Cut the cost far enough and the model stops earning its keep.`,
          table: {
            columns: ["Tenure (months)", "Monthly spend", "Support calls", "Outcome"],
            rows,
          },
        };
      },
    },

    {
      id: "ch15",
      chapter: 15,
      notebook: "Chapter 15.ipynb",
      track: "AI Strategy",
      company: "Transforming enterprise",
      title: "AI Strategy and Organizational Change",
      focus: "Data literacy, automation, and performance",
      methods: ["Linear regression", "Coefficient recovery", "Effect sizes"],
      scenario:
        "An organization invests in data literacy, process automation and AI tooling at once. The case separates how much each lever actually contributes to measured performance.",
      controls: [
        { key: "employees", label: "Employees", min: 100, max: 2000, step: 50, value: 500 },
        { key: "literacyEffect", label: "Literacy effect", min: 0, max: 1.2, step: 0.05, value: 0.4, format: (v) => fmtNum(v, 2) },
        { key: "automationEffect", label: "Automation effect", min: 0, max: 60, step: 2, value: 30 },
        { key: "aiEffect", label: "AI usage effect", min: 0, max: 40, step: 2, value: 10 },
        { key: "noise", label: "Unexplained noise (sd)", min: 2, max: 40, step: 1, value: 10 },
      ],
      run(p, rng) {
        const X = [];
        const performance = [];
        const rows = [];

        for (let i = 0; i < p.employees; i += 1) {
          const literacy = rng.normal(50, 10);
          const automation = rng.uniform(0, 1);
          const aiUsage = rng.bernoulli(0.5);
          const value =
            p.literacyEffect * literacy + p.automationEffect * automation + p.aiEffect * aiUsage + rng.normal(0, p.noise);
          X.push([literacy, automation, aiUsage]);
          performance.push(value);
          if (rows.length < 6) {
            rows.push([fmtNum(literacy, 1), fmtNum(automation, 2), aiUsage ? "yes" : "no", fmtNum(value, 1)]);
          }
        }

        const fit = ols(X, performance);
        const literacyPoints = X.map((row, i) => ({ x: row[0], y: performance[i] }));
        const xs = X.map((row) => row[0]);
        const lo = Math.min(...xs);
        const hi = Math.max(...xs);
        const meanAutomation = mean(X.map((r) => r[1]));
        const meanAi = mean(X.map((r) => r[2]));
        const lineAt = (x) => fit.coef[0] + fit.coef[1] * x + fit.coef[2] * meanAutomation + fit.coef[3] * meanAi;

        return {
          metrics: [
            { label: "R²", value: fit.singular ? "—" : fmtNum(fit.r2, 3), hint: "Variance explained", tone: fit.r2 > 0.5 && !fit.singular ? "good" : undefined },
            { label: "Literacy coefficient", value: fit.singular ? "—" : fmtNum(fit.coef[1], 3), hint: `True value ${fmtNum(p.literacyEffect, 2)}`, tone: "accent" },
            { label: "Automation coefficient", value: fit.singular ? "—" : fmtNum(fit.coef[2], 1), hint: `True value ${fmtInt(p.automationEffect)}` },
            { label: "AI usage coefficient", value: fit.singular ? "—" : fmtNum(fit.coef[3], 1), hint: `True value ${fmtInt(p.aiEffect)}` },
          ],
          charts: [
            {
              title: "Data literacy vs performance",
              note: "The fitted line holds automation and AI usage at their sample means.",
              svg: scatterChart({
                groups: [{ name: "Employees", color: PALETTE[0], points: literacyPoints, opacity: 0.4 }],
                line: { points: [{ x: lo, y: lineAt(lo) }, { x: hi, y: lineAt(hi) }], color: "#131a2b" },
                xTitle: "Data literacy score",
                yTitle: "Performance",
              }),
            },
            {
              title: "Estimated vs true effects",
              svg: groupedBarChart({
                labels: ["Literacy ×10", "Automation", "AI usage"],
                series: [
                  { name: "True", color: PALETTE[0], values: [p.literacyEffect * 10, p.automationEffect, p.aiEffect] },
                  { name: "Estimated", color: PALETTE[1], values: [fit.coef[1] * 10, fit.coef[2], fit.coef[3]] },
                ],
                yTitle: "Effect size",
                valueFormat: (v) => fmtNum(v, 1),
              }),
              legendItems: [{ label: "True effect", color: PALETTE[0] }, { label: "Estimated", color: PALETTE[1] }],
            },
          ],
          insight: `Because the three levers are drawn independently, regression recovers each one cleanly — R² of ${fmtNum(fit.r2, 3)} with ${fmtInt(p.employees)} employees. Real transformation programmes roll them out together, and correlated levers make these coefficients impossible to separate. The clean answer here is a property of the simulation.`,
          table: {
            columns: ["Literacy", "Automation", "AI usage", "Performance"],
            rows,
          },
        };
      },
    },

    {
      id: "ch16",
      chapter: 16,
      notebook: "Chapter 16.ipynb",
      track: "Sustainability",
      company: "Operations and ESG",
      title: "Energy, Emissions, and ESG Performance",
      focus: "What actually moves emissions",
      methods: ["Linear regression", "Scenario comparison", "ESG metrics"],
      scenario:
        "A company tracks energy use, renewable share and whether AI optimization is enabled at each site, then reports emissions. The case separates the contribution of each lever.",
      controls: [
        { key: "sites", label: "Sites", min: 100, max: 1500, step: 50, value: 400 },
        { key: "renewableFloor", label: "Minimum renewable share", min: 0, max: 0.8, step: 0.05, value: 0, format: (v) => fmtPct(v, 0) },
        { key: "aiRollout", label: "AI optimization rollout", min: 0, max: 1, step: 0.05, value: 0.5, format: (v) => fmtPct(v, 0) },
        { key: "renewableEffect", label: "Renewable effect", min: 0, max: 60, step: 2, value: 30 },
      ],
      run(p, rng) {
        const X = [];
        const emissions = [];
        const rows = [];

        for (let i = 0; i < p.sites; i += 1) {
          const energy = rng.normal(100, 15);
          const renewable = rng.uniform(p.renewableFloor, 1);
          const aiOptimization = rng.bernoulli(p.aiRollout);
          const value = 0.8 * energy - p.renewableEffect * renewable - 10 * aiOptimization + rng.normal(0, 10);
          X.push([energy, renewable, aiOptimization]);
          emissions.push(value);
          if (rows.length < 6) {
            rows.push([fmtNum(energy, 1), fmtPct(renewable, 0), aiOptimization ? "on" : "off", fmtNum(value, 1)]);
          }
        }

        const fit = ols(X, emissions);
        const withAi = emissions.filter((_, i) => X[i][2] === 1);
        const withoutAi = emissions.filter((_, i) => X[i][2] === 0);
        const renewablePoints = X.map((row, i) => ({ x: row[1], y: emissions[i] }));
        const meanEnergy = mean(X.map((r) => r[0]));
        const meanAi = mean(X.map((r) => r[2]));
        const lineAt = (x) => fit.coef[0] + fit.coef[1] * meanEnergy + fit.coef[2] * x + fit.coef[3] * meanAi;

        return {
          metrics: [
            { label: "Mean emissions", value: fmtNum(mean(emissions), 1), hint: `${fmtInt(p.sites)} sites` },
            {
              label: "Renewable coefficient",
              value: fit.singular ? "—" : fmtNum(fit.coef[2], 1),
              hint: fit.singular ? "Not identifiable — a lever has no variation" : `True value −${fmtInt(p.renewableEffect)}`,
              tone: fit.singular ? "bad" : "accent",
            },
            {
              label: "AI optimization saving",
              value: withAi.length && withoutAi.length ? fmtNum(mean(withoutAi) - mean(withAi), 1) : "—",
              hint: withAi.length && withoutAi.length ? "Mean difference, AI on vs off" : "Every site is on the same setting",
              tone: withAi.length && withoutAi.length ? "good" : "bad",
            },
            { label: "R²", value: fit.singular ? "—" : fmtNum(fit.r2, 3), hint: "Variance explained" },
          ],
          charts: [
            {
              title: "Renewable share vs emissions",
              note: "The fitted line holds energy use and AI rollout at their sample means.",
              svg: scatterChart({
                groups: [{ name: "Sites", color: PALETTE[0], points: renewablePoints, opacity: 0.42 }],
                line: { points: [{ x: p.renewableFloor, y: lineAt(p.renewableFloor) }, { x: 1, y: lineAt(1) }], color: "#131a2b" },
                xTitle: "Renewable share",
                yTitle: "Emissions",
              }),
            },
            {
              title: "Mean emissions by AI optimization",
              svg: barChart({
                labels: ["AI off", "AI on"],
                values: [withoutAi.length ? mean(withoutAi) : 0, withAi.length ? mean(withAi) : 0],
                color: PALETTE[0],
                yTitle: "Mean emissions",
                valueFormat: (v) => fmtNum(v, 1),
              }),
            },
          ],
          insight: fit.singular
            ? `AI optimization is now at ${fmtPct(p.aiRollout, 0)} across every site, so there is no comparison group and none of the coefficients can be estimated. Reporting "—" is the honest answer here; a regression that still printed numbers would be printing artefacts.`
            : `Raising the renewable floor to ${fmtPct(p.renewableFloor, 0)} cuts mean emissions directly, but it also compresses the range of renewable share — and with less variation left, the coefficient gets harder to estimate. Policy that removes variation removes your ability to measure its own effect.`,
          table: {
            columns: ["Energy use", "Renewable", "AI optimization", "Emissions"],
            rows,
          },
        };
      },
    },

    {
      id: "ch17",
      chapter: 17,
      notebook: "Chapter 17.ipynb",
      track: "AI Strategy",
      company: "Decision intelligence",
      title: "The Future of AI in Business",
      focus: "Real-time access and explainability as levers",
      methods: ["Linear regression", "Scenario comparison", "Readiness"],
      scenario:
        "A global company routes decisions through an AI layer. Data quality, real-time access and explainability are each varied to see which one moves decision quality most.",
      controls: [
        { key: "decisions", label: "Decisions", min: 100, max: 2000, step: 50, value: 600 },
        { key: "dataQuality", label: "Mean data quality", min: 40, max: 95, step: 1, value: 70 },
        { key: "realTime", label: "Real-time adoption", min: 0, max: 1, step: 0.05, value: 0.5, format: (v) => fmtPct(v, 0) },
        { key: "xai", label: "Explainability adoption", min: 0, max: 1, step: 0.05, value: 0.4, format: (v) => fmtPct(v, 0) },
      ],
      run(p, rng) {
        const X = [];
        const quality = [];
        const rows = [];

        for (let i = 0; i < p.decisions; i += 1) {
          const dataQuality = rng.normal(p.dataQuality, 10);
          const realTime = rng.bernoulli(p.realTime);
          const xai = rng.bernoulli(p.xai);
          const value = 0.5 * dataQuality + 15 * realTime + 10 * xai + rng.normal(0, 10);
          X.push([dataQuality, realTime, xai]);
          quality.push(value);
          if (rows.length < 6) {
            rows.push([fmtNum(dataQuality, 1), realTime ? "yes" : "no", xai ? "yes" : "no", fmtNum(value, 1)]);
          }
        }

        const fit = ols(X, quality);
        const combos = [
          { label: "Neither", rt: 0, x: 0 },
          { label: "Real-time", rt: 1, x: 0 },
          { label: "XAI", rt: 0, x: 1 },
          { label: "Both", rt: 1, x: 1 },
        ];
        const comboMeans = combos.map((combo) => {
          const values = quality.filter((_, i) => X[i][1] === combo.rt && X[i][2] === combo.x);
          return values.length ? mean(values) : 0;
        });

        return {
          metrics: [
            { label: "Mean decision quality", value: fmtNum(mean(quality), 1), hint: `${fmtInt(p.decisions)} decisions` },
            {
              label: "Real-time uplift",
              value: fit.singular ? "—" : `+${fmtNum(fit.coef[2], 1)}`,
              hint: fit.singular ? "Not identifiable at this adoption rate" : "True value +15",
              tone: fit.singular ? "bad" : "accent",
            },
            {
              label: "Explainability uplift",
              value: fit.singular ? "—" : `+${fmtNum(fit.coef[3], 1)}`,
              hint: fit.singular ? "Not identifiable at this adoption rate" : "True value +10",
              tone: fit.singular ? "bad" : undefined,
            },
            { label: "R²", value: fit.singular ? "—" : fmtNum(fit.r2, 3), hint: "Variance explained" },
          ],
          charts: [
            {
              title: "Decision quality by capability mix",
              note: "Empty bars mean that combination never occurred at the current adoption rates.",
              svg: barChart({
                labels: combos.map((c) => c.label),
                values: comboMeans,
                color: PALETTE[0],
                yTitle: "Mean decision quality",
                valueFormat: (v) => fmtNum(v, 1),
              }),
            },
          ],
          insight: fit.singular
            ? `At least one capability is now at 0% or 100%, so every decision shares the same value and its effect cannot be estimated at all — there is no comparison group left. The regression does not return a weak estimate here; it returns none.`
            : `Full rollout is good operations and bad measurement. Push either adoption slider to 0% or 100% and that lever's effect stops being identifiable, which is the argument for staging a rollout rather than flipping it on everywhere at once.`,
          table: {
            columns: ["Data quality", "Real-time", "XAI", "Decision quality"],
            rows,
          },
        };
      },
    },

    {
      id: "ch18",
      chapter: 18,
      notebook: "Chapter_18.ipynb",
      track: "AI Strategy",
      company: "FutureProof Inc.",
      title: "Integrated AI Strategy and Sustainability",
      focus: "Readiness and sustainability side by side",
      methods: ["Composite scoring", "Unit comparison", "Governance risk"],
      scenario:
        "FutureProof Inc. is deploying AI across five business units. Each is scored for AI readiness and for sustainability, so that governance attention can go where both signals are weak.",
      controls: [
        { key: "qualityWeight", label: "Data quality weight", min: 0, max: 1, step: 0.05, value: 0.5, format: (v) => fmtNum(v, 2) },
        { key: "xaiWeight", label: "Explainability weight", min: 0, max: 40, step: 1, value: 20 },
        { key: "genaiWeight", label: "GenAI adoption weight", min: 0, max: 20, step: 1, value: 10 },
      ],
      run(p, rng) {
        const units = ["Finance", "Retail Ops", "Customer Support", "Supply Chain", "HR"];
        const readiness = [];
        const sustainability = [];
        const rows = [];

        units.forEach((unit) => {
          const dataQuality = rng.normal(75, 8);
          const xaiEnabled = rng.bernoulli(0.6);
          const genaiAdoption = rng.randint(0, 4);
          const energy = rng.normal(600, 200);
          const renewableShare = rng.uniform(0.1, 0.8);
          const cyberIncidents = rng.poisson(1.2);

          const readinessScore = p.qualityWeight * dataQuality + p.xaiWeight * xaiEnabled + p.genaiWeight * genaiAdoption;
          const sustainabilityScore = renewableShare * 100 - energy / 20;

          readiness.push(readinessScore);
          sustainability.push(sustainabilityScore);
          rows.push([
            unit,
            fmtNum(dataQuality, 1),
            xaiEnabled ? "yes" : "no",
            String(genaiAdoption),
            fmtNum(energy, 0),
            fmtPct(renewableShare, 0),
            String(cyberIncidents),
          ]);
        });

        const topIdx = readiness.indexOf(Math.max(...readiness));
        const weakIdx = sustainability.indexOf(Math.min(...sustainability));

        return {
          metrics: [
            { label: "Most AI-ready unit", value: units[topIdx], hint: `Score ${fmtNum(readiness[topIdx], 1)}`, tone: "accent" },
            { label: "Mean readiness", value: fmtNum(mean(readiness), 1), hint: "Across five units" },
            { label: "Weakest on sustainability", value: units[weakIdx], hint: `Score ${fmtNum(sustainability[weakIdx], 1)}`, tone: "bad" },
            { label: "Readiness spread", value: fmtNum(Math.max(...readiness) - Math.min(...readiness), 1), hint: "Max minus min" },
          ],
          charts: [
            {
              title: "AI readiness by business unit",
              svg: barChart({ labels: units, values: readiness, color: PALETTE[0], yTitle: "Readiness score", valueFormat: (v) => fmtNum(v, 0) }),
            },
            {
              title: "Sustainability score by business unit",
              note: "Renewable share minus an energy penalty; negative scores mean energy use dominates.",
              svg: barChart({ labels: units, values: sustainability, color: PALETTE[2], yTitle: "Sustainability score", valueFormat: (v) => fmtNum(v, 0) }),
            },
          ],
          insight: `Five units is a very small sample: press "New random sample" and the ranking reorders almost every time. That instability is the real governance finding — a league table built on five noisy observations should not drive investment decisions on its own.`,
          table: {
            columns: ["Unit", "Data quality", "XAI", "GenAI", "Energy (MWh)", "Renewable", "Incidents"],
            rows,
          },
        };
      },
    },
  ];

  /* ------------------------------------------------------------------ *
   * UI
   * ------------------------------------------------------------------ */

  const listNode = root.querySelector("[data-sim-list]");
  const detailNode = root.querySelector("[data-sim-detail]");
  const searchNode = root.querySelector("[data-sim-search]");
  const statusNode = root.querySelector("[data-sim-status]");

  let activeId = CASES[0].id;
  let seed = 42;
  let params = {};
  let frameRequest = 0;

  const activeCase = () => CASES.find((item) => item.id === activeId) || CASES[0];

  function defaultParams(caseDef) {
    const next = {};
    caseDef.controls.forEach((control) => {
      next[control.key] = control.value;
    });
    return next;
  }

  function renderRail(filter = "") {
    const needle = filter.trim().toLowerCase();
    const matches = CASES.filter((item) => {
      if (!needle) return true;
      return [item.company, item.title, item.track, item.focus, `chapter ${item.chapter}`, ...item.methods]
        .join(" ")
        .toLowerCase()
        .includes(needle);
    });

    if (!matches.length) {
      listNode.innerHTML = `<p class="sim-rail-empty">No case matches “${escapeHtml(filter)}”.</p>`;
      return;
    }

    let html = "";
    let lastTrack = null;
    matches.forEach((item) => {
      if (item.track !== lastTrack) {
        html += `<p class="sim-track-label">${escapeHtml(item.track)}</p>`;
        lastTrack = item.track;
      }
      html += `
        <button class="sim-case-button" type="button" role="tab" data-case="${item.id}" aria-selected="${item.id === activeId}">
          <span class="sim-case-chapter" aria-hidden="true">${item.chapter}</span>
          <span>
            <span class="sim-case-name">${escapeHtml(item.company)}</span>
            <span class="sim-case-sub">${escapeHtml(item.focus)}</span>
          </span>
        </button>`;
    });
    listNode.innerHTML = html;
  }

  function controlMarkup(control, value) {
    if (control.type === "toggle") {
      return `
        <div class="sim-control">
          <label class="sim-toggle">
            <input type="checkbox" data-control="${control.key}" ${value ? "checked" : ""}>
            <span>${escapeHtml(control.label)}</span>
          </label>
          ${control.hint ? `<p class="sim-control-hint">${escapeHtml(control.hint)}</p>` : ""}
        </div>`;
    }

    const display = control.format ? control.format(value) : fmtInt(value);
    const fill = fillPercent(control, value);
    return `
      <div class="sim-control">
        <div class="sim-control-top">
          <span>${escapeHtml(control.label)}</span>
          <output data-output="${control.key}">${escapeHtml(display)}</output>
        </div>
        <input type="range" data-control="${control.key}" min="${control.min}" max="${control.max}" step="${control.step}" value="${value}"
          style="--fill:${fill}" aria-label="${escapeHtml(control.label)}">
        ${control.hint ? `<p class="sim-control-hint">${escapeHtml(control.hint)}</p>` : ""}
      </div>`;
  }

  /** Position of the thumb as a percentage, for the filled part of the track. */
  function fillPercent(control, value) {
    const span = control.max - control.min;
    if (!span) return "0%";
    return `${(((value - control.min) / span) * 100).toFixed(1)}%`;
  }

  function renderCase() {
    const caseDef = activeCase();

    detailNode.innerHTML = `
      <header class="sim-case-head">
        <p class="project-label">Chapter ${caseDef.chapter} · ${escapeHtml(caseDef.track)}</p>
        <h2>${escapeHtml(caseDef.company)} — ${escapeHtml(caseDef.title)}</h2>
        <p class="sim-scenario">${escapeHtml(caseDef.scenario)}</p>
        <ul class="sim-method-tags">${caseDef.methods.map((m) => `<li>${escapeHtml(m)}</li>`).join("")}</ul>
      </header>
      <div class="sim-workbench">
        <form class="sim-controls" data-sim-controls novalidate>
          <h3>Simulation parameters</h3>
          ${caseDef.controls.map((control) => controlMarkup(control, params[control.key])).join("")}
          <div class="sim-control-actions">
            <button class="sim-button is-primary" type="button" data-sim-reseed>New random sample</button>
            <button class="sim-button" type="button" data-sim-reset>Reset parameters</button>
          </div>
          <p class="sim-seed">Seed <strong data-sim-seed>${seed}</strong></p>
        </form>
        <div class="sim-metrics" data-sim-metrics></div>
      </div>
      <div class="sim-wide">
        <div class="sim-charts" data-sim-charts></div>
        <p class="sim-insight" data-sim-insight></p>
        <details class="sim-preview">
          <summary><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 7 7-7 7" /></svg>Simulated sample rows</summary>
          <div class="sim-table-wrap" data-sim-table></div>
        </details>
        <p class="sim-source">Generated in your browser from the process in
          <a href="${REPO}${encodeURIComponent(caseDef.notebook)}" target="_blank" rel="noreferrer">${escapeHtml(caseDef.notebook)}</a>.
        </p>
      </div>`;

    runCase();
  }

  function runCase() {
    const caseDef = activeCase();
    const started = performance.now();
    let result;

    try {
      result = caseDef.run(params, makeRng(seed));
    } catch (error) {
      detailNode.querySelector("[data-sim-insight]").textContent =
        "This configuration could not be simulated. Reset the parameters to continue.";
      if (statusNode) statusNode.textContent = "Simulation error";
      return;
    }

    const elapsed = Math.max(1, Math.round(performance.now() - started));

    detailNode.querySelector("[data-sim-metrics]").innerHTML = result.metrics
      .map(
        (metric) => `
        <div class="sim-metric${metric.tone ? ` is-${metric.tone}` : ""}">
          <span class="sim-metric-label">${escapeHtml(metric.label)}</span>
          <span class="sim-metric-value">${escapeHtml(metric.value)}</span>
          <span class="sim-metric-hint">${metric.hint ? escapeHtml(metric.hint) : ""}</span>
        </div>`
      )
      .join("");

    detailNode.querySelector("[data-sim-charts]").innerHTML = result.charts
      .map(
        (chart) => `
        <figure class="sim-chart">
          <h4>${escapeHtml(chart.title)}</h4>
          ${chart.note ? `<p class="sim-chart-note">${escapeHtml(chart.note)}</p>` : ""}
          <div class="sim-chart-canvas">${chart.svg}</div>
          ${chart.legendItems ? legend(chart.legendItems) : ""}
        </figure>`
      )
      .join("");

    // Insight strings intentionally carry light inline markup (<code>, <em>).
    detailNode.querySelector("[data-sim-insight]").innerHTML = result.insight;

    const table = result.table;
    detailNode.querySelector("[data-sim-table]").innerHTML = table
      ? `<table class="sim-table">
          <thead><tr>${table.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>
          <tbody>${table.rows
            .map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`)
            .join("")}</tbody>
        </table>`
      : "";

    if (statusNode) {
      statusNode.textContent = `Case ${activeCase().chapter} simulated in ${elapsed} ms`;
    }
  }

  // requestAnimationFrame stops firing while the tab is hidden, so fall back to
  // a timer and let whichever lands first do the work.
  function scheduleRun() {
    if (frameRequest) return;
    let done = false;
    const fire = () => {
      if (done) return;
      done = true;
      frameRequest = 0;
      runCase();
    };
    frameRequest = window.requestAnimationFrame(fire);
    window.setTimeout(fire, 60);
  }

  function selectCase(id) {
    if (id === activeId) return;
    activeId = id;
    params = defaultParams(activeCase());
    renderRail(searchNode ? searchNode.value : "");
    renderCase();
  }

  listNode.addEventListener("click", (event) => {
    const button = event.target.closest("[data-case]");
    if (!button) return;
    selectCase(button.dataset.case);
  });

  detailNode.addEventListener("input", (event) => {
    const input = event.target.closest("[data-control]");
    if (!input) return;
    const caseDef = activeCase();
    const control = caseDef.controls.find((c) => c.key === input.dataset.control);
    if (!control) return;

    if (control.type === "toggle") {
      params[control.key] = input.checked;
    } else {
      const value = Number(input.value);
      params[control.key] = value;
      input.style.setProperty("--fill", fillPercent(control, value));
      const output = detailNode.querySelector(`[data-output="${control.key}"]`);
      if (output) output.textContent = control.format ? control.format(value) : fmtInt(value);
    }
    scheduleRun();
  });

  detailNode.addEventListener("click", (event) => {
    if (event.target.closest("[data-sim-reseed]")) {
      seed = Math.floor(Math.random() * 100000);
      const seedNode = detailNode.querySelector("[data-sim-seed]");
      if (seedNode) seedNode.textContent = String(seed);
      scheduleRun();
      return;
    }
    if (event.target.closest("[data-sim-reset]")) {
      params = defaultParams(activeCase());
      renderCase();
    }
  });

  if (searchNode) {
    searchNode.addEventListener("input", () => renderRail(searchNode.value));
  }

  params = defaultParams(activeCase());
  renderRail();
  renderCase();

  // Exposed so the browser tests can drive the explorer without clicking.
  window.simulationCases = {
    select: selectCase,
    count: CASES.length,
    ids: CASES.map((c) => c.id),
  };
})();

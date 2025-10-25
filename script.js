// Convolutional Layer Visualizer (no frameworks)
// - Splits input image into RGB channels
// - Applies K 3x3x3 kernels (depth-wise + sum) with stride=1 and optional same padding
// - Shows kernels and resulting K feature maps

(function () {
  const imageCanvas = document.getElementById('imageCanvas');
  const rCanvas = document.getElementById('rCanvas');
  const gCanvas = document.getElementById('gCanvas');
  const bCanvas = document.getElementById('bCanvas');
  const kernelsContainer = document.getElementById('kernelsContainer');
  const outputsContainer = document.getElementById('outputsContainer');
  // Practice section elements
  const pRCanvas = document.getElementById('pR');
  const pGCanvas = document.getElementById('pG');
  const pBCanvas = document.getElementById('pB');
  const practiceRegenerate = document.getElementById('practiceRegenerate');
  const practiceReveal = document.getElementById('practiceReveal');
  const practiceOutputs = document.getElementById('practiceOutputs');
  const practiceImageMatrices = document.getElementById('practiceImageMatrices');
  const practiceKernelMatrices = document.getElementById('practiceKernelMatrices');
  const copyCodeBtn = document.getElementById('copyCode');

  const imageFile = document.getElementById('imageFile');
  const useSampleBtn = document.getElementById('useSample');
  const kernelsRange = document.getElementById('kernelsRange');
  const kernelsCount = document.getElementById('kernelsCount');
  const kernelSet = document.getElementById('kernelSet');
  const randomizeBtn = document.getElementById('randomizeKernels');
  // Removed ReLU and padding toggles; always use no ReLU and no padding

  // State
  let W = 0, H = 0;
  let R = null, G = null, B = null; // Float32Array length W*H (0..255)
  let kernels = []; // Array of { r: number[9], g: number[9], b: number[9] }
  // Practice state
  let pW = 0, pH = 0;
  let pR = null, pG = null, pB = null; // Float32Array (0..255), size pW*pH
  let pKernels = []; // same shape as kernels
  let practiceRevealed = false;

  // Utils
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function drawSampleToCanvas(canvas, size = 256) {
    canvas.width = size; canvas.height = size;
    const ctx = canvas.getContext('2d');
    // Gradient background
    const grad = ctx.createLinearGradient(0, 0, size, size);
    grad.addColorStop(0, '#1e3a8a');
    grad.addColorStop(0.5, '#10b981');
    grad.addColorStop(1, '#ef4444');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, size);
    // Grid of colored squares
    const colors = ['#f59e0b', '#3b82f6', '#8b5cf6', '#22d3ee', '#f472b6'];
    const n = 5; const pad = 8; const cell = (size - pad * (n + 1)) / n;
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        ctx.fillStyle = colors[(x + y) % colors.length];
        ctx.fillRect(pad + x * (cell + pad), pad + y * (cell + pad), cell, cell);
      }
    }
    // Lines
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    for (let i = 0; i < 6; i++) {
      ctx.beginPath();
      ctx.moveTo(0, (i + 1) * size / 7);
      ctx.lineTo(size, i * size / 7);
      ctx.stroke();
    }
  }

  async function drawImageFileToCanvas(file, canvas, maxSize = 256) {
    const img = new Image();
    const objectUrl = URL.createObjectURL(file);
    try {
      await new Promise((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = reject;
        img.src = objectUrl;
      });
      const { width: w, height: h } = img;
      const scale = Math.min(1, maxSize / Math.max(w, h));
      const outW = Math.max(1, Math.round(w * scale));
      const outH = Math.max(1, Math.round(h * scale));
      canvas.width = outW; canvas.height = outH;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, outW, outH);
    } finally {
      URL.revokeObjectURL(objectUrl);
    }
  }

  function extractRGB(canvas) {
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    const img = ctx.getImageData(0, 0, width, height);
    const data = img.data;
    const size = width * height;
    const r = new Float32Array(size);
    const g = new Float32Array(size);
    const b = new Float32Array(size);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      r[p] = data[i];
      g[p] = data[i + 1];
      b[p] = data[i + 2];
    }
    return { r, g, b, width, height };
  }

  function renderGrayscale(canvas, arr, width, height) {
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(width, height);
    for (let i = 0, p = 0; p < arr.length; i += 4, p++) {
      const v = clamp(Math.round(arr[p]), 0, 255);
      img.data[i] = v; img.data[i + 1] = v; img.data[i + 2] = v; img.data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  function normalizeToBytes(arr) {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < arr.length; i++) { const v = arr[i]; if (v < min) min = v; if (v > max) max = v; }
    const out = new Uint8ClampedArray(arr.length);
    if (max === min) {
      const val = clamp(Math.round(clamp(min, 0, 255)), 0, 255);
      out.fill(val);
      return { bytes: out, min, max };
    }
    const inv = 255 / (max - min);
    for (let i = 0; i < arr.length; i++) out[i] = clamp(Math.round((arr[i] - min) * inv), 0, 255);
    return { bytes: out, min, max };
  }

  function renderBytesToCanvas(canvas, bytes, width, height) {
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(width, height);
    for (let i = 0, p = 0; p < bytes.length; i += 4, p++) {
      const v = bytes[p];
      img.data[i] = v; img.data[i + 1] = v; img.data[i + 2] = v; img.data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  function downsampleTo(canvas, outSize) {
    const tmp = document.createElement('canvas');
    tmp.width = outSize; tmp.height = outSize;
    const ctx = tmp.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(canvas, 0, 0, outSize, outSize);
    return tmp;
  }

  // Convolution helpers
  // inp: Float32Array length W*H, ker: number[9] in row-major 3x3
  function conv2dSingle(inp, W, H, ker, samePadding) {
    const out = new Float32Array(W * H);
    const k = ker; // alias
    const get = (x, y) => inp[y * W + x];
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let acc = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const ix = x + kx;
            const iy = y + ky;
            let v = 0;
            if (samePadding) {
              if (ix >= 0 && ix < W && iy >= 0 && iy < H) v = get(ix, iy); else v = 0;
            } else {
              if (ix < 0 || ix >= W || iy < 0 || iy >= H) continue;
              v = get(ix, iy);
            }
            const kk = k[(ky + 1) * 3 + (kx + 1)];
            acc += v * kk;
          }
        }
        out[y * W + x] = acc;
      }
    }
    return out;
  }

  function convolveRGB(R, G, B, W, H, kernel, samePadding, relu) {
    const or = conv2dSingle(R, W, H, kernel.r, samePadding);
    const og = conv2dSingle(G, W, H, kernel.g, samePadding);
    const ob = conv2dSingle(B, W, H, kernel.b, samePadding);
    const out = new Float32Array(W * H);
    for (let i = 0; i < out.length; i++) {
      let v = or[i] + og[i] + ob[i];
      if (relu && v < 0) v = 0;
      out[i] = v;
    }
    return out;
  }

  function randomNormalizedKernel3x3() {
    // Create 27 values ~N(0,1), subtract mean, normalize by L1
    const vals = new Array(27);
    let sum = 0;
    for (let i = 0; i < 27; i++) { const v = gaussian(); vals[i] = v; sum += v; }
    const mean = sum / 27;
    let l1 = 0;
    for (let i = 0; i < 27; i++) { vals[i] -= mean; l1 += Math.abs(vals[i]); }
    const s = l1 === 0 ? 1 : l1;
    for (let i = 0; i < 27; i++) vals[i] /= s;
    return { r: vals.slice(0, 9), g: vals.slice(9, 18), b: vals.slice(18, 27) };
  }

  function gaussian() {
    // Box-Muller transform
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  function classicKernelSet2D() {
    return [
      { name: 'Edge', k: [ -1,-1,-1, -1, 8,-1, -1,-1,-1 ] },
      { name: 'Sharpen', k: [ 0,-1, 0, -1, 5,-1, 0,-1,0 ] },
      { name: 'Box blur', k: [ 1,1,1, 1,1,1, 1,1,1 ].map(v=>v/9) },
      { name: 'Gaussian blur', k: [ 1,2,1, 2,4,2, 1,2,1 ].map(v=>v/16) },
      { name: 'Emboss', k: [ -2,-1,0, -1,1,1, 0,1,2 ] },
      { name: 'Sobel X', k: [ -1,0,1, -2,0,2, -1,0,1 ] },
      { name: 'Sobel Y', k: [ -1,-2,-1, 0,0,0, 1,2,1 ] },
      { name: 'Outline', k: [ 0,-1,0, -1,4,-1, 0,-1,0 ] },
    ];
  }

  function getKernels(count, mode) {
    const ks = [];
    if (mode === 'classic') {
      const base = classicKernelSet2D();
      for (let i = 0; i < Math.min(count, base.length); i++) {
        const k2 = base[i].k;
        // Same kernel repeated across RGB for simplicity
        ks.push({ r: k2.slice(), g: k2.slice(), b: k2.slice(), name: base[i].name });
      }
      for (let i = base.length; i < count; i++) ks.push({ ...randomNormalizedKernel3x3(), name: 'Random' });
    } else {
      for (let i = 0; i < count; i++) ks.push({ ...randomNormalizedKernel3x3(), name: 'Random' });
    }
    return ks;
  }

  function weightToColor(v, vmax) {
    // Diverging gradient: blue (neg) -> white -> red (pos)
    const t = vmax > 0 ? clamp((v + vmax) / (2 * vmax), 0, 1) : 0.5;
    // Interpolate between blue (#4e6cf6) and white (#ffffff) and red (#f7768e)
    if (t < 0.5) {
      const u = t / 0.5; // 0..1 from blue to white
      return lerpColor('#4e6cf6', '#ffffff', u);
    } else {
      const u = (t - 0.5) / 0.5; // 0..1 from white to red
      return lerpColor('#ffffff', '#f7768e', u);
    }
  }

  function lerpColor(a, b, t) {
    const ca = hexToRgb(a), cb = hexToRgb(b);
    const r = Math.round(ca.r + (cb.r - ca.r) * t);
    const g = Math.round(ca.g + (cb.g - ca.g) * t);
    const b2 = Math.round(ca.b + (cb.b - ca.b) * t);
    return `rgb(${r}, ${g}, ${b2})`;
  }

  function hexToRgb(hex) {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return m ? { r: parseInt(m[1], 16), g: parseInt(m[2], 16), b: parseInt(m[3], 16) } : { r: 255, g: 255, b: 255 };
  }

  function renderKernels(kernels) {
    kernelsContainer.innerHTML = '';
    kernels.forEach((ker, idx) => {
      const card = document.createElement('div');
      card.className = 'kernel-card';
      const title = document.createElement('div');
      title.className = 'kernel-title';
      title.textContent = `Kernel ${idx + 1}${ker.name ? ' — ' + ker.name : ''}`;
      card.appendChild(title);

      const { vmax } = kernelAbsMax(ker);

      const grid = document.createElement('div');
      grid.className = 'kernel-grid';
      grid.appendChild(renderKernelMatrix(ker.r, 'R', vmax));
      grid.appendChild(renderKernelMatrix(ker.g, 'G', vmax));
      grid.appendChild(renderKernelMatrix(ker.b, 'B', vmax));
      card.appendChild(grid);

      // Legend
      const legendWrap = document.createElement('div');
      legendWrap.style.marginTop = '8px';
      const legend = document.createElement('div');
      legend.className = 'legend';
      for (let i = 0; i < 10; i++) {
        const v = -1 + (i / 9) * 2; // -1..1
        const seg = document.createElement('div');
        seg.style.background = weightToColor(v, 1);
        legend.appendChild(seg);
      }
      const labs = document.createElement('div');
      labs.className = 'legend-labels';
      labs.innerHTML = '<span>−</span><span>0</span><span>+</span>';
      legendWrap.appendChild(legend);
      legendWrap.appendChild(labs);
      card.appendChild(legendWrap);

      kernelsContainer.appendChild(card);
    });
  }

  function kernelAbsMax(ker) {
    let vmax = 0;
    for (const v of ker.r) vmax = Math.max(vmax, Math.abs(v));
    for (const v of ker.g) vmax = Math.max(vmax, Math.abs(v));
    for (const v of ker.b) vmax = Math.max(vmax, Math.abs(v));
    return { vmax };
  }

  function renderKernelMatrix(k, label, vmax) {
    const wrap = document.createElement('div');
    const lab = document.createElement('div');
    lab.className = 'kernel-label';
    lab.textContent = label;
    wrap.appendChild(lab);

    const grid = document.createElement('div');
    grid.className = 'kernel-matrix';
    for (let i = 0; i < 9; i++) {
      const cell = document.createElement('div');
      cell.className = 'kernel-cell';
      cell.style.background = weightToColor(k[i], vmax);
      cell.title = k[i].toFixed(3);
      cell.textContent = formatWeight(k[i]);
      grid.appendChild(cell);
    }
    wrap.appendChild(grid);
    return wrap;
  }

  function formatWeight(v) {
    const s = v.toFixed(2);
    // Align minus sign visually: use thin space for positives
    return (v >= 0 ? ' ' : '') + s;
  }

  function renderOutputs() {
    if (!R || kernels.length === 0) return;
    outputsContainer.innerHTML = '';
    const relu = false;
    const same = false; // valid convolution (no padding)

    kernels.forEach((ker, idx) => {
      const out = convolveRGB(R, G, B, W, H, ker, same, relu);
      const { bytes, min, max } = normalizeToBytes(out);
      const fig = document.createElement('figure');
      const cap = document.createElement('figcaption');
      cap.textContent = `Output ${idx + 1} (min ${min.toFixed(2)}, max ${max.toFixed(2)})`;
      const c = document.createElement('canvas');
      renderBytesToCanvas(c, bytes, W, H);
      fig.appendChild(cap);
      fig.appendChild(c);
      outputsContainer.appendChild(fig);
    });
  }

  // ---------- Practice section ----------
  function buildPracticeFromCurrent() {
    const size = 16; // small for easy manual work
    const tmp = downsampleTo(imageCanvas, size);
    const ch = extractRGB(tmp);
    pW = ch.width; pH = ch.height; pR = ch.r; pG = ch.g; pB = ch.b;
    // Render practice channels scaled up for visibility
    renderGrayscale(pRCanvas, pR, pW, pH);
    renderGrayscale(pGCanvas, pG, pW, pH);
    renderGrayscale(pBCanvas, pB, pW, pH);
    renderPracticeImageMatrices();
  }

  function getPracticeKernels(k) {
    const ks = [];
    for (let i = 0; i < k; i++) ks.push({ ...randomNormalizedKernel3x3(), name: 'Random' });
    return ks;
  }

  function generateNumpyCode() {
    // image as HxWx3 uint8 (but we write as float32 for downstream math). Values 0..255
    const lines = [];
    lines.push('import numpy as np');
    lines.push('');
    lines.push(`# Image shape: (${pH}, ${pW}, 3), values in [0, 255]`);
    lines.push('image = np.array([');
    for (let y = 0; y < pH; y++) {
      const row = [];
      for (let x = 0; x < pW; x++) {
        const idx = y * pW + x;
        row.push(`[${Math.round(pR[idx])}, ${Math.round(pG[idx])}, ${Math.round(pB[idx])}]`);
      }
      lines.push('  [' + row.join(', ') + '],');
    }
    lines.push(`], dtype=np.float32)`);
    lines.push('');
    const K = pKernels.length;
    lines.push(`# Kernels shape: (K, 3, 3, 3) with K = ${K}`);
    lines.push('kernels = np.array([');
    for (let i = 0; i < K; i++) {
      const k = pKernels[i];
      function rowOf(arr, y) { return `${arr[y*3+0].toFixed(4)}, ${arr[y*3+1].toFixed(4)}, ${arr[y*3+2].toFixed(4)}`; }
      lines.push('  [');
      // R
      lines.push('    [');
      for (let y = 0; y < 3; y++) lines.push(`      [${rowOf(k.r, y)}],`);
      lines.push('    ],');
      // G
      lines.push('    [');
      for (let y = 0; y < 3; y++) lines.push(`      [${rowOf(k.g, y)}],`);
      lines.push('    ],');
      // B
      lines.push('    [');
      for (let y = 0; y < 3; y++) lines.push(`      [${rowOf(k.b, y)}],`);
      lines.push('    ]');
      lines.push(i === K - 1 ? '  ]' : '  ],');
    }
    lines.push('], dtype=np.float32)');
    lines.push('');
    lines.push('# Task: Compute K outputs with stride=1, no padding (valid).');
    lines.push('# Each output is sum over channels of 2D convs:');
    lines.push('# out[k] = conv2d(image[...,0], kernels[k,0]) + conv2d(image[...,1], kernels[k,1]) + conv2d(image[...,2], kernels[k,2])');
    lines.push('# You can implement conv2d manually with nested loops, or using scipy.signal.correlate2d.');
    return lines.join('\n');
  }

  function renderPracticeOutputs() {
    practiceOutputs.innerHTML = '';
    const relu = false, same = false;
    pKernels.forEach((ker, idx) => {
      const out = convolveRGB(pR, pG, pB, pW, pH, ker, same, relu);
      // Build a valid (no-padding) matrix by taking only fully-covered positions
      const H2 = Math.max(0, pH - 2);
      const W2 = Math.max(0, pW - 2);
      let min = Infinity, max = -Infinity;
      for (let y = 1; y < pH - 1; y++) {
        for (let x = 1; x < pW - 1; x++) {
          const v = out[y * pW + x];
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
      const fig = document.createElement('figure');
      const cap = document.createElement('figcaption');
      cap.textContent = `Output ${idx + 1} matrix (shape ${H2}×${W2}) — min ${isFinite(min)?min.toFixed(2):'NA'}, max ${isFinite(max)?max.toFixed(2):'NA'}`;
      fig.appendChild(cap);
      const wrap = document.createElement('div');
      wrap.className = 'matrix-wrap';
      const table = document.createElement('table');
      table.className = 'matrix';
      for (let y = 1; y < pH - 1; y++) {
        const tr = document.createElement('tr');
        for (let x = 1; x < pW - 1; x++) {
          const td = document.createElement('td');
          const v = out[y * pW + x];
          td.textContent = Math.round(v).toString();
          tr.appendChild(td);
        }
        table.appendChild(tr);
      }
      wrap.appendChild(table);
      fig.appendChild(wrap);
      practiceOutputs.appendChild(fig);
    });
    practiceOutputs.style.display = practiceRevealed ? '' : 'none';
  }

  function regeneratePractice() {
    pKernels = getPracticeKernels(3); // fixed to three kernels for practice
    renderPracticeKernelMatrices();
    renderPracticeOutputs();
  }

  function renderPracticeImageMatrices() {
    if (!practiceImageMatrices) return;
    practiceImageMatrices.innerHTML = '';
    const items = [
      { name: 'R', arr: pR },
      { name: 'G', arr: pG },
      { name: 'B', arr: pB },
    ];
    items.forEach(({ name, arr }) => {
      const fig = document.createElement('figure');
      const cap = document.createElement('figcaption');
      cap.textContent = `${name} channel values (${pH}×${pW})`;
      fig.appendChild(cap);
      const wrap = document.createElement('div');
      wrap.className = 'matrix-wrap';
      const table = document.createElement('table');
      table.className = 'matrix';
      for (let y = 0; y < pH; y++) {
        const tr = document.createElement('tr');
        for (let x = 0; x < pW; x++) {
          const td = document.createElement('td');
          const v = Math.round(arr[y * pW + x]);
          td.textContent = String(v);
          tr.appendChild(td);
        }
        table.appendChild(tr);
      }
      wrap.appendChild(table);
      fig.appendChild(wrap);
      practiceImageMatrices.appendChild(fig);
    });
  }

  function renderPracticeKernelMatrices() {
    if (!practiceKernelMatrices) return;
    practiceKernelMatrices.innerHTML = '';
    const K = Math.min(3, pKernels.length);
    for (let i = 0; i < K; i++) {
      const ker = pKernels[i];
      const card = document.createElement('div');
      card.className = 'kernel-card';
      const title = document.createElement('div');
      title.className = 'kernel-title';
      title.textContent = `Kernel ${i + 1}`;
      card.appendChild(title);
      const { vmax } = kernelAbsMax(ker);
      const grid = document.createElement('div');
      grid.className = 'kernel-grid';
      grid.appendChild(renderKernelMatrix(ker.r, 'R', vmax));
      grid.appendChild(renderKernelMatrix(ker.g, 'G', vmax));
      grid.appendChild(renderKernelMatrix(ker.b, 'B', vmax));
      card.appendChild(grid);
      practiceKernelMatrices.appendChild(card);
    }
  }

  function updateAll() {
    renderGrayscale(rCanvas, R, W, H);
    renderGrayscale(gCanvas, G, W, H);
    renderGrayscale(bCanvas, B, W, H);
    renderKernels(kernels);
    renderOutputs();
  }

  function regenerateKernels() {
    const count = parseInt(kernelsRange.value, 10);
    kernels = getKernels(count, kernelSet.value);
    renderKernels(kernels);
    renderOutputs();
  }

  // Event wiring
  imageFile.addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    await drawImageFileToCanvas(file, imageCanvas, 256);
    const ch = extractRGB(imageCanvas);
    ({ r: R, g: G, b: B, width: W, height: H } = ch);
    updateAll();
    // rebuild practice from current image too
    buildPracticeFromCurrent();
    regeneratePractice();
  });

  useSampleBtn.addEventListener('click', () => {
    drawSampleToCanvas(imageCanvas, 256);
    const ch = extractRGB(imageCanvas);
    ({ r: R, g: G, b: B, width: W, height: H } = ch);
    updateAll();
    buildPracticeFromCurrent();
    regeneratePractice();
  });

  kernelsRange.addEventListener('input', () => {
    kernelsCount.textContent = kernelsRange.value;
    regenerateKernels();
  });
  kernelSet.addEventListener('change', regenerateKernels);
  randomizeBtn.addEventListener('click', regenerateKernels);
  // Practice UI events
  practiceRegenerate.addEventListener('click', regeneratePractice);
  practiceReveal.addEventListener('click', () => {
    practiceRevealed = !practiceRevealed;
    practiceReveal.textContent = practiceRevealed ? 'Hide results' : 'Reveal results';
    renderPracticeOutputs();
  });
  copyCodeBtn.addEventListener('click', async () => {
    try {
      const code = generateNumpyCode();
      await navigator.clipboard.writeText(code);
      copyCodeBtn.textContent = 'Copied!';
      setTimeout(() => (copyCodeBtn.textContent = 'Copy NumPy code'), 1200);
    } catch (e) {
      // Fallback: select text
      copyCodeBtn.textContent = 'Copy failed: use script.js generateNumpyCode()';
      setTimeout(() => (copyCodeBtn.textContent = 'Copy NumPy code'), 1800);
    }
  });

  // Initialize
  function init() {
    kernelsCount.textContent = kernelsRange.value;
    drawSampleToCanvas(imageCanvas, 256);
    const ch = extractRGB(imageCanvas);
    ({ r: R, g: G, b: B, width: W, height: H } = ch);
    kernels = getKernels(parseInt(kernelsRange.value, 10), kernelSet.value);
    updateAll();
    // practice init
    buildPracticeFromCurrent();
    pKernels = getPracticeKernels(3);
    renderPracticeKernelMatrices();
    renderPracticeOutputs();
  }

  init();
})();

#!/usr/bin/env python3
#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Execution Graph Visualizer
===========================

Generates an interactive HTML visualization from an nnScaler execution
plan JSON dump (``execplan.pkl.json``).

Usage::

    python visualize_graph.py <path/to/execplan.pkl.json> [-o output.html]

Then open the generated HTML in a browser.  The **Graph** tab renders every
operator as a node with dependency edges, using a topological layer layout
and Canvas rendering for performance with large models (3000+ ops).
"""

import argparse
import json
import html as _html
import os
import sys

# ---------------------------------------------------------------------------
# The entire visualizer is a single self-contained HTML string.  Python just
# injects the JSON data and writes the file.
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>nnScaler Execution Graph – __TITLE__</title>
<style>
:root {
  --bg:#0d1117;--sf:#161b22;--bd:#30363d;--tx:#e6edf3;--t2:#8b949e;--ac:#58a6ff;
  --fw:#3fb950;--bw:#f0883e;--cm:#a371f7;--da:#79c0ff;--rd:#f778ba;--un:#8b949e;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
     background:var(--bg);color:var(--tx);font-size:14px;overflow:hidden;height:100vh;display:flex;flex-direction:column}
.hdr{background:var(--sf);border-bottom:1px solid var(--bd);padding:10px 20px;display:flex;align-items:center;gap:14px;flex-shrink:0}
.hdr h1{font-size:16px;font-weight:600}
.hdr .m{color:var(--t2);font-size:12px}
.tabs{display:flex;background:var(--sf);border-bottom:1px solid var(--bd);padding:0 20px;flex-shrink:0}
.tab{padding:8px 16px;cursor:pointer;color:var(--t2);border-bottom:2px solid transparent;font-size:13px;user-select:none}
.tab:hover{color:var(--tx)}.tab.active{color:var(--ac);border-bottom-color:var(--ac)}
.main{flex:1;overflow:hidden;position:relative}
.panel{display:none;width:100%;height:100%}.panel.active{display:flex;flex-direction:column}

/* -- Graph tab -- */
#graph-panel{position:relative}
#graph-canvas{width:100%;height:100%;cursor:grab}
#graph-canvas.dragging{cursor:grabbing}
#graph-toolbar{position:absolute;top:12px;left:12px;display:flex;gap:6px;z-index:2}
#graph-toolbar button{background:var(--sf);border:1px solid var(--bd);color:var(--tx);padding:6px 10px;
  border-radius:6px;cursor:pointer;font-size:12px}
#graph-toolbar button:hover{background:var(--bd)}
#graph-toolbar select{background:var(--sf);border:1px solid var(--bd);color:var(--tx);padding:6px 8px;
  border-radius:6px;font-size:12px}
#graph-info{position:absolute;top:12px;right:12px;background:var(--sf);border:1px solid var(--bd);
  border-radius:8px;padding:12px 16px;max-width:380px;max-height:50vh;overflow-y:auto;font-size:12px;z-index:2;display:none}
#graph-info h4{margin-bottom:6px;font-size:13px}
#graph-info .close{position:absolute;top:8px;right:10px;cursor:pointer;color:var(--t2);font-size:16px}
#graph-legend{position:absolute;bottom:12px;left:12px;background:var(--sf);border:1px solid var(--bd);
  border-radius:8px;padding:10px 14px;display:flex;gap:14px;font-size:11px;z-index:2}
.leg-item{display:flex;align-items:center;gap:4px}
.leg-dot{width:10px;height:10px;border-radius:2px}

/* -- Summary tab -- */
#summary-panel{overflow-y:auto;padding:20px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;margin-bottom:20px}
.card{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:14px}
.card .lb{color:var(--t2);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.card .vl{font-size:22px;font-weight:600;margin-top:2px}
.card .vl.fw{color:var(--fw)}.card .vl.bw{color:var(--bw)}.card .vl.cm{color:var(--cm)}
.card .vl.rd{color:var(--rd)}.card .vl.da{color:var(--da)}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--bd)}
th{background:var(--sf);color:var(--t2);font-weight:600;position:sticky;top:0;z-index:1}
tr:hover{background:rgba(88,166,255,.05)}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;font-weight:600;color:#fff}
.badge-forward,.badge-forward_segment{background:var(--fw)}.badge-backward,.badge-backward_segment{background:var(--bw)}
.badge-communication{background:var(--cm)}.badge-data{background:var(--da)}.badge-weight_reducer{background:var(--rd)}
details>summary{cursor:pointer;color:var(--ac);font-size:13px;margin:10px 0 4px}
.scroll-t{max-height:60vh;overflow-y:auto}
.search-input{background:var(--bg);border:1px solid var(--bd);color:var(--tx);padding:7px 10px;border-radius:6px;width:320px;font-size:12px;margin-bottom:10px}
</style>
</head>
<body>
<div class="hdr"><h1>nnScaler Execution Graph</h1><span class="m" id="hdr-m"></span></div>
<div class="tabs" id="tabs"></div>
<div class="main" id="main"></div>

<script>
"use strict";
const D = __JSON_DATA__;

/* ── colours ── */
const CAT_COL = {
  forward:'#3fb950',forward_segment:'#3fb950',
  backward:'#f0883e',backward_segment:'#f0883e',
  communication:'#a371f7',data:'#79c0ff',weight_reducer:'#f778ba',unknown:'#8b949e'
};
const catC = c => CAT_COL[c]||CAT_COL.unknown;
const badge = c => `<span class="badge badge-${c}">${c}</span>`;
const esc = s => s==null?'':String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');

/* ── state ── */
let curTab = 'graph';
let curDev = Object.keys(D.per_device)[0];
let graphState = null;  // lazily initialized per device

/* ── tabs ── */
const TABS = [['graph','Graph'],['summary','Summary']];
function renderTabs() {
  document.getElementById('tabs').innerHTML = TABS.map(([id,lb]) =>
    `<div class="tab ${curTab===id?'active':''}" data-t="${id}">${lb}</div>`
  ).join('') +
  `<div style="margin-left:auto;display:flex;align-items:center;gap:8px;padding:0 12px">` +
  `<span style="color:var(--t2);font-size:11px">Device:</span>` +
  Object.keys(D.per_device).map(d =>
    `<div class="tab ${curDev===d?'active':''}" data-d="${d}" style="padding:6px 10px">${d}</div>`
  ).join('') + `</div>`;
  document.querySelectorAll('.tab[data-t]').forEach(t => t.onclick = () => { curTab=t.dataset.t; render(); });
  document.querySelectorAll('.tab[data-d]').forEach(t => t.onclick = () => { curDev=t.dataset.d; graphState=null; render(); });
}

function render() {
  const m = D.metadata;
  document.getElementById('hdr-m').textContent = `${m.train?'Training':'Inference'} · ${m.num_devices} devices`;
  renderTabs();
  const main = document.getElementById('main');
  main.innerHTML = TABS.map(([id]) => `<div class="panel ${curTab===id?'active':''}" id="${id}-panel"></div>`).join('');
  if (curTab === 'graph') initGraph();
  if (curTab === 'summary') renderSummary();
}

/* ═══════════════════════════════════════════════════════════════
   GRAPH TAB – Canvas-based node/edge rendering with zoom/pan
   ═══════════════════════════════════════════════════════════════ */

function getSegmentData() {
  /* Find the first segment node (forward or backward) that has sub_nodes */
  const dev = D.per_device[curDev];
  const segments = [];
  for (const nid of dev.execution_order) {
    const n = dev.nodes[nid];
    if (n.sub_nodes && n.sub_nodes.length > 0) segments.push({nid, node: n});
  }
  return segments;
}

function initGraph() {
  const panel = document.getElementById('graph-panel');
  const segments = getSegmentData();
  const segOptions = segments.map((s,i) => `<option value="${i}">${s.node.category} (${s.node.sub_nodes.length} ops)</option>`).join('');

  panel.innerHTML = `
    <div id="graph-toolbar">
      <select id="seg-select">${segOptions}</select>
      <button id="btn-fit">Fit</button>
      <button id="btn-zin">+</button>
      <button id="btn-zout">−</button>
      <span style="color:var(--t2);font-size:11px;padding:6px" id="graph-stats"></span>
    </div>
    <canvas id="graph-canvas"></canvas>
    <div id="graph-info"></div>
    <div id="graph-legend">
      <div class="leg-item"><div class="leg-dot" style="background:var(--fw)"></div>Forward</div>
      <div class="leg-item"><div class="leg-dot" style="background:var(--bw)"></div>Backward</div>
      <div class="leg-item"><div class="leg-dot" style="background:var(--cm)"></div>Comm</div>
      <div class="leg-item"><div class="leg-dot" style="background:var(--da)"></div>Data</div>
      <div class="leg-item"><div class="leg-dot" style="background:var(--rd)"></div>Reducer</div>
    </div>`;

  const canvas = document.getElementById('graph-canvas');
  const ctx = canvas.getContext('2d');
  let W, H;
  let nodes = [], edges = [], layers = [];
  let cam = {x:0, y:0, z:1};  // camera: translate x,y + zoom z
  let drag = null; // {sx,sy,cx,cy}
  let hoverIdx = -1;

  const NW = 14, NH = 14, PAD_X = 6, PAD_Y = 8;  // node size & padding between nodes

  function resize() {
    W = panel.clientWidth; H = panel.clientHeight;
    canvas.width = W * devicePixelRatio;
    canvas.height = H * devicePixelRatio;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
    paint();
  }

  function loadSegment(idx) {
    const seg = segments[idx];
    if (!seg) return;
    const sn = seg.node.sub_nodes;
    const se = seg.node.sub_edges || [];
    document.getElementById('graph-stats').textContent = `${sn.length} nodes · ${se.length} edges`;

    // Build adjacency & compute topological layers (BFS from sources)
    const inDeg = new Int32Array(sn.length);
    const adjFwd = new Array(sn.length);
    for (let i = 0; i < sn.length; i++) adjFwd[i] = [];
    for (const [u,v] of se) {
      adjFwd[u].push(v);
      inDeg[v]++;
    }
    // BFS layering
    let queue = [];
    const layer = new Int32Array(sn.length);
    for (let i = 0; i < sn.length; i++) if (inDeg[i] === 0) { queue.push(i); layer[i] = 0; }
    let maxLayer = 0;
    let qi = 0;
    while (qi < queue.length) {
      const u = queue[qi++];
      for (const v of adjFwd[u]) {
        inDeg[v]--;
        const nl = layer[u] + 1;
        if (nl > layer[v]) layer[v] = nl;
        if (nl > maxLayer) maxLayer = nl;
        if (inDeg[v] === 0) queue.push(v);
      }
    }

    // Group by layer
    layers = new Array(maxLayer + 1);
    for (let i = 0; i <= maxLayer; i++) layers[i] = [];
    for (let i = 0; i < sn.length; i++) layers[layer[i]].push(i);

    // Assign positions: x = layer, y = position within layer
    nodes = sn.map((n, i) => ({
      ...n, idx: i,
      lyr: layer[i],
      x: 0, y: 0,  // set below
      col: catC(n.category),
    }));

    for (let l = 0; l <= maxLayer; l++) {
      const memberIndices = layers[l];
      for (let p = 0; p < memberIndices.length; p++) {
        const ni = memberIndices[p];
        nodes[ni].x = l * (NW + PAD_X);
        nodes[ni].y = p * (NH + PAD_Y);
      }
    }

    edges = se;
    fitView();
  }

  function fitView() {
    if (nodes.length === 0) return;
    let minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
    for (const n of nodes) {
      if (n.x < minX) minX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.x + NW > maxX) maxX = n.x + NW;
      if (n.y + NH > maxY) maxY = n.y + NH;
    }
    const gw = maxX - minX, gh = maxY - minY;
    const zx = (W - 40) / gw, zy = (H - 80) / gh;
    cam.z = Math.min(zx, zy, 4);
    cam.x = -minX * cam.z + (W - gw * cam.z) / 2;
    cam.y = -minY * cam.z + (H - gh * cam.z) / 2;
    paint();
  }

  function toScreen(gx, gy) { return [gx * cam.z + cam.x, gy * cam.z + cam.y]; }
  function toGraph(sx, sy) { return [(sx - cam.x) / cam.z, (sy - cam.y) / cam.z]; }

  function paint() {
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.translate(cam.x, cam.y);
    ctx.scale(cam.z, cam.z);

    // Draw edges (thin lines)
    ctx.globalAlpha = 0.15;
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 0.5 / cam.z;
    ctx.beginPath();
    for (const [u,v] of edges) {
      const nu = nodes[u], nv = nodes[v];
      ctx.moveTo(nu.x + NW, nu.y + NH/2);
      ctx.lineTo(nv.x, nv.y + NH/2);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Draw nodes
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      ctx.fillStyle = n.col;
      if (i === hoverIdx) {
        ctx.fillRect(n.x - 1, n.y - 1, NW + 2, NH + 2);
      } else {
        ctx.fillRect(n.x, n.y, NW, NH);
      }
    }

    // Highlight hover edges
    if (hoverIdx >= 0) {
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5 / cam.z;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      for (const [u,v] of edges) {
        if (u === hoverIdx || v === hoverIdx) {
          const nu = nodes[u], nv = nodes[v];
          ctx.moveTo(nu.x + NW, nu.y + NH/2);
          ctx.lineTo(nv.x, nv.y + NH/2);
        }
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Draw labels when zoomed in enough
    if (cam.z > 2) {
      ctx.fillStyle = '#fff';
      ctx.font = `${Math.max(3, 5/cam.z*cam.z)}px monospace`;
      ctx.textBaseline = 'middle';
      for (const n of nodes) {
        ctx.fillText(n.name, n.x + NW + 2, n.y + NH/2);
      }
    }

    ctx.restore();
  }

  // Hit test
  function hitTest(sx, sy) {
    const [gx, gy] = toGraph(sx, sy);
    for (let i = nodes.length - 1; i >= 0; i--) {
      const n = nodes[i];
      if (gx >= n.x && gx <= n.x + NW && gy >= n.y && gy <= n.y + NH) return i;
    }
    return -1;
  }

  // Events
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    cam.x = mx - (mx - cam.x) * factor;
    cam.y = my - (my - cam.y) * factor;
    cam.z *= factor;
    paint();
  }, {passive:false});

  canvas.addEventListener('mousedown', e => {
    drag = {sx: e.clientX, sy: e.clientY, cx: cam.x, cy: cam.y};
    canvas.classList.add('dragging');
  });
  window.addEventListener('mousemove', e => {
    if (drag) {
      cam.x = drag.cx + (e.clientX - drag.sx);
      cam.y = drag.cy + (e.clientY - drag.sy);
      paint();
    } else {
      const rect = canvas.getBoundingClientRect();
      const newHover = hitTest(e.clientX - rect.left, e.clientY - rect.top);
      if (newHover !== hoverIdx) { hoverIdx = newHover; paint(); }
    }
  });
  window.addEventListener('mouseup', () => { drag = null; canvas.classList.remove('dragging'); });

  canvas.addEventListener('click', e => {
    const rect = canvas.getBoundingClientRect();
    const idx = hitTest(e.clientX - rect.left, e.clientY - rect.top);
    if (idx >= 0) showInfo(idx); else hideInfo();
  });

  function showInfo(idx) {
    const n = nodes[idx];
    const inEdges = edges.filter(([u,v]) => v === idx);
    const outEdges = edges.filter(([u,v]) => u === idx);
    const info = document.getElementById('graph-info');
    info.style.display = 'block';
    info.innerHTML = `<span class="close" id="info-close">×</span>
      <h4 style="color:${n.col}">${esc(n.name)}</h4>
      <div style="margin:6px 0">
        <div>${badge(n.category)}</div>
        <div style="margin-top:4px;color:var(--t2)">CID: ${n.cid}</div>
        <div style="color:var(--t2)">Layer: ${n.lyr}</div>
        <div style="font-family:monospace;margin-top:4px;word-break:break-all">${esc(n.signature)}</div>
      </div>
      <div style="margin-top:8px">
        <strong>Inputs from:</strong> ${inEdges.length === 0 ? '<em>none (source)</em>' : inEdges.map(([u]) =>
          `<span style="color:${nodes[u].col};cursor:pointer" onclick="window._graphClick(${u})">${esc(nodes[u].name)} [${nodes[u].cid}]</span>`
        ).join(', ')}
      </div>
      <div style="margin-top:4px">
        <strong>Outputs to:</strong> ${outEdges.length === 0 ? '<em>none (sink)</em>' : outEdges.map(([,v]) =>
          `<span style="color:${nodes[v].col};cursor:pointer" onclick="window._graphClick(${v})">${esc(nodes[v].name)} [${nodes[v].cid}]</span>`
        ).join(', ')}
      </div>
      ${n.input_tids && n.input_tids.length ? `<div style="margin-top:4px;color:var(--t2);font-size:11px">Input TIDs: ${n.input_tids.join(', ')}</div>` : ''}
      ${n.output_tids && n.output_tids.length ? `<div style="color:var(--t2);font-size:11px">Output TIDs: ${n.output_tids.join(', ')}</div>` : ''}
    `;
    document.getElementById('info-close').onclick = hideInfo;
  }
  window._graphClick = idx => { hoverIdx = idx; showInfo(idx); centerOn(idx); };

  function centerOn(idx) {
    const n = nodes[idx];
    cam.x = W/2 - n.x * cam.z;
    cam.y = H/2 - n.y * cam.z;
    paint();
  }

  function hideInfo() { document.getElementById('graph-info').style.display = 'none'; }

  // Toolbar
  document.getElementById('seg-select').onchange = e => loadSegment(+e.target.value);
  document.getElementById('btn-fit').onclick = fitView;
  document.getElementById('btn-zin').onclick = () => { cam.z *= 1.3; paint(); };
  document.getElementById('btn-zout').onclick = () => { cam.z /= 1.3; paint(); };

  window.addEventListener('resize', resize);
  resize();
  if (segments.length > 0) loadSegment(0);
}


/* ═══════════════════════════════════════════════════════════════
   SUMMARY TAB
   ═══════════════════════════════════════════════════════════════ */

function renderSummary() {
  const panel = document.getElementById('summary-panel');
  const dev = D.per_device[curDev];
  const s = dev.summary;
  const totalSub = Object.values(dev.nodes).reduce((a,n) => a + (n.sub_nodes?.length||0), 0);

  let h = `<div class="cards">
    <div class="card"><div class="lb">Devices</div><div class="vl">${D.metadata.num_devices}</div></div>
    <div class="card"><div class="lb">Top-Level Nodes</div><div class="vl">${s.total_nodes}</div></div>
    <div class="card"><div class="lb">Sub Operations</div><div class="vl">${totalSub.toLocaleString()}</div></div>
    <div class="card"><div class="lb">Forward</div><div class="vl fw">${s.forward_nodes}</div></div>
    <div class="card"><div class="lb">Backward</div><div class="vl bw">${s.backward_nodes}</div></div>
    <div class="card"><div class="lb">Comm</div><div class="vl cm">${s.communication_nodes}</div></div>
    <div class="card"><div class="lb">Reducers</div><div class="vl rd">${s.weight_reducer_nodes}</div></div>
    <div class="card"><div class="lb">Parameters</div><div class="vl">${D.graph_attributes.length}</div></div>
    <div class="card"><div class="lb">Edges</div><div class="vl">${dev.edges.length}</div></div>
  </div>`;

  // Per-segment breakdown
  for (const nid of dev.execution_order) {
    const n = dev.nodes[nid];
    if (!n.sub_nodes || n.sub_nodes.length === 0) {
      h += `<div class="card" style="margin-bottom:8px;max-width:600px"><div class="lb">${esc(n.category)}</div>
        <div style="margin-top:4px">${badge(n.category)} ${esc(n.name)} <span style="color:var(--t2)">(ID ${nid})</span></div>
        ${n.reducer_weights ? `<div style="color:var(--t2);margin-top:2px">${n.reducer_weights.length} weights</div>` : ''}
      </div>`;
      continue;
    }
    const cats = {};
    n.sub_nodes.forEach(sn => { cats[sn.category] = (cats[sn.category]||0)+1; });
    h += `<details open><summary>${badge(n.category)} ${esc(n.name)} (ID ${nid}) – ${n.sub_nodes.length.toLocaleString()} ops, ${(n.sub_edges||[]).length.toLocaleString()} edges</summary>`;
    h += `<div class="cards" style="max-width:700px;margin:8px 0">`;
    for (const [cat, cnt] of Object.entries(cats).sort((a,b)=>b[1]-a[1]))
      h += `<div class="card"><div class="lb">${esc(cat)}</div><div class="vl" style="color:${catC(cat)}">${cnt.toLocaleString()}</div></div>`;
    h += '</div>';

    // Op frequency
    const freq = {};
    n.sub_nodes.forEach(sn => { const k = sn.signature||sn.name; freq[k]=(freq[k]||0)+1; });
    const sorted = Object.entries(freq).sort((a,b)=>b[1]-a[1]);
    h += `<details><summary>Op frequency (${sorted.length} unique)</summary>`;
    h += '<div class="scroll-t" style="max-height:250px"><table><tr><th>Operation</th><th>Count</th></tr>';
    sorted.forEach(([sig,cnt]) => h += `<tr><td style="font-family:monospace;font-size:11px">${esc(sig)}</td><td>${cnt}</td></tr>`);
    h += '</table></div></details>';

    // Sub-node search table
    h += `<details><summary>Browse sub-nodes</summary>
      <input class="search-input" placeholder="Filter…" oninput="filterTable(this)">
      <div class="scroll-t" style="max-height:300px"><table><tr><th>CID</th><th>Category</th><th>Name</th><th>Signature</th></tr>`;
    n.sub_nodes.forEach(sn => {
      h += `<tr data-s="${(sn.category+' '+sn.name+' '+sn.signature).toLowerCase()}">
        <td>${sn.cid}</td><td>${badge(sn.category)}</td><td>${esc(sn.name)}</td>
        <td style="font-family:monospace;font-size:11px">${esc(sn.signature)}</td></tr>`;
    });
    h += '</table></div></details></details>';
  }

  // Parameters
  h += `<details><summary>Parameters (${D.graph_attributes.length})</summary>
    <input class="search-input" placeholder="Filter…" oninput="filterTable(this)">
    <div class="scroll-t"><table><tr><th>TID</th><th>Name</th><th>Shape</th><th>Dtype</th><th>Grad</th></tr>`;
  D.graph_attributes.forEach(a => {
    h += `<tr data-s="${(a.name+' '+JSON.stringify(a.shape)).toLowerCase()}">
      <td>${a.tid}</td><td style="font-family:monospace;font-size:11px">${esc(a.name)}</td>
      <td>${JSON.stringify(a.shape)}</td><td>${esc(a.dtype)}</td><td>${a.requires_grad?'✓':''}</td></tr>`;
  });
  h += '</table></div></details>';

  panel.innerHTML = h;
}

window.filterTable = function(input) {
  const q = input.value.toLowerCase();
  const table = input.nextElementSibling?.querySelector?.('table') || input.parentElement.querySelector('table');
  if (!table) return;
  table.querySelectorAll('tr[data-s]').forEach(r => r.style.display = r.dataset.s.includes(q) ? '' : 'none');
};

render();
</script>
</body>
</html>"""


def generate_html(json_path: str, output_path: str) -> str:
    with open(json_path) as f:
        data = json.load(f)

    title = _html.escape(os.path.basename(os.path.dirname(json_path)) or 'Execution Graph')
    json_str = json.dumps(data, default=str)
    html_content = _HTML_TEMPLATE.replace('__TITLE__', title).replace('__JSON_DATA__', json_str)

    with open(output_path, 'w') as f:
        f.write(html_content)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Visualize an nnScaler execution plan JSON dump as interactive HTML.',
    )
    parser.add_argument('input', help='Path to execplan.pkl.json')
    parser.add_argument('-o', '--output', default=None,
                        help='Output HTML path (default: <input>.html)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'Error: {args.input} not found', file=sys.stderr)
        sys.exit(1)

    output = args.output or args.input + '.html'
    generate_html(args.input, output)
    print(f'Visualization written to: {output}')
    print(f'Open in browser: file://{os.path.abspath(output)}')


if __name__ == '__main__':
    main()

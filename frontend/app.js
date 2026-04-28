/* Feature Mining Platform - Front-end app */
const API = (path, opts={}) => fetch(path, opts).then(async r => {
  const ct = r.headers.get('content-type') || '';
  const data = ct.includes('application/json') ? await r.json() : await r.text();
  if (!r.ok) throw new Error(typeof data === 'string' ? data : (data.detail || JSON.stringify(data)));
  return data;
});

const toast = (msg, kind='') => {
  const el = document.createElement('div');
  el.className = `toast ${kind}`; el.textContent = msg;
  document.getElementById('toast').appendChild(el);
  requestAnimationFrame(() => el.classList.add('show'));
  setTimeout(() => { el.classList.remove('show'); setTimeout(() => el.remove(), 300); }, 3000);
};

const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));
const el = (tag, props={}, ...children) => {
  const e = document.createElement(tag);
  Object.entries(props).forEach(([k, v]) => {
    if (k === 'class') e.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(e.dataset, v);
    else e.setAttribute(k, v);
  });
  children.flat().forEach(c => e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
  return e;
};

// ===== Tabs =====
$$('.sidebar .nav').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.sidebar .nav').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    $$('.tab').forEach(t => t.classList.remove('active'));
    $('#tab-' + tab).classList.add('active');
    onTabActivate(tab);
  });
});

// ===== Health =====
API('/api/health').then(h => {
  $('#health').textContent = `✓ healthy · GPU: ${h.torch_cuda ? 'available' : 'cpu only'}`;
}).catch(e => $('#health').textContent = '✗ ' + e.message);

// ============================================================
// Caches
// ============================================================
const STATE = {
  datasets: [], features: [], labels: [],
  fcCfgs: [], selCfgs: [], modelCfgs: [], hpoCfgs: [], pipelines: [], archives: [],
  algos: [], opCatalog: [],
};

async function refreshAll() {
  const [ds, fs_, lb, fc, sel, md, hpo, pl, ar, algos, opc] = await Promise.all([
    API('/api/datasets'), API('/api/features'), API('/api/labels'),
    API('/api/feature_compute'), API('/api/feature_select'),
    API('/api/models'), API('/api/hpo'), API('/api/pipelines'),
    API('/api/archives'), API('/api/models/algorithms'),
    API('/api/feature_compute/op_catalog'),
  ]);
  STATE.datasets = ds; STATE.features = fs_; STATE.labels = lb;
  STATE.fcCfgs = fc; STATE.selCfgs = sel; STATE.modelCfgs = md;
  STATE.hpoCfgs = hpo; STATE.pipelines = pl; STATE.archives = ar;
  STATE.algos = algos; STATE.opCatalog = opc;
}

function onTabActivate(tab) {
  refreshAll().then(() => {
    if (tab === 'datasets') renderDatasets();
    if (tab === 'features') { renderFeatures(); renderFcCfgs(); }
    if (tab === 'labels') renderLabels();
    if (tab === 'select') renderSelCfgs();
    if (tab === 'viz') renderVizSelector();
    if (tab === 'models') renderModelCfgs();
    if (tab === 'hpo') renderHpoCfgs();
    if (tab === 'pipeline') renderPipelines();
    if (tab === 'archives') renderArchives();
  });
}

// ============================================================
// MODULE 1: DATASETS
// ============================================================
function renderDatasets() {
  const root = $('#datasets-list');
  root.innerHTML = '';
  if (STATE.datasets.length === 0) { root.textContent = '暂无数据集'; return; }
  STATE.datasets.forEach(d => {
    const summary = d.summary || {};
    const kinds = Object.entries(summary.value_kinds || {}).map(([k,v]) => `${k}:${v}`).join(', ');
    const vlen = summary.vector_length;
    const stats = summary.value_stats;
    const item = el('div', { class: 'item' },
      el('div', {},
        el('div', { class: 'it-name' }, `${d.name}  `,
          el('span', { class: 'tag ' + (d.source==='ssh'?'amber':'green') }, d.source || 'upload'),
          el('span', { class: 'tag gray' }, d.format || '?')),
        el('div', { class: 'it-meta' },
          `id=${d.id} · ${(d.size_bytes/1024).toFixed(1)} KB · ${summary.num_keys || 0} keys · ${kinds}`,
          vlen ? `· vector: ${vlen.min}~${vlen.max} (uniform=${vlen.uniform})` : '',
          stats ? ` · value: [${stats.min.toFixed(3)}, ${stats.max.toFixed(3)}] mean=${stats.mean.toFixed(3)} std=${stats.std.toFixed(3)}` : '',
        ),
        el('div', { class: 'summary-mini' },
          'sample: ', JSON.stringify(summary.sample_keys || []).slice(0, 80))
      ),
      el('div', { class: 'it-actions' },
        el('button', { class: 'ghost', onclick: () => viewStruct(d) }, '🔍 结构'),
        el('button', { class: 'success', onclick: () => promoteDs(d) }, '➕ 转为特征集'),
        el('button', { class: 'danger', onclick: () => delItem('datasets', d.id) }, '删除'),
      )
    );
    root.appendChild(item);
  });
}
function viewStruct(d) {
  const html = el('div', {},
    el('h3', {}, d.name + ' · 结构'),
    el('div', { class: 'kv' },
      ...Object.entries(d.summary || {}).flatMap(([k, v]) => [
        el('div', { class: 'k' }, k),
        el('div', { class: 'v' }, typeof v === 'object' ? JSON.stringify(v, null, 2) : String(v))
      ])
    )
  );
  $('#modal-content').innerHTML = ''; $('#modal-content').appendChild(html);
  $('#modal').classList.remove('hidden');
}
async function promoteDs(d) {
  const name = prompt('归档为特征集，请输入名称：', d.name + '_feat');
  if (!name) return;
  const fd = new FormData(); fd.append('name', name);
  await API(`/api/datasets/${d.id}/promote_to_feature`, { method: 'POST', body: fd });
  toast('已新建特征集', 'success'); refreshAll().then(renderDatasets);
}
async function delItem(kind, id) {
  if (!confirm('确定删除？')) return;
  await API(`/api/${kind}/${id}`, { method: 'DELETE' });
  toast('已删除', 'success'); refreshAll().then(() => onTabActivate(currentTab()));
}
function currentTab() {
  return $$('.sidebar .nav').find(b => b.classList.contains('active'))?.dataset.tab;
}

$('#ds-upload-form').addEventListener('submit', async e => {
  e.preventDefault();
  const fd = new FormData(e.target);
  try {
    await API('/api/datasets/upload', { method: 'POST', body: fd });
    toast('上传成功', 'success'); e.target.reset(); refreshAll().then(renderDatasets);
  } catch (err) { toast(err.message, 'error'); }
});
$('#ds-ssh-form').addEventListener('submit', async e => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const body = JSON.stringify(Object.fromEntries(fd));
  try {
    await API('/api/datasets/ssh_fetch', { method: 'POST',
      headers: {'Content-Type':'application/json'}, body });
    toast('SSH 拉取成功', 'success'); e.target.reset();
    refreshAll().then(renderDatasets);
  } catch (err) { toast(err.message, 'error'); }
});

// ============================================================
// MODULE 2: FEATURES list + Feature-compute graph editor
// ============================================================
function renderFeatures() {
  const root = $('#features-list');
  root.innerHTML = '';
  if (STATE.features.length === 0) { root.textContent = '暂无特征集'; return; }
  STATE.features.forEach(f => {
    const s = f.summary || {};
    const vlen = s.vector_length;
    const item = el('div', { class: 'item' },
      el('div', {},
        el('div', { class: 'it-name' }, f.name + '  ',
          el('span', { class: 'tag' }, f.source || 'manual')),
        el('div', { class: 'it-meta' },
          `id=${f.id} · ${s.num_keys || 0} keys · ${vlen ? 'dim ' + vlen.min : ''}`)
      ),
      el('div', { class: 'it-actions' },
        el('a', { href: `/api/features/${f.id}/download?fmt=json`, download: `${f.name}.json`,
                  class: 'ghost', style: { textDecoration:'none', padding:'6px 10px',
                  border:'1px solid #c7d2fe', borderRadius:'5px', color:'#4f46e5' } }, '⬇ JSON'),
        el('a', { href: `/api/features/${f.id}/download?fmt=pkl`, download: `${f.name}.pkl`,
                  class: 'ghost', style: { textDecoration:'none', padding:'6px 10px',
                  border:'1px solid #c7d2fe', borderRadius:'5px', color:'#4f46e5' } }, '⬇ PKL'),
        el('button', { class: 'danger', onclick: () => delItem('features', f.id) }, '删除'),
      )
    );
    root.appendChild(item);
  });
}

function renderFcCfgs() {
  const root = $('#fc-cfg-list');
  root.innerHTML = '';
  if (STATE.fcCfgs.length === 0) { root.textContent = '暂无特征计算配置'; return; }
  STATE.fcCfgs.forEach(c => {
    const item = el('div', { class: 'item' },
      el('div', {},
        el('div', { class: 'it-name' }, c.name),
        el('div', { class: 'it-meta' },
          `id=${c.id} · 节点数: ${(c.graph?.nodes||[]).length} · 边数: ${(c.graph?.edges||[]).length}`)
      ),
      el('div', { class: 'it-actions' },
        el('button', { class: 'ghost', onclick: () => openFcEditor(c) }, '编辑'),
        el('button', { class: 'success', onclick: () => runFcCfgQuick(c) }, '▶ 运行'),
        el('button', { class: 'danger', onclick: () => delItem('feature_compute', c.id) }, '删除'),
      )
    );
    root.appendChild(item);
  });
}

async function runFcCfgQuick(c) {
  const name = prompt('输出特征集前缀名：', c.name + '_out');
  if (!name) return;
  try {
    const r = await API(`/api/feature_compute/${c.id}/run`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ output_name: name, archive: true }),
    });
    toast(`已归档 ${r.archived_feature_sets.length} 个特征集`, 'success');
    refreshAll().then(() => { renderFeatures(); renderFcCfgs(); });
  } catch (e) { toast(e.message, 'error'); }
}

// ----- Feature-compute editor -----
let FC_STATE = { id: null, nodes: [], edges: [], selected: null, dragging: null, edgeDraft: null };

$('#fc-new').addEventListener('click', () => openFcEditor(null));
$('#fc-close').addEventListener('click', () => $('#fc-editor').classList.add('hidden'));
$('#fc-help').addEventListener('click', () => showFcHelp());

function showFcHelp() {
  $('#modal-content').innerHTML = `
    <h3>📘 画布拖拽使用说明</h3>
    <p><b>1. 拖入资源</b>：从左侧"资源"区域，把 数据集 或 已注册特征集 拖到画布作为输入节点。</p>
    <p><b>2. 拖入算子</b>：从"算子"区域，把 standardize / concat / add / mean / aggregate_xxx / filter_keys 等算子拖入画布。</p>
    <p><b>3. 连接节点</b>：每个节点有 <span style="color:#16a34a">▶ 输出口 (右)</span> 和 <span style="color:#4f46e5">▶ 输入口 (左)</span>，
       从一个节点的输出口按住鼠标拖到下一节点的输入口即可建立连边。多输入算子(如 concat、add)按连接顺序确定 port。</p>
    <p><b>4. 设置参数</b>：点击节点选中，在右侧<b>检查器</b>填参数 (例如 filter_keys 的目标参考资源 / scale 的 factor)。</p>
    <p><b>5. 输出节点</b>：拖入"输出节点"并连接最终算子到它，运行时该节点的张量将被归档为新特征集，可在"已注册特征集"中下载 JSON/PKL。</p>
    <p><b>6. 保存 & 运行</b>：先<b>保存配置</b>给一个名字，再点<b>运行 & 归档</b>。运行后右下方将看到每个输出节点的预览统计。</p>
    <p>典型示例 (人数特征 → 标准化 → 与原始拼接 → 按标签集Key过滤 → 输出)：</p>
    <pre>[ds:人数] -> [standardize]
[ds:人数] ----------┐
[standardize] ----┤ -> [concat] -> [filter_keys (allow_from_ref=label_xxx)] -> [output]</pre>
    <p>对应"商场分类"任务里的 <b>特征标准化 + 拼接 + 按标签集Key过滤</b>。</p>
  `;
  $('#modal').classList.remove('hidden');
}

function openFcEditor(cfg) {
  $('#fc-editor').classList.remove('hidden');
  $('#fc-editor-title').textContent = cfg ? `编辑配置: ${cfg.name}` : '新建特征计算配置';
  $('#fc-name').value = cfg?.name || '';
  $('#fc-desc').value = cfg?.description || '';
  FC_STATE.id = cfg?.id || null;
  FC_STATE.nodes = cfg?.graph?.nodes ? JSON.parse(JSON.stringify(cfg.graph.nodes)) : [];
  FC_STATE.edges = cfg?.graph?.edges ? JSON.parse(JSON.stringify(cfg.graph.edges)) : [];
  FC_STATE.selected = null; FC_STATE.dragging = null; FC_STATE.edgeDraft = null;
  // ensure all nodes have x/y
  FC_STATE.nodes.forEach((n, i) => {
    if (n.x == null) n.x = 60 + (i % 4) * 180;
    if (n.y == null) n.y = 40 + Math.floor(i / 4) * 90;
  });
  buildPalette();
  renderFcCanvas();
}

function buildPalette() {
  // resources
  const pr = $('#palette-resources'); pr.innerHTML = '';
  STATE.datasets.forEach(d => pr.appendChild(makePalItem(`📦 数据集: ${d.name}`,
    { kind: 'input', refKind: 'datasets', refId: d.id, label: d.name })));
  STATE.features.forEach(f => pr.appendChild(makePalItem(`📊 特征集: ${f.name}`,
    { kind: 'input', refKind: 'features', refId: f.id, label: f.name })));
  STATE.labels.forEach(l => pr.appendChild(makePalItem(`🏷 标签集: ${l.name}`,
    { kind: 'input', refKind: 'labels', refId: l.id, label: l.name })));
  // ops
  const po = $('#palette-ops'); po.innerHTML = '';
  STATE.opCatalog.forEach(o => po.appendChild(makePalItem(`🧮 ${o.label}`,
    { kind: 'op', opType: o.op_type, label: o.op_type })));
}

function makePalItem(text, data) {
  const item = el('div', { class: 'pal-item', draggable: 'true' }, text);
  item.addEventListener('dragstart', e => {
    e.dataTransfer.setData('text/plain', JSON.stringify(data));
  });
  return item;
}

const fcCanvas = $('#fc-canvas');
fcCanvas.addEventListener('dragover', e => e.preventDefault());
fcCanvas.addEventListener('drop', e => {
  e.preventDefault();
  let data;
  try { data = JSON.parse(e.dataTransfer.getData('text/plain')); }
  catch { data = null; }
  if (!data) {
    // could be the special output drag
    const k = e.dataTransfer.getData('text/plain');
    if (k === 'output') data = { kind: 'output' };
  }
  if (!data) return;
  const rect = fcCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left - 60;
  const y = e.clientY - rect.top - 18;
  const id = 'n_' + Math.random().toString(36).slice(2, 8);
  const node = { id, x, y };
  if (data.kind === 'input') {
    Object.assign(node, { type: 'input', ref_kind: data.refKind, ref_id: data.refId, label: data.label });
  } else if (data.kind === 'op') {
    Object.assign(node, { type: 'op', op_type: data.opType, params: {}, label: data.opType });
  } else if (data.kind === 'output') {
    Object.assign(node, { type: 'output', label: 'output' });
  } else { return; }
  FC_STATE.nodes.push(node);
  renderFcCanvas();
});
// also accept the special "output" pal item
document.addEventListener('dragstart', e => {
  if (e.target.classList?.contains('pal-item') && e.target.dataset.dragKind === 'output') {
    e.dataTransfer.setData('text/plain', JSON.stringify({ kind: 'output' }));
  }
});

function renderFcCanvas() {
  // clear nodes (keep svg)
  fcCanvas.querySelectorAll('.cnode').forEach(n => n.remove());
  const empty = fcCanvas.querySelector('.canvas-empty');
  if (empty) empty.style.display = FC_STATE.nodes.length === 0 ? '' : 'none';
  FC_STATE.nodes.forEach(n => fcCanvas.appendChild(buildNodeEl(n, 'fc')));
  drawEdges('#fc-edges', FC_STATE.nodes, FC_STATE.edges);
  renderFcInspector();
}

function buildNodeEl(n, scope) {
  const cls = scope === 'fc' ? n.type : ('pl-' + n.type);
  const node = el('div', { class: `cnode ${cls}`,
    style: { left: n.x + 'px', top: n.y + 'px' } });
  if (FC_STATE.selected === n.id || PL_STATE.selected === n.id) node.classList.add('selected');
  node.appendChild(el('div', { class: 'nh' }, nodeTitle(n)));
  if (n.label) node.appendChild(el('div', { class: 'nb' }, n.label));
  // ports
  if (n.type !== 'input') node.appendChild(el('div', { class: 'port in' }));
  if (n.type !== 'output') node.appendChild(el('div', { class: 'port out' }));
  // close
  node.appendChild(el('div', { class: 'nclose', onclick: ev => {
    ev.stopPropagation();
    deleteNode(scope, n.id);
  } }, '✕'));
  // selection
  node.addEventListener('mousedown', ev => {
    if (ev.target.classList.contains('port')) return;
    if (ev.target.classList.contains('nclose')) return;
    if (scope === 'fc') FC_STATE.selected = n.id;
    else PL_STATE.selected = n.id;
    startDrag(scope, n, ev);
  });
  // edge from out port
  const outP = node.querySelector('.port.out');
  if (outP) outP.addEventListener('mousedown', ev => {
    ev.stopPropagation();
    startEdgeDraft(scope, n.id, ev);
  });
  const inP = node.querySelector('.port.in');
  if (inP) inP.addEventListener('mouseup', ev => {
    ev.stopPropagation();
    finishEdgeDraft(scope, n.id);
  });
  return node;
}

function nodeTitle(n) {
  if (n.type === 'input') return '📥 ' + (n.ref_kind === 'datasets' ? '数据集'
                                         : n.ref_kind === 'features' ? '特征集'
                                         : '标签集');
  if (n.type === 'output') return '📤 输出';
  if (n.type === 'op') return '🧮 ' + n.op_type;
  // pipeline node types
  return ({
    feature: '📊 特征集',
    fc: '🧮 特征计算',
    label: '🏷 标签集',
    select: '🔬 特征选择',
    model: '🧠 模型',
    hpo: '🎯 寻优',
  })[n.type] || n.type;
}

function deleteNode(scope, nid) {
  const S = scope === 'fc' ? FC_STATE : PL_STATE;
  S.nodes = S.nodes.filter(n => n.id !== nid);
  S.edges = S.edges.filter(e => e.from !== nid && e.to !== nid);
  if (S.selected === nid) S.selected = null;
  if (scope === 'fc') renderFcCanvas(); else renderPlCanvas();
}

function startDrag(scope, n, ev) {
  const canvas = scope === 'fc' ? fcCanvas : plCanvas;
  const rect = canvas.getBoundingClientRect();
  const offX = ev.clientX - rect.left - n.x;
  const offY = ev.clientY - rect.top - n.y;
  const move = e => {
    n.x = Math.max(0, e.clientX - rect.left - offX);
    n.y = Math.max(0, e.clientY - rect.top - offY);
    if (scope === 'fc') renderFcCanvas(); else renderPlCanvas();
  };
  const up = () => {
    document.removeEventListener('mousemove', move);
    document.removeEventListener('mouseup', up);
  };
  document.addEventListener('mousemove', move);
  document.addEventListener('mouseup', up);
}

let EDGE_DRAFT = null;
function startEdgeDraft(scope, fromId, ev) {
  EDGE_DRAFT = { scope, from: fromId };
  const move = e => drawEdgeDraft(scope, e);
  const up = () => {
    document.removeEventListener('mousemove', move);
    document.removeEventListener('mouseup', up);
    setTimeout(() => { EDGE_DRAFT = null; if (scope === 'fc') renderFcCanvas(); else renderPlCanvas(); }, 50);
  };
  document.addEventListener('mousemove', move);
  document.addEventListener('mouseup', up);
}
function drawEdgeDraft(scope, ev) {
  const canvas = scope === 'fc' ? fcCanvas : plCanvas;
  const rect = canvas.getBoundingClientRect();
  const svgId = scope === 'fc' ? '#fc-edges' : '#pl-edges';
  const S = scope === 'fc' ? FC_STATE : PL_STATE;
  const fromNode = S.nodes.find(n => n.id === EDGE_DRAFT.from);
  if (!fromNode) return;
  const x1 = fromNode.x + 130, y1 = fromNode.y + 18;
  const x2 = ev.clientX - rect.left, y2 = ev.clientY - rect.top;
  drawEdges(svgId, S.nodes, S.edges, { x1, y1, x2, y2 });
}
function finishEdgeDraft(scope, toId) {
  if (!EDGE_DRAFT || EDGE_DRAFT.scope !== scope) return;
  if (EDGE_DRAFT.from === toId) return;
  const S = scope === 'fc' ? FC_STATE : PL_STATE;
  const port = S.edges.filter(e => e.to === toId).length;
  S.edges.push({ from: EDGE_DRAFT.from, to: toId, port });
  EDGE_DRAFT = null;
  if (scope === 'fc') renderFcCanvas(); else renderPlCanvas();
}

function drawEdges(svgSel, nodes, edges, draft=null) {
  const svg = $(svgSel);
  if (!svg) return;
  svg.innerHTML = '';
  edges.forEach(e => {
    const a = nodes.find(n => n.id === e.from);
    const b = nodes.find(n => n.id === e.to);
    if (!a || !b) return;
    const x1 = a.x + 130, y1 = a.y + 18;
    const x2 = b.x, y2 = b.y + 18;
    svg.appendChild(makeEdgePath(x1, y1, x2, y2));
  });
  if (draft) svg.appendChild(makeEdgePath(draft.x1, draft.y1, draft.x2, draft.y2, true));
}
function makeEdgePath(x1, y1, x2, y2, dashed=false) {
  const dx = Math.abs(x2 - x1) * 0.5;
  const d = `M${x1},${y1} C${x1+dx},${y1} ${x2-dx},${y2} ${x2},${y2}`;
  const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  p.setAttribute('d', d);
  if (dashed) p.setAttribute('stroke-dasharray', '4,3');
  return p;
}

function renderFcInspector() {
  const root = $('#fc-inspector');
  root.innerHTML = '<h4>🔍 节点检查器</h4>';
  if (!FC_STATE.selected) { root.appendChild(el('div', {class:'hint'}, '点击画布节点')); return; }
  const n = FC_STATE.nodes.find(x => x.id === FC_STATE.selected);
  if (!n) return;
  root.appendChild(el('div', { class: 'kv' },
    el('div', {class:'k'}, 'id'), el('div', {class:'v'}, n.id),
    el('div', {class:'k'}, 'type'), el('div', {class:'v'}, n.type),
    el('div', {class:'k'}, n.type === 'op' ? 'op' : 'ref'),
    el('div', {class:'v'}, n.op_type || (n.ref_kind + '/' + (n.ref_id||'')))
  ));
  if (n.type === 'op') {
    const opSpec = STATE.opCatalog.find(o => o.op_type === n.op_type);
    if (opSpec && opSpec.params.length) {
      n.params = n.params || {};
      opSpec.params.forEach(p => {
        const wrap = el('div', { class: 'insp-row' },
          el('label', {}, p.label || p.name));
        let inp;
        if (p.type === 'choice') {
          inp = el('select', { onchange: e => { n.params[p.name] = e.target.value; } });
          (p.choices || []).forEach(c => inp.appendChild(el('option', {value:c}, c)));
          inp.value = n.params[p.name] || (p.choices||[])[0] || '';
        } else if (p.type === 'float' || p.type === 'int') {
          inp = el('input', { type: 'number',
            value: n.params[p.name] ?? (p.default ?? ''),
            onchange: e => { n.params[p.name] = parseFloat(e.target.value); } });
        } else {
          inp = el('input', { type: 'text',
            value: n.params[p.name] ?? '',
            onchange: e => { n.params[p.name] = e.target.value; } });
        }
        wrap.appendChild(inp);
        root.appendChild(wrap);
      });
    }
  }
}

$('#fc-save').addEventListener('click', async () => {
  const body = {
    name: $('#fc-name').value || ('cfg_' + Date.now()),
    description: $('#fc-desc').value,
    graph: { nodes: FC_STATE.nodes, edges: FC_STATE.edges },
  };
  try {
    const url = FC_STATE.id ? `/api/feature_compute/${FC_STATE.id}` : '/api/feature_compute';
    const method = FC_STATE.id ? 'PUT' : 'POST';
    const r = await API(url, { method, headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body) });
    FC_STATE.id = r.id;
    toast('已保存', 'success');
    refreshAll().then(renderFcCfgs);
  } catch (e) { toast(e.message, 'error'); }
});

$('#fc-run').addEventListener('click', async () => {
  if (!FC_STATE.id) { toast('请先保存配置', 'error'); return; }
  const name = $('#fc-name').value || 'computed';
  try {
    const r = await API(`/api/feature_compute/${FC_STATE.id}/run`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ output_name: name, archive: true }),
    });
    const out = $('#fc-run-result'); out.innerHTML = '';
    out.appendChild(el('h4', {}, '运行结果'));
    Object.entries(r.previews).forEach(([nid, sum]) => {
      out.appendChild(el('div', { class: 'kv' },
        el('div', {class:'k'}, 'node'), el('div', {class:'v'}, nid),
        el('div', {class:'k'}, 'num_keys'), el('div', {class:'v'}, String(sum.num_keys)),
        el('div', {class:'k'}, 'vector_length'), el('div', {class:'v'}, JSON.stringify(sum.vector_length||{})),
        el('div', {class:'k'}, 'value_stats'), el('div', {class:'v'}, JSON.stringify(sum.value_stats||{})),
      ));
    });
    if (r.archived_feature_sets.length) {
      out.appendChild(el('div', {}, `归档为新特征集: ${r.archived_feature_sets.map(f=>f.name).join(', ')}`));
      toast(`归档 ${r.archived_feature_sets.length} 个`, 'success');
    }
    refreshAll().then(renderFeatures);
  } catch (e) { toast(e.message, 'error'); }
});

// ============================================================
// MODULE 3: LABELS
// ============================================================
function renderLabels() {
  const root = $('#labels-list');
  root.innerHTML = '';
  if (STATE.labels.length === 0) { root.textContent = '暂无标签集'; return; }
  STATE.labels.forEach(l => {
    const dist = l.summary?.distribution || {};
    const item = el('div', { class: 'item' },
      el('div', {},
        el('div', { class: 'it-name' }, l.name),
        el('div', { class: 'it-meta' },
          `id=${l.id} · ${l.summary?.num_keys || 0} samples · ${l.summary?.num_classes || 0} classes`),
        el('div', { class: 'summary-mini' },
          ...Object.entries(dist).map(([k, v]) =>
            el('span', { class: 'tag' }, `${k}: ${v}`)
          )
        )
      ),
      el('div', { class: 'it-actions' },
        el('button', { class: 'ghost', onclick: () => showDistChart(l) }, '📊 分布图'),
        el('button', { class: 'danger', onclick: () => delItem('labels', l.id) }, '删除'),
      )
    );
    root.appendChild(item);
  });
}

function showDistChart(l) {
  const dist = l.summary?.distribution || {};
  const labels = Object.keys(dist), values = Object.values(dist);
  const html = el('div', {},
    el('h3', {}, l.name + ' · 分布'),
    el('canvas', { id: 'dist-chart-' + l.id, height: '260' })
  );
  $('#modal-content').innerHTML = ''; $('#modal-content').appendChild(html);
  $('#modal').classList.remove('hidden');
  setTimeout(() => {
    new Chart($('#dist-chart-' + l.id), {
      type: 'bar',
      data: { labels, datasets: [{ label: 'count', data: values,
              backgroundColor: ['#4f46e5','#16a34a','#f59e0b','#ec4899','#06b6d4'] }] },
      options: { responsive: true, plugins: { legend: { display: false } } },
    });
  }, 50);
}

$('#lb-upload-form').addEventListener('submit', async e => {
  e.preventDefault();
  const fd = new FormData(e.target);
  try {
    await API('/api/labels/upload', { method: 'POST', body: fd });
    toast('已新增标签集', 'success'); e.target.reset();
    refreshAll().then(renderLabels);
  } catch (err) { toast(err.message, 'error'); }
});

// ============================================================
// MODULE 4: FEATURE SELECT
// ============================================================
function renderSelCfgs() {
  const root = $('#sel-list');
  root.innerHTML = '';
  if (STATE.selCfgs.length === 0) { root.textContent = '暂无配置'; return; }
  STATE.selCfgs.forEach(c => {
    const item = el('div', { class: 'item' },
      el('div', {},
        el('div', { class: 'it-name' }, c.name),
        el('div', { class: 'it-meta' },
          `id=${c.id} · 过滤器: ${(c.filters||[]).length} · 重要性: ${c.importance?.strategy || 'uniform'}`),
        el('div', { class: 'summary-mini' }, c.description || '')
      ),
      el('div', { class: 'it-actions' },
        el('button', { class: 'ghost', onclick: () => openSelEditor(c) }, '编辑'),
        el('button', { class: 'danger', onclick: () => delItem('feature_select', c.id) }, '删除'),
      )
    );
    root.appendChild(item);
  });
}

let SEL_STATE = { id: null, filters: [] };
$('#sel-new').addEventListener('click', () => openSelEditor(null));
$('#sel-close').addEventListener('click', () => $('#sel-editor').classList.add('hidden'));

function openSelEditor(c) {
  $('#sel-editor').classList.remove('hidden');
  $('#sel-editor-title').textContent = c ? `编辑配置: ${c.name}` : '新建特征选择配置';
  $('#sel-name').value = c?.name || '';
  $('#sel-desc').value = c?.description || '';
  $('#sel-imp-strategy').value = c?.importance?.strategy || 'uniform';
  $('#sel-imp-weights').value = (c?.importance?.weights || []).join(',');
  SEL_STATE.id = c?.id || null;
  SEL_STATE.filters = c?.filters ? JSON.parse(JSON.stringify(c.filters)) : [];
  renderSelFilters();
  $('#sel-preview-out').innerHTML = '';
}

function renderSelFilters() {
  const root = $('#sel-filters'); root.innerHTML = '';
  SEL_STATE.filters.forEach((f, i) => {
    const row = el('div', { class: 'row' });
    const sel = el('select', { onchange: e => { f.type = e.target.value;
      delete f.column; delete f.min; delete f.max; delete f.min_var; delete f.k;
      renderSelFilters();
    } });
    ['value_range','variance','missing_drop','topk_variance'].forEach(t =>
      sel.appendChild(el('option', { value: t, selected: f.type === t ? '' : null }, t)));
    row.appendChild(sel);
    if (f.type === 'value_range') {
      row.appendChild(el('input', { placeholder: 'column (索引或 "all")',
        value: f.column ?? 'all', onchange: e => { f.column = e.target.value === 'all' ? 'all' : parseInt(e.target.value); }}));
      row.appendChild(el('input', { placeholder: 'min', type: 'number', value: f.min ?? '',
        onchange: e => { f.min = e.target.value === '' ? null : parseFloat(e.target.value); }}));
      row.appendChild(el('input', { placeholder: 'max', type: 'number', value: f.max ?? '',
        onchange: e => { f.max = e.target.value === '' ? null : parseFloat(e.target.value); }}));
    } else if (f.type === 'variance') {
      row.appendChild(el('input', { placeholder: 'min_var', type: 'number', step: '0.001',
        value: f.min_var ?? 0.0,
        onchange: e => { f.min_var = parseFloat(e.target.value); }}));
    } else if (f.type === 'topk_variance') {
      row.appendChild(el('input', { placeholder: 'k', type: 'number', value: f.k ?? 10,
        onchange: e => { f.k = parseInt(e.target.value); }}));
    }
    row.appendChild(el('button', { class: 'danger',
      onclick: () => { SEL_STATE.filters.splice(i,1); renderSelFilters(); } }, '✕'));
    root.appendChild(row);
  });
}

$('#sel-add-filter').addEventListener('click', () => {
  SEL_STATE.filters.push({ type: 'variance', min_var: 0.0 });
  renderSelFilters();
});

function buildSelCfgBody() {
  const wstr = ($('#sel-imp-weights').value || '').trim();
  const weights = wstr ? wstr.split(',').map(x => parseFloat(x.trim())).filter(x => !isNaN(x)) : [];
  return {
    name: $('#sel-name').value || ('sel_' + Date.now()),
    description: $('#sel-desc').value,
    filters: SEL_STATE.filters,
    importance: { strategy: $('#sel-imp-strategy').value, weights },
  };
}

$('#sel-save').addEventListener('click', async () => {
  try {
    const body = buildSelCfgBody();
    const url = SEL_STATE.id ? `/api/feature_select/${SEL_STATE.id}` : '/api/feature_select';
    const method = SEL_STATE.id ? 'PUT' : 'POST';
    const r = await API(url, { method, headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    SEL_STATE.id = r.id;
    toast('已保存', 'success'); refreshAll().then(renderSelCfgs);
  } catch (e) { toast(e.message, 'error'); }
});

$('#sel-preview').addEventListener('click', async () => {
  const fid = prompt('在哪个特征集上预览？输入特征集 id (在 ②已注册特征集 中查看):',
    STATE.features[0]?.id || '');
  if (!fid) return;
  try {
    const r = await API('/api/feature_select/preview', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ feature_set_id: fid, cfg_inline: buildSelCfgBody() })
    });
    const out = $('#sel-preview-out'); out.innerHTML = '';
    out.appendChild(el('div', { class: 'code-block' },
      JSON.stringify(r, null, 2)));
  } catch (e) { toast(e.message, 'error'); }
});

// ============================================================
// MODULE 5: FEATURE VIZ
// ============================================================
function renderVizSelector() {
  const sel = $('#viz-fs'); sel.innerHTML = '';
  STATE.features.forEach(f => sel.appendChild(el('option', { value: f.id }, f.name + ' (' + f.id + ')')));
}
let vizCharts = [null, null];
$('#viz-run').addEventListener('click', async () => {
  const fid = $('#viz-fs').value;
  const mode = $('#viz-mode').value;
  const key = $('#viz-key').value;
  if (!fid) return toast('请先选择特征集', 'error');
  try {
    const r = await API(`/api/features/${fid}/visualize?mode=${mode}&sample_key=${encodeURIComponent(key)}&max_keys=200`);
    vizCharts.forEach(c => c && c.destroy());
    vizCharts = [null, null];
    const c1 = $('#viz-canvas-1'), c2 = $('#viz-canvas-2');
    const stats = $('#viz-stats'); stats.innerHTML = '';
    if (r.empty) { stats.textContent = '空数据集'; return; }
    if (r.mode === 'distribution') {
      vizCharts[0] = new Chart(c1, {
        type: 'bar',
        data: {
          labels: r.hist.edges.slice(0, -1).map(x => x.toFixed(2)),
          datasets: [{ label: '直方图 (全量值分布)', data: r.hist.counts, backgroundColor: '#4f46e5' }]
        }
      });
      const cols = r.per_col_stats;
      vizCharts[1] = new Chart(c2, {
        type: 'line',
        data: {
          labels: cols.map(c => c.col),
          datasets: [
            { label: 'mean', data: cols.map(c => c.mean), borderColor: '#16a34a', fill: false },
            { label: 'std',  data: cols.map(c => c.std),  borderColor: '#f59e0b', fill: false },
          ]
        }
      });
      stats.innerHTML = `<div class="kv">
        <div class="k">shape</div><div class="v">${r.n_rows} × ${r.n_cols}</div>
        <div class="k">global</div><div class="v">${JSON.stringify(r.global)}</div>
      </div>`;
    } else if (r.mode === 'instance') {
      vizCharts[0] = new Chart(c1, {
        type: 'line',
        data: {
          labels: r.vector.map((_, i) => i),
          datasets: [{ label: `key=${r.key} (vector)`, data: r.vector, borderColor: '#4f46e5', fill: false }]
        }
      });
      stats.textContent = `key=${r.key}, dim=${r.vector.length}`;
    } else if (r.mode === 'tsne') {
      if (r.error) { stats.textContent = '降维失败: ' + r.error; return; }
      vizCharts[0] = new Chart(c1, {
        type: 'scatter',
        data: { datasets: [{ label: 'PCA-2D', data: r.points.map(p => ({x: p.x, y: p.y})),
                              backgroundColor: '#4f46e5' }] },
        options: { plugins: { tooltip: { callbacks: {
          label: ctx => r.points[ctx.dataIndex].key + ': ' +
            ctx.parsed.x.toFixed(2) + ', ' + ctx.parsed.y.toFixed(2)
        } } } }
      });
      stats.textContent = `${r.points.length} pts`;
    }
  } catch (e) { toast(e.message, 'error'); }
});

// ============================================================
// MODULE 6: MODELS
// ============================================================
function renderModelCfgs() {
  const root = $('#md-list'); root.innerHTML = '';
  if (STATE.modelCfgs.length === 0) { root.textContent = '暂无模型配置'; return; }
  STATE.modelCfgs.forEach(m => {
    const item = el('div', { class: 'item' },
      el('div', {},
        el('div', { class: 'it-name' }, m.name + ' ',
          el('span', { class: 'tag' }, m.algo),
          el('span', { class: 'tag gray' }, m.task || '')),
        el('div', { class: 'it-meta' },
          `id=${m.id} · params: ${JSON.stringify(m.params || {}).slice(0,80)}`),
      ),
      el('div', { class: 'it-actions' },
        el('button', { class: 'ghost', onclick: () => openMdEditor(m) }, '编辑'),
        el('button', { class: 'danger', onclick: () => delItem('models', m.id) }, '删除'),
      )
    );
    root.appendChild(item);
  });
  // populate algo selector
  const sel = $('#md-algo'); sel.innerHTML = '';
  STATE.algos.forEach(a => sel.appendChild(el('option', { value: a.name }, `${a.name} (${a.task})`)));
}

let MD_STATE = { id: null };
$('#md-new').addEventListener('click', () => openMdEditor(null));
$('#md-close').addEventListener('click', () => $('#md-editor').classList.add('hidden'));
$('#md-algo').addEventListener('change', () => buildMdParams(null, null));

function openMdEditor(m) {
  $('#md-editor').classList.remove('hidden');
  $('#md-editor-title').textContent = m ? '编辑模型配置: ' + m.name : '新建模型配置';
  $('#md-name').value = m?.name || '';
  $('#md-desc').value = m?.description || '';
  $('#md-algo').value = m?.algo || STATE.algos[0]?.name || '';
  MD_STATE.id = m?.id || null;
  buildMdParams(m?.params || {}, m?.param_ranges || {});
}

function buildMdParams(curParams, curRanges) {
  const algo = $('#md-algo').value;
  const spec = STATE.algos.find(a => a.name === algo);
  if (!spec) return;
  curParams = curParams || {}; curRanges = curRanges || {};
  const pRoot = $('#md-params'); pRoot.innerHTML = '';
  const rRoot = $('#md-ranges'); rRoot.innerHTML = '';
  spec.param_schema.forEach(p => {
    // default value input
    const row = el('div', { class: 'row' },
      el('label', { style:{minWidth:'140px'} }, p.name + ` (${p.type})`));
    let inp;
    if (p.type === 'choice') {
      inp = el('select', { onchange: e => { curParams[p.name] = e.target.value; } });
      (p.choices||[]).forEach(c => inp.appendChild(el('option', { value: c }, c)));
      inp.value = curParams[p.name] ?? p.default ?? (p.choices||[])[0];
    } else {
      inp = el('input', { type: 'number', step: 'any',
        value: curParams[p.name] ?? p.default ?? '',
        onchange: e => { curParams[p.name] = p.type === 'int'
          ? parseInt(e.target.value) : parseFloat(e.target.value); }});
    }
    row.appendChild(inp);
    pRoot.appendChild(row);

    // range inputs (low/high or values + log)
    const rrow = el('div', { class: 'row' },
      el('label', { style:{minWidth:'140px'} }, p.name));
    if (p.type === 'choice') {
      const inp2 = el('input', { placeholder: 'values 逗号分隔',
        value: (curRanges[p.name]?.values || []).join(','),
        onchange: e => {
          curRanges[p.name] = { type: 'choice',
            values: e.target.value.split(',').map(s => s.trim()).filter(Boolean) };
        }});
      rrow.appendChild(inp2);
    } else {
      const lo = el('input', { placeholder: 'low', type: 'number', step: 'any',
        value: curRanges[p.name]?.low ?? p.range?.[0] ?? '' });
      const hi = el('input', { placeholder: 'high', type: 'number', step: 'any',
        value: curRanges[p.name]?.high ?? p.range?.[1] ?? '' });
      const nn = el('input', { placeholder: 'grid点数 n', type: 'number',
        value: curRanges[p.name]?.n ?? 4 });
      const log = el('label', {},
        el('input', { type: 'checkbox', checked: curRanges[p.name]?.log ? '' : null }),
        ' log');
      const sync = () => {
        curRanges[p.name] = { type: p.type,
          low: parseFloat(lo.value), high: parseFloat(hi.value),
          n: parseInt(nn.value),
          log: log.querySelector('input').checked };
      };
      [lo, hi, nn].forEach(i => i.addEventListener('change', sync));
      log.querySelector('input').addEventListener('change', sync);
      rrow.appendChild(lo); rrow.appendChild(hi); rrow.appendChild(nn); rrow.appendChild(log);
    }
    rRoot.appendChild(rrow);
  });
  pRoot.dataset.cur = JSON.stringify(curParams);
  rRoot.dataset.cur = JSON.stringify(curRanges);
  // store the live references
  MD_STATE.curParams = curParams;
  MD_STATE.curRanges = curRanges;
}

$('#md-save').addEventListener('click', async () => {
  const body = {
    name: $('#md-name').value || ('md_' + Date.now()),
    algo: $('#md-algo').value,
    description: $('#md-desc').value,
    params: MD_STATE.curParams || {},
    param_ranges: MD_STATE.curRanges || {},
  };
  try {
    const url = MD_STATE.id ? `/api/models/${MD_STATE.id}` : '/api/models';
    const method = MD_STATE.id ? 'PUT' : 'POST';
    await API(url, { method, headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toast('已保存', 'success'); $('#md-editor').classList.add('hidden');
    refreshAll().then(renderModelCfgs);
  } catch (e) { toast(e.message, 'error'); }
});

// ============================================================
// MODULE 7: HPO
// ============================================================
function renderHpoCfgs() {
  const root = $('#hpo-list'); root.innerHTML = '';
  if (STATE.hpoCfgs.length === 0) { root.textContent = '暂无寻优配置'; return; }
  STATE.hpoCfgs.forEach(h => {
    const item = el('div', { class: 'item' },
      el('div', {},
        el('div', { class: 'it-name' }, h.name + ' ',
          el('span', { class: 'tag' }, h.strategy)),
        el('div', { class: 'it-meta' },
          `id=${h.id} · max_trials=${h.max_trials} · 参数:${Object.keys(h.param_space||{}).join(',')}`)
      ),
      el('div', { class: 'it-actions' },
        el('button', { class: 'ghost', onclick: () => openHpoEditor(h) }, '编辑'),
        el('button', { class: 'danger', onclick: () => delItem('hpo', h.id) }, '删除'),
      )
    );
    root.appendChild(item);
  });
}

let HPO_STATE = { id: null };
$('#hpo-new').addEventListener('click', () => openHpoEditor(null));
$('#hpo-close').addEventListener('click', () => $('#hpo-editor').classList.add('hidden'));

function openHpoEditor(h) {
  $('#hpo-editor').classList.remove('hidden');
  $('#hpo-name').value = h?.name || '';
  $('#hpo-strategy').value = h?.strategy || 'grid';
  $('#hpo-trials').value = h?.max_trials || 10;
  $('#hpo-priorities').value = (h?.priorities || []).join(',');
  $('#hpo-space').value = JSON.stringify(h?.param_space || {}, null, 2);
  $('#hpo-bayes').value = JSON.stringify(h?.bayes || {}, null, 2);
  HPO_STATE.id = h?.id || null;
}

$('#hpo-save').addEventListener('click', async () => {
  let space, bayes;
  try { space = JSON.parse($('#hpo-space').value || '{}'); }
  catch { return toast('param_space JSON 解析失败', 'error'); }
  try { bayes = JSON.parse($('#hpo-bayes').value || '{}'); }
  catch { return toast('bayes JSON 解析失败', 'error'); }
  const body = {
    name: $('#hpo-name').value || ('hpo_' + Date.now()),
    strategy: $('#hpo-strategy').value,
    max_trials: parseInt($('#hpo-trials').value) || 10,
    priorities: ($('#hpo-priorities').value || '').split(',').map(s=>s.trim()).filter(Boolean),
    param_space: space, bayes,
  };
  try {
    const url = HPO_STATE.id ? `/api/hpo/${HPO_STATE.id}` : '/api/hpo';
    const method = HPO_STATE.id ? 'PUT' : 'POST';
    await API(url, { method, headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toast('已保存', 'success'); $('#hpo-editor').classList.add('hidden');
    refreshAll().then(renderHpoCfgs);
  } catch (e) { toast(e.message, 'error'); }
});

// ============================================================
// MODULE 8: PIPELINES
// ============================================================
const plCanvas = $('#pl-canvas');
let PL_STATE = { id: null, nodes: [], edges: [], selected: null };

function renderPipelines() {
  const root = $('#pl-list'); root.innerHTML = '';
  if (STATE.pipelines.length === 0) { root.textContent = '暂无任务'; return; }
  STATE.pipelines.forEach(p => {
    const item = el('div', { class: 'item' },
      el('div', {},
        el('div', { class: 'it-name' }, p.name),
        el('div', { class: 'it-meta' },
          `id=${p.id} · 特征集×${(p.feature_set_ids||[]).length} · 模型×${(p.model_node_ids||[]).length} · 设备=${p.device}`)
      ),
      el('div', { class: 'it-actions' },
        el('button', { class: 'ghost', onclick: () => openPlEditor(p) }, '编辑'),
        el('button', { class: 'success', onclick: () => trainPipeline(p.id) }, '▶ 训练'),
        el('button', { class: 'danger', onclick: () => delItem('pipelines', p.id) }, '删除'),
      )
    );
    root.appendChild(item);
  });
}

$('#pl-new').addEventListener('click', () => openPlEditor(null));
$('#pl-close').addEventListener('click', () => $('#pl-editor').classList.add('hidden'));

function openPlEditor(p) {
  $('#pl-editor').classList.remove('hidden');
  $('#pl-editor-title').textContent = p ? '编辑训练任务: ' + p.name : '新建训练任务';
  $('#pl-name').value = p?.name || '';
  $('#pl-desc').value = p?.description || '';
  $('#pl-device').value = p?.device || 'cpu';
  $('#pl-maxiter').value = p?.max_iterations || '';
  $('#pl-valsplit').value = p?.validation_split ?? 0.2;
  $('#pl-assembly').value = p?.assembly || 'concat';
  $('#pl-asm-weights').value = (p?.assembly_weights || []).join(',');
  PL_STATE.id = p?.id || null;
  PL_STATE.nodes = p?.canvas?.nodes ? JSON.parse(JSON.stringify(p.canvas.nodes)) : [];
  PL_STATE.edges = p?.canvas?.edges ? JSON.parse(JSON.stringify(p.canvas.edges)) : [];
  PL_STATE.selected = null;
  buildPlPalette();
  renderPlCanvas();
  $('#pl-train-out').innerHTML = '';
}

function buildPlPalette() {
  const pr = $('#pl-palette'); pr.innerHTML = '';
  const types = [
    { type: 'feature', label: '📊 特征集 (可多选)' },
    { type: 'fc',      label: '🧮 特征计算配置' },
    { type: 'label',   label: '🏷 标签集' },
    { type: 'select',  label: '🔬 特征选择配置' },
    { type: 'model',   label: '🧠 模型 (可多个: AE→MLP)' },
    { type: 'hpo',     label: '🎯 寻优配置' },
  ];
  types.forEach(t => {
    const item = el('div', { class: 'pal-item', draggable: 'true' }, t.label);
    item.addEventListener('dragstart', e => {
      e.dataTransfer.setData('text/plain', JSON.stringify({ kind: 'plnode', type: t.type }));
    });
    pr.appendChild(item);
  });
}

plCanvas.addEventListener('dragover', e => e.preventDefault());
plCanvas.addEventListener('drop', e => {
  e.preventDefault();
  let data;
  try { data = JSON.parse(e.dataTransfer.getData('text/plain')); } catch { return; }
  if (!data || data.kind !== 'plnode') return;
  const rect = plCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left - 60;
  const y = e.clientY - rect.top - 18;
  const id = 'pn_' + Math.random().toString(36).slice(2, 8);
  PL_STATE.nodes.push({ id, type: data.type, x, y, ref_id: null, ref_ids: [], label: data.type });
  renderPlCanvas();
});

function renderPlCanvas() {
  plCanvas.querySelectorAll('.cnode').forEach(n => n.remove());
  const empty = plCanvas.querySelector('.canvas-empty');
  if (empty) empty.style.display = PL_STATE.nodes.length === 0 ? '' : 'none';
  PL_STATE.nodes.forEach(n => plCanvas.appendChild(buildNodeEl(n, 'pl')));
  drawEdges('#pl-edges', PL_STATE.nodes, PL_STATE.edges);
  renderPlInspector();
}

function renderPlInspector() {
  const root = $('#pl-inspector');
  root.innerHTML = '<h4>🔍 节点检查器</h4>';
  if (!PL_STATE.selected) {
    root.appendChild(el('div', { class: 'hint' }, '点击节点选择具体配置'));
    return;
  }
  const n = PL_STATE.nodes.find(x => x.id === PL_STATE.selected);
  if (!n) return;
  root.appendChild(el('div', { class: 'kv' },
    el('div', {class:'k'}, 'type'), el('div', {class:'v'}, n.type)));

  const opts = ({
    feature: STATE.features,
    fc:      STATE.fcCfgs,
    label:   STATE.labels,
    select:  STATE.selCfgs,
    model:   STATE.modelCfgs,
    hpo:     STATE.hpoCfgs,
  })[n.type] || [];

  if (n.type === 'feature') {
    // multi-select
    const wrap = el('div', { class: 'insp-row' },
      el('label', {}, '选择特征集 (Ctrl/Cmd 多选)'));
    const selEl = el('select', { multiple: 'multiple', size: Math.min(opts.length+1, 6),
      onchange: e => {
        n.ref_ids = Array.from(e.target.selectedOptions).map(o => o.value);
      } });
    opts.forEach(o => selEl.appendChild(el('option', {
      value: o.id, selected: (n.ref_ids||[]).includes(o.id) ? '' : null }, o.name)));
    wrap.appendChild(selEl); root.appendChild(wrap);
  } else if (n.type === 'model') {
    // ordered list
    const wrap = el('div', { class: 'insp-row' },
      el('label', {}, '选择模型 (按顺序选择, AE先, MLP后)'));
    const selEl = el('select', { multiple: 'multiple', size: Math.min(opts.length+1, 6),
      onchange: e => {
        n.ref_ids = Array.from(e.target.selectedOptions).map(o => o.value);
      } });
    opts.forEach(o => selEl.appendChild(el('option', {
      value: o.id, selected: (n.ref_ids||[]).includes(o.id) ? '' : null },
      `${o.name} [${o.algo}]`)));
    wrap.appendChild(selEl); root.appendChild(wrap);
    root.appendChild(el('div', { class: 'hint' },
      '提示: 在多选 select 中按住 Ctrl/Cmd 点击。已选项的顺序由你点击顺序决定。'));
  } else {
    const wrap = el('div', { class: 'insp-row' },
      el('label', {}, '选择具体配置'));
    const selEl = el('select', { onchange: e => { n.ref_id = e.target.value || null; } });
    selEl.appendChild(el('option', { value: '' }, '— 请选择 —'));
    opts.forEach(o => selEl.appendChild(el('option', { value: o.id,
      selected: n.ref_id === o.id ? '' : null }, o.name)));
    wrap.appendChild(selEl); root.appendChild(wrap);
  }
}

function pipelineFromCanvas() {
  // collect nodes by type
  const nodes = PL_STATE.nodes;
  const featNode  = nodes.find(n => n.type === 'feature');
  const fcNode    = nodes.find(n => n.type === 'fc');
  const labelNode = nodes.find(n => n.type === 'label');
  const selNode   = nodes.find(n => n.type === 'select');
  const modelNode = nodes.find(n => n.type === 'model');
  const hpoNode   = nodes.find(n => n.type === 'hpo');
  return {
    name: $('#pl-name').value || ('pipeline_' + Date.now()),
    description: $('#pl-desc').value,
    feature_set_ids: featNode?.ref_ids || [],
    feature_compute_id: fcNode?.ref_id || null,
    feature_select_id: selNode?.ref_id || null,
    label_set_id: labelNode?.ref_id || null,
    model_node_ids: modelNode?.ref_ids || [],
    hpo_id: hpoNode?.ref_id || null,
    device: $('#pl-device').value,
    max_iterations: parseInt($('#pl-maxiter').value) || null,
    validation_split: parseFloat($('#pl-valsplit').value) || 0.2,
    assembly: $('#pl-assembly').value,
    assembly_weights: ($('#pl-asm-weights').value||'').split(',').map(s=>parseFloat(s.trim())).filter(x=>!isNaN(x)),
    canvas: { nodes: PL_STATE.nodes, edges: PL_STATE.edges },
  };
}

$('#pl-save').addEventListener('click', async () => {
  try {
    const body = pipelineFromCanvas();
    const url = PL_STATE.id ? `/api/pipelines/${PL_STATE.id}` : '/api/pipelines';
    const method = PL_STATE.id ? 'PUT' : 'POST';
    const r = await API(url, { method, headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    PL_STATE.id = r.id;
    toast('已保存', 'success'); refreshAll().then(renderPipelines);
  } catch (e) { toast(e.message, 'error'); }
});

$('#pl-train').addEventListener('click', async () => {
  if (!PL_STATE.id) {
    // auto-save
    document.getElementById('pl-save').click();
    setTimeout(() => $('#pl-train').click(), 400);
    return;
  }
  trainPipeline(PL_STATE.id);
});

async function trainPipeline(pid) {
  try {
    const r = await API(`/api/pipelines/${pid}/run`, { method: 'POST' });
    const out = $('#pl-train-out'); out.innerHTML = '';
    out.appendChild(el('h4', {}, `训练任务已启动: job=${r.job_id}`));
    const logBox = el('div', { class: 'code-block' }, '正在训练...');
    out.appendChild(logBox);
    const iterBox = el('div', { class: 'code-block' });
    out.appendChild(el('h4', {}, '迭代记录 (含原因)')); out.appendChild(iterBox);
    const poll = setInterval(async () => {
      try {
        const j = await API('/api/jobs/' + r.job_id);
        logBox.textContent = j.logs.slice(-200).join('\n');
        iterBox.textContent = j.iter_records.slice(-50).map(rec =>
          `[${rec.phase || 'iter'}#${rec.trial ?? ''}] score=${rec.score?.toFixed?.(4) ?? '?'} reason: ${rec.reason} params=${JSON.stringify(rec.params || {})}`
        ).join('\n');
        if (j.status === 'completed') {
          clearInterval(poll);
          out.appendChild(el('div', { class: 'tag green' }, '✓ 完成'));
          out.appendChild(el('button', { class: 'primary',
            onclick: () => { $$('.sidebar .nav').find(b => b.dataset.tab==='archives').click();
                             setTimeout(() => openArchive(j.archive_id), 200); }},
            '查看归档详情'));
          toast('训练完成 ✓', 'success');
        } else if (j.status === 'failed') {
          clearInterval(poll);
          out.appendChild(el('div', { class: 'tag amber' }, '✗ 失败: ' + j.error));
          if (j.traceback) out.appendChild(el('pre', {}, j.traceback));
          toast('训练失败', 'error');
        }
      } catch (e) { console.error(e); }
    }, 1000);
  } catch (e) { toast(e.message, 'error'); }
}

// ============================================================
// MODULE 9: ARCHIVES
// ============================================================
function renderArchives() {
  const root = $('#ar-list'); root.innerHTML = '';
  if (STATE.archives.length === 0) { root.textContent = '暂无归档'; return; }
  STATE.archives.forEach(a => {
    const ev = a.evaluation || {};
    const item = el('div', { class: 'item' },
      el('div', {},
        el('div', { class: 'it-name' }, (a.name || a.archive_id) + ' ',
          el('span', { class: 'tag' }, a.task || '?')),
        el('div', { class: 'it-meta' },
          `archive=${a.archive_id} · n=${a.n_samples}` +
          (ev.f1_macro ? ` · F1_macro=${ev.f1_macro.toFixed(4)} · acc=${ev.accuracy?.toFixed(4)}` : '') +
          (a.best_loss != null ? ` · best_loss=${a.best_loss.toFixed(4)}` : '')),
        el('div', { class: 'summary-mini' },
          `best_params: ${JSON.stringify(a.best_params || {})}`)
      ),
      el('div', { class: 'it-actions' },
        el('button', { class: 'ghost', onclick: () => openArchive(a.archive_id) }, '详情'),
        el('button', { class: 'danger', onclick: () => delItem('archives', a.archive_id) }, '删除'),
      )
    );
    root.appendChild(item);
  });
}

async function openArchive(aid) {
  try {
    const a = await API('/api/archives/' + aid);
    $('#ar-detail').classList.remove('hidden');
    $('#ar-title').textContent = (a.name || aid) + ' · 详情';
    const body = $('#ar-body'); body.innerHTML = '';
    // Top-level info
    body.appendChild(el('div', { class: 'kv' },
      el('div', {class:'k'}, 'task'), el('div', {class:'v'}, a.task || ''),
      el('div', {class:'k'}, 'device'), el('div', {class:'v'}, a.device || ''),
      el('div', {class:'k'}, 'n_samples'), el('div', {class:'v'}, String(a.n_samples)),
      el('div', {class:'k'}, 'X shape'), el('div', {class:'v'}, JSON.stringify(a.input_X_shape) + ' → ' + JSON.stringify(a.encoded_X_shape)),
      el('div', {class:'k'}, 'best_params'), el('div', {class:'v'}, JSON.stringify(a.best_params || {})),
      el('div', {class:'k'}, 'feature_set_ids'), el('div', {class:'v'}, (a.feature_set_ids||[]).join(', ')),
      el('div', {class:'k'}, 'label_set_id'), el('div', {class:'v'}, a.label_set_id || ''),
      el('div', {class:'k'}, 'model_node_ids'), el('div', {class:'v'}, (a.model_node_ids||[]).join(', ')),
      el('div', {class:'k'}, 'hpo_id'), el('div', {class:'v'}, a.hpo_id || ''),
    ));

    // Downloads
    const dl = el('div', { class: 'row' },
      el('a', { href: `/api/archives/${aid}/download_iters`, class: 'ghost',
        style: { padding:'6px 12px', border:'1px solid #c7d2fe', borderRadius:'5px', textDecoration:'none', color:'#4f46e5'},
        download: '' }, '⬇ 迭代记录 JSON')
    );
    if (a.weights_path) dl.appendChild(el('a', { href: `/api/archives/${aid}/download_weights`,
      class: 'ghost', style:{padding:'6px 12px', border:'1px solid #c7d2fe', borderRadius:'5px', textDecoration:'none', color:'#4f46e5'},
      download: '' }, '⬇ 模型权重 .pt'));
    if (a.breakdown_json_path) dl.appendChild(el('a', { href: `/api/archives/${aid}/download_breakdown`,
      class: 'ghost', style:{padding:'6px 12px', border:'1px solid #c7d2fe', borderRadius:'5px', textDecoration:'none', color:'#4f46e5'},
      download: '' }, '⬇ TP/FP/FN/TN Key JSON'));
    body.appendChild(dl);

    // Classification details
    if (a.task === 'classification' && a.evaluation) {
      const ev = a.evaluation;
      body.appendChild(el('h4', {}, '混淆矩阵'));
      const cm = ev.confusion_matrix; const cls = ev.classes;
      const tbl = el('table', { class: 'simple' });
      const head = el('tr', {}, el('th', {}, 'true \\ pred'), ...cls.map(c => el('th', {}, String(c))));
      tbl.appendChild(head);
      cm.forEach((row, i) => {
        const tr = el('tr', {}, el('th', {}, String(cls[i])));
        row.forEach((v, j) => tr.appendChild(el('td', { class: 'cm-cell ' + (i===j?'tp':'fn') }, String(v))));
        tbl.appendChild(tr);
      });
      body.appendChild(tbl);
      body.appendChild(el('div', { class: 'kv' },
        el('div', {class:'k'}, 'accuracy'), el('div', {class:'v'}, ev.accuracy.toFixed(4)),
        el('div', {class:'k'}, 'F1 macro'), el('div', {class:'v'}, ev.f1_macro.toFixed(4)),
        el('div', {class:'k'}, 'F1 weighted'), el('div', {class:'v'}, ev.f1_weighted.toFixed(4)),
        el('div', {class:'k'}, 'precision macro'), el('div', {class:'v'}, ev.precision_macro.toFixed(4)),
        el('div', {class:'k'}, 'recall macro'), el('div', {class:'v'}, ev.recall_macro.toFixed(4)),
      ));
      // sample breakdown
      if (a.sample_breakdown) {
        body.appendChild(el('h4', {}, 'TP/FP/FN/TN 样本'));
        const sb = a.sample_breakdown;
        if (sb.TP) {
          body.appendChild(el('div', { class: 'kv' },
            el('div', {class:'k'}, 'positive_class'), el('div',{class:'v'}, String(sb.positive_class)),
            el('div', {class:'k'}, 'negative_class'), el('div',{class:'v'}, String(sb.negative_class)),
            el('div', {class:'k'}, 'TP'), el('div',{class:'v'}, `${sb.TP.length} keys: ${sb.TP.slice(0,5).join(', ')}${sb.TP.length>5?'...':''}`),
            el('div', {class:'k'}, 'FP'), el('div',{class:'v'}, `${sb.FP.length} keys: ${sb.FP.slice(0,5).join(', ')}${sb.FP.length>5?'...':''}`),
            el('div', {class:'k'}, 'FN'), el('div',{class:'v'}, `${sb.FN.length} keys: ${sb.FN.slice(0,5).join(', ')}${sb.FN.length>5?'...':''}`),
            el('div', {class:'k'}, 'TN'), el('div',{class:'v'}, `${sb.TN.length} keys: ${sb.TN.slice(0,5).join(', ')}${sb.TN.length>5?'...':''}`),
          ));
        }
      }
      // loss curves
      if (a.train_loss_history) {
        body.appendChild(el('h4', {}, '训练曲线'));
        const c = el('canvas', { id: 'ar-loss-' + aid, height: '220' });
        body.appendChild(c);
        setTimeout(() => {
          new Chart(c, {
            type: 'line',
            data: {
              labels: a.train_loss_history.map((_, i) => i),
              datasets: [
                { label: 'train', data: a.train_loss_history, borderColor: '#4f46e5', fill: false },
                { label: 'val', data: a.val_loss_history || [], borderColor: '#16a34a', fill: false },
              ]
            }
          });
        }, 50);
      }
    }
    // clustering
    if (a.task === 'clustering' && a.clusters) {
      body.appendChild(el('h4', {}, '聚类结果分布'));
      const counts = {};
      Object.values(a.clusters).forEach(c => counts[c] = (counts[c]||0) + 1);
      const cv = el('canvas', { id: 'ar-cl-' + aid, height: '200' });
      body.appendChild(cv);
      setTimeout(() => new Chart(cv, {
        type: 'bar', data: { labels: Object.keys(counts),
          datasets: [{ label: 'count', data: Object.values(counts), backgroundColor: '#7c3aed' }] }
      }), 50);
      body.appendChild(el('div', { class: 'kv' },
        el('div',{class:'k'},'inertia'), el('div',{class:'v'}, String(a.inertia))));
    }
    // feature importances
    if (a.feature_importances) {
      body.appendChild(el('h4', {}, '特征重要性'));
      const cv = el('canvas', { id: 'ar-fi-' + aid, height: '200' });
      body.appendChild(cv);
      setTimeout(() => new Chart(cv, {
        type: 'bar', data: { labels: a.feature_importances.map((_,i)=>i),
          datasets: [{ label: 'importance', data: a.feature_importances, backgroundColor: '#f59e0b' }] }
      }), 50);
    }
    // HPO history
    if (a.hpo_result?.history?.length) {
      body.appendChild(el('h4', {}, 'HPO 搜索历史'));
      const cv = el('canvas', { id: 'ar-hpo-' + aid, height: '200' });
      body.appendChild(cv);
      const hist = a.hpo_result.history;
      setTimeout(() => new Chart(cv, {
        type: 'line', data: { labels: hist.map(h=>h.idx),
          datasets: [{ label: 'score', data: hist.map(h=>h.score), borderColor:'#dc2626', fill:false}] }
      }), 50);
    }
  } catch (e) { toast(e.message, 'error'); }
}
$('#ar-close').addEventListener('click', () => $('#ar-detail').classList.add('hidden'));

// ============================================================
// Init
// ============================================================
onTabActivate('datasets');

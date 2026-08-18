// 在 node 里用最小 DOM 桩把生成好的报告页脚本原样跑一遍，把每个 innerHTML 快照下来。
//
//   node report_page_snapshot.js <report.html> <out.json>
//
// 为什么不在 Python 里另写一份表格计算：报告的每一格都带着口径（tooltip 里那些
// 「同锚/减了什么/为什么是 —」），另写一份必然和页面漂移。这里直接跑页面自己的代码，
// 天然不可能分叉。
//
// 关键点：报告主体是一个 IIFE，只通过 `window.xxx = ...` 往外挂符号（ensurePlot / sw /
// __dbgAlign …），所以 window 必须**就是** vm 的 global 本身，否则 IIFE 外的 sw() 里
// 引用 ensurePlot 会 ReferenceError。
'use strict';

const fs = require('fs');
const vm = require('vm');

const htmlPath = process.argv[2];
const outPath = process.argv[3];
if (!htmlPath || !outPath) {
  console.error('usage: node report_page_snapshot.js <report.html> <out.json>');
  process.exit(2);
}
const html = fs.readFileSync(htmlPath, 'utf-8');

// 取最长的 inline <script>（报告代码；另一条是 plotly CDN，有 src= 跳过）
let source = null;
const tagRe = /<script([^>]*)>/g;
let tag;
while ((tag = tagRe.exec(html)) !== null) {
  if (/src=/.test(tag[1])) continue;
  const end = html.indexOf('</script>', tag.index + tag[0].length);
  if (end < 0) continue;
  const body = html.slice(tag.index + tag[0].length, end);
  if (!source || body.length > source.length) source = body;
}
if (!source) {
  console.error('no inline <script> in ' + htmlPath);
  process.exit(2);
}

const CAPTURED = {};
const ORDER = [];

function makeElement(id) {
  const el = {
    id: id || '', _html: '', style: {}, dataset: {}, value: '', textContent: '',
    checked: false, disabled: false, children: [], options: [], selectedIndex: 0,
    offsetWidth: 800, offsetHeight: 400, clientWidth: 800, clientHeight: 400,
    scrollTop: 0, scrollHeight: 400,
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    addEventListener() {}, removeEventListener() {}, dispatchEvent() { return true; },
    appendChild(child) { this.children.push(child); return child; },
    insertBefore(child) { this.children.push(child); return child; },
    removeChild() {}, remove() {},
    setAttribute() {}, removeAttribute() {}, getAttribute() { return null; },
    querySelector() { return null; }, querySelectorAll() { return []; },
    getBoundingClientRect() {
      return { left: 0, top: 0, width: 800, height: 400, right: 800, bottom: 400 };
    },
    scrollIntoView() {}, focus() {}, blur() {}, click() {},
    on() {}, once() {}, closest() { return null; }, contains() { return false; },
  };
  Object.defineProperty(el, 'innerHTML', {
    get() { return this._html; },
    set(value) {
      this._html = String(value);
      if (this.id) {
        if (!(this.id in CAPTURED)) ORDER.push(this.id);
        CAPTURED[this.id] = this._html;
      }
    },
  });
  Object.defineProperty(el, 'innerText', {
    get() { return this._html; },
    set(value) { this.textContent = String(value); },
  });
  return el;
}

const elements = new Map();
const documentStub = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
  },
  createElement() { return makeElement(''); },
  querySelector() { return null; },
  querySelectorAll() { return []; },
  addEventListener() {}, removeEventListener() {}, dispatchEvent() { return true; },
  body: makeElement('body'),
  documentElement: makeElement('html'),
  readyState: 'complete',
};

const plotlyStub = {
  newPlot(target) {
    return Promise.resolve(typeof target === 'string' ? documentStub.getElementById(target) : target);
  },
  react() { return Promise.resolve(); },
  relayout() { return Promise.resolve(); },
  restyle() { return Promise.resolve(); },
  update() { return Promise.resolve(); },
  addTraces() { return Promise.resolve(); },
  deleteTraces() { return Promise.resolve(); },
  purge() {},
  Plots: { resize() {} },
};

const sandbox = {
  document: documentStub,
  Plotly: plotlyStub,
  console,
  setTimeout, clearTimeout, setInterval, clearInterval,
  navigator: { userAgent: 'node' },
  location: { href: '', search: '' },
  devicePixelRatio: 1, innerWidth: 1600, innerHeight: 900,
  addEventListener() {}, removeEventListener() {}, dispatchEvent() { return true; },
  getComputedStyle() { return { getPropertyValue() { return ''; } }; },
  matchMedia() { return { matches: false, addEventListener() {} }; },
  requestAnimationFrame(fn) { return setTimeout(fn, 0); },
  cancelAnimationFrame(id) { clearTimeout(id); },
  alert() {}, prompt() { return null; },
  Event: class Event { constructor(type) { this.type = type; } },
  CustomEvent: class CustomEvent {
    constructor(type, init) { this.type = type; this.detail = init && init.detail; }
  },
  MouseEvent: class MouseEvent { constructor(type) { this.type = type; } },
  KeyboardEvent: class KeyboardEvent { constructor(type) { this.type = type; } },
  URL, URLSearchParams, TextEncoder, TextDecoder,
};
const context = vm.createContext(sandbox);
vm.runInContext('this.window = this; this.self = this; this.globalThis = this;', context);

let error = null;
try {
  vm.runInContext(source, context, { filename: 'report.js', timeout: 900000 });
} catch (e) {
  error = String((e && e.stack) || e);
}

// 逐个 tab 切一遍，逼出懒加载的面板/表格（页面自己只初始化默认那一个）
const tabErrors = [];
for (let idx = 0; idx <= 7; idx += 1) {
  try {
    if (typeof context.sw === 'function') context.sw(idx);
  } catch (e) {
    tabErrors.push('sw(' + idx + '): ' + String(e).split('\n')[0]);
  }
}

const align = {};
try {
  const dbg = context.__dbgAlign;
  if (dbg) {
    align.auto = dbg.auto();
    if (dbg.clockBridge) align.clockBridge = dbg.clockBridge();
    if (dbg.poseLock) align.poseLock = dbg.poseLock();
    if (dbg.clockAnchor) align.clockAnchor = dbg.clockAnchor();
  }
  align.timeMap = context.__rkTimeMap || null;
} catch (e) {
  align.error = String(e);
}

fs.writeFileSync(
  outPath,
  JSON.stringify({ order: ORDER, captured: CAPTURED, align, error, tabErrors }),
  'utf-8',
);
console.log(JSON.stringify({
  ids: ORDER.length,
  error: error ? error.split('\n')[0] : null,
  tabErrors,
}));

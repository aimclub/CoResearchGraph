/* Tactile workshop pieces. All labels, selection and links remain live SVG. */
(() => {
  'use strict';
  const shortTitles = {
    OrchestratorAgent: 'Координатор', RootOrchestrator: 'Координатор синтеза',
    ContextInitAgent: 'Подготовка', PlannerAgent: 'План исследования', ResearchAgent: 'Литература',
    HypothesesAgent: 'Гипотезы', ExperimentModuleAgent: 'Эксперимент', ExperimentExecutorAgent: 'Исполнитель',
    ExperimentPlannerAgent: 'План эксперимента', ExperimentResultReviewAgent: 'Проверка',
    ExperimentAgent: 'ReAct · MCP', DatasetCollectorAgent: 'Сбор данных', MedicalAgent: 'Медицина',
    __mas_session__: 'Конфигуратор сессии',
  };
  function definitions(el) {
    const defs = el('g');
    [['workshop-ceramic', ['#fff9ed', '#f1e6d4', '#e0cdb2']],
      ['workshop-brass', ['#5a3615', '#f5d88c', '#ab7532', '#e6bd69', '#583716']]].forEach(([id, colors]) => {
      const gradient = el('linearGradient', { id, x1: 0, y1: 0, x2: id.endsWith('brass') ? 1 : .6, y2: 1 });
      colors.forEach((color, i) => gradient.append(el('stop', { offset: `${i / (colors.length - 1) * 100}%`, 'stop-color': color })));
      defs.append(gradient);
    });
    const shadow = el('filter', { id: 'workshop-shadow', x: '-20%', y: '-20%', width: '150%', height: '160%' });
    shadow.append(el('feDropShadow', { dx: 2, dy: 8, stdDeviation: 4, 'flood-color': '#03131e', 'flood-opacity': .65 }));
    defs.append(shadow); return [...defs.children];
  }
  function shape(el, card, className) {
    const w = card.w, h = card.h;
    if (card.shape === 'circle') return el('circle', { class: className, cx: w / 2, cy: h / 2, r: w / 2 });
    if (card.shape === 'hexagon') return el('path', { class: className, d: `M${w / 4},0 H${w * .75} L${w},${h / 2} ${w * .75},${h} H${w / 4} L0,${h / 2} Z` });
    return el('rect', { class: className, width: w, height: h, rx: 7 });
  }
  function draw({ el, g, card, node, title, icons, titleLines }) {
    g.dataset.shape = card.shape; g.dataset.width = card.w; g.dataset.height = card.h;
    // The material contains no typography. Its top face follows the same
    // geometric boundary used for hit testing and connection routing.
    const material = el('svg', { class: 'workshop-material', x: -2, y: -2, width: card.w + 4, height: card.h + 12,
      viewBox: { circle: '50 35 1140 1165', rect: '34 43 1862 717', hexagon: '17 143 1220 995' }[card.shape],
      preserveAspectRatio: 'none', overflow: 'visible', 'aria-hidden': true });
    const imageSize = card.shape === 'rect' ? [1920, 800] : [1280, 1280];
    material.append(el('image', { href: `/static/images/workshop/ceramic-${card.shape}.png`, width: imageSize[0], height: imageSize[1] }));
    const face = shape(el, card, 'node-card');
    g.append(material, face);
    const titleText = shortTitles[card.agentId] || title, tile = card.shape !== 'rect';
    const iconX = tile ? card.w / 2 - 14 : 16, iconY = tile ? card.h * .21 : card.h / 2 - 14;
    const icon = el('g', { class: 'node-icon', transform: `translate(${iconX},${iconY}) scale(1.15)`, 'aria-hidden': true });
    (icons[node?.icon] || icons.agent).forEach(d => icon.append(el('path', { d }))); g.append(icon);
    const textWidth = tile ? card.w - 26 : card.w - 68;
    let size = 18, lines = titleLines(titleText, textWidth, size, 'Georgia, serif');
    // A long Russian role name should remain a word, not an ellipsis on a token.
    while (size > 14 && lines.some(line => line.endsWith('…'))) {
      size--; lines = titleLines(titleText, textWidth, size, 'Georgia, serif');
    }
    const y = tile ? card.h * .64 - (lines.length - 1) * 9 : card.h / 2 + 6 - (lines.length - 1) * 10;
    lines.forEach((line, i) => g.append(el('text', { class: 'node-title', x: tile ? card.w / 2 : 54,
      y: y + i * 20, 'text-anchor': tile ? 'middle' : 'start', style: `font-size:${size}px` }, line)));
    return g;
  }
  function port(el, p) {
    const g = el('g', { class: 'workshop-terminal', 'aria-hidden': true, transform: `translate(${p.x},${p.y})` });
    g.append(el('rect', { x: -5, y: -3, width: 10, height: 10, rx: 2.5, fill: '#543718' }),
      el('rect', { x: -5, y: -5, width: 10, height: 10, rx: 2.5, fill: 'url(#workshop-brass)', stroke: '#563d1e', 'stroke-width': .8 }),
      el('path', { d: 'M-2,-4 V4 M2,-4 V4', stroke: '#fff0b9', 'stroke-width': .7, opacity: .75 }));
    return g;
  }
  window.AgentTreeWorkshop = { definitions, draw, port };
})();

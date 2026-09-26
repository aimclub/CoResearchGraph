/* Tactile workshop pieces. All labels, selection and links remain live SVG. */
(() => {
  'use strict';
  const shortTitles = {
    OrchestratorAgent: 'Координатор', RootOrchestrator: 'Координатор синтеза',
    ContextInitAgent: 'Подготовка', PlannerAgent: 'План исследования', ResearchAgent: 'Литература',
    HypothesesAgent: 'Гипотезы', ExperimentModuleAgent: 'Эксперимент', ExperimentExecutorAgent: 'Исполнитель',
    ExperimentPlannerAgent: 'План эксперимента', ExperimentResultReviewAgent: 'Проверка',
    ExperimentAgent: 'ReAct · MCP', DatasetCollectorAgent: 'Сбор данных', MedicalAgent: 'Медицина',
    __mas_session__: 'Настройки',
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
    shadow.append(el('feDropShadow', { dx: 1, dy: 4, stdDeviation: 2.5, 'flood-color': '#03131e', 'flood-opacity': .55 }));
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
    const bottom = shape(el, card, 'workshop-base'); bottom.setAttribute('transform', 'translate(0,3)');
    const face = shape(el, card, 'node-card');
    const rim = shape(el, card, 'workshop-glaze');
    rim.setAttribute('transform', `translate(3,3) scale(${(card.w - 6) / card.w},${(card.h - 6) / card.h})`);
    g.append(bottom, face, rim);
    const titleText = shortTitles[card.agentId] || title, tile = card.shape !== 'rect';
    const iconX = tile ? card.w / 2 - 14 : 16, iconY = tile ? card.h * .21 : card.h / 2 - 14;
    const icon = el('g', { class: 'node-icon', transform: `translate(${iconX},${iconY}) scale(1.15)`, 'aria-hidden': true });
    (icons[node?.icon] || icons.agent).forEach(d => icon.append(el('path', { d }))); g.append(icon);
    const textWidth = tile ? card.w - 18 : card.w - 62;
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
    g.append(el('rect', { x: -4.2, y: -4.2, width: 8.4, height: 8.4, rx: 2, fill: 'url(#workshop-brass)', stroke: '#563d1e', 'stroke-width': .8 }),
      el('path', { d: 'M-1.8,-3 V3 M1.8,-3 V3', stroke: '#fff0b9', 'stroke-width': .6, opacity: .75 }));
    return g;
  }
  window.AgentTreeWorkshop = { definitions, draw, port };
})();

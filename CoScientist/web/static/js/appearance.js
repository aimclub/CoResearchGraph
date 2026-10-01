// Per-browser appearance: theme, accent colour and UI font.
//
// Loaded synchronously in <head>, before Tailwind and the stylesheet, so the
// stored choice is on <html> before the first paint (no flash of the default
// look on reload). Everything here only sets classes and CSS variables on
// <html>; main.css holds the palettes, the settings modal holds the controls.

const THEME_KEY = 'cos-theme';
const ACCENT_KEY = 'cos-accent';
const FONT_KEY = 'cos-font';
const LIGHT_DIM_KEY = 'cos-light-dim';

// Presets shown as swatches. `null` accent = the theme's own blue; the first
// preset is the brand cyan the interface used before.
const ACCENT_PRESETS = ['#00DAF3', '#8B5CF6', '#10B981', '#F59E0B', '#F43F5E'];

// UI fonts. Source Sans 3 is vendored (works offline); the others are fetched
// from Google Fonts only once chosen. All of them cover Cyrillic.
const UI_FONTS = {
  'source-sans': { name: 'Source Sans 3', stack: "'Source Sans 3', sans-serif" },
  'inter': { name: 'Inter', stack: "'Inter', sans-serif", css: 'Inter:wght@400..700' },
  'ibm-plex-sans': { name: 'IBM Plex Sans', stack: "'IBM Plex Sans', sans-serif", css: 'IBM+Plex+Sans:wght@400;500;600;700' },
  'golos': { name: 'Golos Text', stack: "'Golos Text', sans-serif", css: 'Golos+Text:wght@400..700' },
  'manrope': { name: 'Manrope', stack: "'Manrope', sans-serif", css: 'Manrope:wght@400..700' },
  'nunito-sans': { name: 'Nunito Sans', stack: "'Nunito Sans', sans-serif", css: 'Nunito+Sans:wght@400;600;700' },
  'pt-sans': { name: 'PT Sans', stack: "'PT Sans', sans-serif", css: 'PT+Sans:wght@400;700' },
  'system': { name: 'System', stack: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif" },
};
const DEFAULT_FONT = 'source-sans';

// Light theme brightness (Interface → Light theme brightness): every light
// token runs from its brightest value (dim 0) to a muted grey one (dim 100).
// Text darkens along with the surfaces, so muted text keeps >= 4.5:1 and body
// text >= 11:1 over the whole range. The html.light block in main.css holds
// the DEFAULT_LIGHT_DIM point of this ramp for the first paint.
const DEFAULT_LIGHT_DIM = 50;
const LIGHT_RAMP = {
  'surface-container-lowest': [[246, 248, 250], [214, 220, 227]],
  'surface-bright': [[246, 248, 250], [214, 220, 227]],
  'surface-container-low': [[239, 242, 246], [206, 213, 221]],
  'background': [[233, 237, 242], [200, 207, 215]],
  'surface': [[233, 237, 242], [200, 207, 215]],
  'surface-container': [[228, 233, 238], [194, 201, 210]],
  'surface-dim': [[224, 229, 235], [180, 188, 198]],
  'surface-container-high': [[221, 227, 233], [186, 194, 204]],
  'surface-container-highest': [[210, 218, 226], [175, 184, 195]],
  'surface-variant': [[210, 218, 226], [175, 184, 195]],
  'hairline': [[200, 209, 219], [165, 175, 188]],
  'on-surface': [[20, 37, 58], [12, 26, 44]],
  'on-background': [[20, 37, 58], [12, 26, 44]],
  'on-surface-variant': [[64, 80, 102], [42, 55, 74]],
  'outline': [[64, 78, 98], [40, 52, 70]],
  'outline-variant': [[82, 98, 120], [52, 65, 84]],
  'primary-text': [[30, 88, 196], [22, 66, 150]],
  'shimmer': [[30, 88, 196], [22, 66, 150]],
  'primary-container': [[214, 228, 252], [180, 198, 232]],
  'secondary': [[18, 115, 60], [10, 84, 44]],
  'tertiary': [[133, 92, 0], [96, 66, 0]],
  'error': [[180, 35, 35], [140, 22, 22]],
  'error-container': [[250, 222, 219], [222, 190, 187]],
};
// Accent text must stay readable on the dark page, and on the darkest light
// surface (inputs/chips) at the current brightness.
const DARK_TEXT_BG = [16, 19, 25];
const NAVY = [3, 22, 41];
const WHITE = [255, 255, 255];

function readStored(key) {
  try { return localStorage.getItem(key); } catch (_) { return null; }
}

function writeStored(key, value) {
  try {
    if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, value);
  } catch (_) { }
}

let currentTheme = readStored(THEME_KEY) === 'light' ? 'light' : 'dark';
let currentAccent = /^#[0-9a-f]{6}$/i.test(readStored(ACCENT_KEY) || '') ? readStored(ACCENT_KEY).toUpperCase() : null;
let currentFont = UI_FONTS[readStored(FONT_KEY)] ? readStored(FONT_KEY) : DEFAULT_FONT;
let currentLightDim = clampDim(readStored(LIGHT_DIM_KEY));

function clampDim(value) {
  const n = Number(value);
  return value === null || value === '' || !Number.isFinite(n) ? DEFAULT_LIGHT_DIM : Math.min(100, Math.max(0, Math.round(n)));
}

// ── colour maths (WCAG relative luminance) ──────────────────────────────────
function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function luminance(rgb) {
  const [r, g, b] = rgb.map(v => {
    v /= 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

function mix(a, b, t) {
  return a.map((v, i) => Math.round(v + (b[i] - v) * t));
}

// Nudge the accent toward white (dark theme) or black (light theme) until it
// reads as text on that theme's surfaces.
function readableOn(rgb, bg, toward) {
  for (let t = 0; t <= 1; t += 0.04) {
    const c = mix(rgb, toward, t);
    if (contrast(c, bg) >= 4.5) return c;
  }
  return toward;
}

const channels = rgb => rgb.join(' ');

// ── apply ───────────────────────────────────────────────────────────────────
// Every variable this file may set inline; anything not in the computed map
// is removed so main.css's palette shows through.
const MANAGED_VARS = [...new Set([
  ...Object.keys(LIGHT_RAMP),
  'primary', 'primary-text', 'on-primary', 'primary-container', 'on-primary-container', 'shimmer',
])];

function lightTone(token) {
  const [bright, dim] = LIGHT_RAMP[token];
  return mix(bright, dim, currentLightDim / 100);
}

function applyColors() {
  const root = document.documentElement;
  const light = currentTheme === 'light';
  const vars = {};
  if (light) Object.keys(LIGHT_RAMP).forEach(token => { vars[token] = lightTone(token); });
  if (currentAccent) {
    const base = hexToRgb(currentAccent);
    const text = light
      ? readableOn(base, lightTone('surface-container-high'), [0, 0, 0])
      : readableOn(base, DARK_TEXT_BG, WHITE);
    vars.primary = base;
    vars['primary-text'] = text;
    vars['on-primary'] = contrast(NAVY, base) >= contrast(WHITE, base) ? NAVY : WHITE;
    vars['primary-container'] = light ? mix(base, lightTone('surface-container-lowest'), 0.8) : mix(base, [0, 0, 0], 0.45);
    vars['on-primary-container'] = light ? NAVY : mix(base, WHITE, 0.9);
    if (light) vars.shimmer = text;
    root.dataset.accent = currentAccent;
  } else {
    delete root.dataset.accent;
  }
  MANAGED_VARS.forEach(token => {
    if (vars[token]) root.style.setProperty(`--c-${token}`, channels(vars[token]));
    else root.style.removeProperty(`--c-${token}`);
  });
}

function applyFont() {
  const font = UI_FONTS[currentFont];
  if (font.css && !document.getElementById(`font-${currentFont}`)) {
    const link = document.createElement('link');
    link.id = `font-${currentFont}`;
    link.rel = 'stylesheet';
    link.href = `https://fonts.googleapis.com/css2?family=${font.css}&display=swap`;
    document.head.appendChild(link);
  }
  if (currentFont === DEFAULT_FONT) document.documentElement.style.removeProperty('--font-ui');
  else document.documentElement.style.setProperty('--font-ui', font.stack);
}

function applyTheme() {
  const root = document.documentElement;
  root.classList.toggle('light', currentTheme === 'light');
  root.classList.toggle('dark', currentTheme === 'dark');
  applyColors();
}

function setTheme(theme) {
  currentTheme = theme === 'light' ? 'light' : 'dark';
  writeStored(THEME_KEY, currentTheme);
  applyTheme();
}

function setAccent(hex) {
  currentAccent = hex && /^#[0-9a-f]{6}$/i.test(hex) ? hex.toUpperCase() : null;
  writeStored(ACCENT_KEY, currentAccent);
  applyColors();
}

function setFont(id) {
  currentFont = UI_FONTS[id] ? id : DEFAULT_FONT;
  writeStored(FONT_KEY, currentFont === DEFAULT_FONT ? null : currentFont);
  applyFont();
}

function setLightDim(value) {
  currentLightDim = clampDim(value);
  writeStored(LIGHT_DIM_KEY, currentLightDim === DEFAULT_LIGHT_DIM ? null : String(currentLightDim));
  applyColors();
}

function resetAppearance() {
  setAccent(null);
  setFont(DEFAULT_FONT);
  setLightDim(DEFAULT_LIGHT_DIM);
}

applyTheme();
applyFont();

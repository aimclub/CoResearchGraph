// =========================================================================
// WebSocket & State / Global Utilities
// =========================================================================

// Session expiry, in one place. The server answers 401 to every API call once
// the cookie is gone, and the app makes those calls from apiJson and from a
// dozen direct fetch() sites. Wrapping fetch itself covers all of them, so no
// call site has to remember. state.js loads before every other app script.
(function guardAgainstExpiredSession() {
  const realFetch = window.fetch.bind(window);
  window.fetch = async function (...args) {
    const response = await realFetch(...args);
    if (response.status === 401 && window.location.pathname !== '/login') {
      window.location.href = '/login';
    }
    return response;
  };
})();

let ws = null;
let eventCount = 0;
let reconnectTimer = null;
let intentionalDisconnect = false;
let activeUser = null;
let activeSession = null;
let runStatusVersion = -1;
let knownUsers = [];
let knownSessions = [];
let datasetUrl = '';
let reportLanguage = '';
// True while the active session processes a run. Locks the language control.
let runActive = false;
let activeSandboxWatchUrl = null;

const USER_STORAGE_KEY = 'coscientist.user_id';
const SESSION_STORAGE_KEY = 'coscientist.session_id';
const NICK_STORAGE_KEY = 'coscientist.nickname';
const BOOT_STORAGE_KEY = 'coscientist.server_boot_id';
let serverBootId = null;
const SIDE_NAV_KEY = 'coscientist.side_nav';
const SIDE_NAV_WIDTH_KEY = 'coscientist.side_nav_width';
const RIGHT_PANEL_KEY = 'coscientist.right_panel';
const LANG_STORAGE_KEY = 'coscientist.lang';
const SHOW_INTERNAL_KEY = 'coscientist.show_internal';
const SHOW_HIDDEN_SESSIONS_KEY = 'coscientist.show_hidden_sessions';
const SIDE_RAIL_KEY = 'coscientist.side_rail';
const SIDE_RAIL_WIDTH_KEY = 'coscientist.side_rail_width';

// Per-browser view preference: show the agents and tools the system YAML marks
// internal. The html class drives the Work Order chips through CSS, so cards
// already on the page follow the switch without being redrawn. Only a choice
// that differs from the server default (SHOW_INTERNAL__ENABLED) is stored;
// without one the browser follows that default once /api/settings answers.
let showInternalStored = null;
try {
  const stored = localStorage.getItem(SHOW_INTERNAL_KEY);
  if (stored === 'true' || stored === 'false') showInternalStored = stored === 'true';
} catch (_) { }
let showInternal = !!showInternalStored;
document.documentElement.classList.toggle('show-internal', showInternal);

// Mirror of the server settings (/api/settings). The settings modal edits a
// draft copy and writes it back here after a successful save.
const appSettings = {
  general: {
    openrouterProviderSort: 'default', // 'default' | 'price' | 'throughput' | 'latency'
    openrouterProviderOrder: '',       // e.g. 'Together, DeepInfra'
    startMode: 'planner',   // 'planner' | 'orchestrator' | 'orchestrator_planner'
    maxRetries: 3,
    hitlEnabled: false,
    // auto | basic | debug — the one knob for every confirmation. See
    // CoScientist/hitl/mode.py; silence approves in none of the three.
    hitlMode: 'basic',
    hitlAutoApproveTimeout: -1,        // seconds; -1 = wait for the human
    workOrderEnabled: true,
    workOrderVetoSeconds: -1,          // seconds; -1 = wait for the human
    useProxy: true,                    // read-only: USE_PROXY in .env
    opikEnabled: false,
    autoNamingEnabled: true,
    showInternal: false,               // default only: SHOW_INTERNAL__ENABLED; the browser's choice wins
    callGraphCollapseSeconds: 10,      // side-nav call graph: unfolded parallel calls fold back after this
    contextInitEnabled: true,
    knowledgeGraphEnabled: true,
    autoClearGraphEnabled: false,      // read-only: GRAPH__AUTO_CLEAR in .env
    researchGraphEnabled: true,
    coscientistUsername: '',
  },
  researchAgent: {
    maxSearches: 2,
  },
  medicalAgent: {
    enabled: true,                     // MEDICAL__ENABLED; the agent and the experiment medical route
  },
  nirReport: {
    enabled: false,                    // NIR__ENABLED; offer the GOST 7.32-2017 DOCX at the end of a run
    available: false,                  // read-only: whether MCP__NORMCONTROL_URL is configured at all
  },
  taskExecutorAgent: {
    keepScore: 0.3,
    abstainScore: 0.2,
    fedotFallback: true,               // EXECUTOR__FEDOT_FALLBACK; FedotAgent in the main profile
  },
  coderAgent: {
    sandboxUrl: '',                    // empty: the sandbox button falls back to localhost:8884
    workspaceId: '',
    mode: 'local',
  },
  plannerAgent: {
    retrievalEnabled: true,
    graphEnabled: true,
    criticEnabled: false,
    criticRounds: 1,
    mergeTasksEnabled: true,
  },
  hypothesesAgent: {
    maxActiveHypotheses: 1,
  },
  // The experiment module's own reviews. Not under general.hitlEnabled: the
  // module asks for these two even when that switch is off, and a window that
  // runs out pauses the run instead of approving it.
  experimentModule: {
    planAutoApprove: false,
    resultAutoApprove: false,
    planReviewTimeoutS: 300,           // seconds; runs out => paused, not approved
    resultReviewTimeoutS: 300,
    routeFedot: false,                 // FEDOT.MAS route; applies to the next session
    routeAlembic: false,               // offer reviewed reuse→MCP wrapping
  },
  // Settings → Agents: the operator's changes over system.yaml, applied when
  // the next session's agent tree is built. Only what differs is stored.
  agents: {
    defaultReasoning: '',              // AGENTS__DEFAULT_REASONING; '' = as the profile declares
    overrides: {},                     // AGENTS__OVERRIDES; { AgentName: { enabled?, reasoning?, model? } }
  },
  alembicHub: {
    searchEnabled: true,
    autoUpload: false,
    agentBuildEnabled: false,
  },
};

function escHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function escJs(str) {
  if (!str) return '';
  return str.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"').replace(/\n/g, '\\n').replace(/\r/g, '\\r');
}

function ts(timestamp = null) {
  const date = timestamp ? new Date(timestamp) : new Date();
  return date.toLocaleTimeString('en-GB', { hour12: false });
}

function formatDatasetSize(mb) {
  const num = Number(mb) || 0;
  if (num >= 1024) {
    return (num / 1024).toFixed(1) + ' GB';
  }
  return Math.round(num) + ' MB';
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  let data = {};
  try { data = await response.json(); } catch (_) { /* empty response */ }
  if (!response.ok) throw new Error(data.detail || data.error || `HTTP ${response.status}`);
  return data;
}

function sessionApi(path = '') {
  if (!activeUser || !activeSession) throw new Error('Select a user and session first.');
  return `/api/users/${encodeURIComponent(activeUser.id)}/sessions/${encodeURIComponent(activeSession.id)}${path}`;
}

function roadmapUrl() { return sessionApi('/roadmap'); }

async function fetchErrorMessage(resp) {
  try {
    const data = await resp.json();
    if (data && data.detail) return data.detail;
  } catch (_) {
    try {
      const text = await resp.text();
      if (text) return text.slice(0, 200);
    } catch (_) { }
  }
  return resp.statusText || `HTTP ${resp.status}`;
}

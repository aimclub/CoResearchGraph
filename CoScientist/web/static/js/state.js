// =========================================================================
// WebSocket & State / Global Utilities
// =========================================================================
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
let activeSandboxWatchUrl = null;

const USER_STORAGE_KEY = 'coscientist.user_id';
const SESSION_STORAGE_KEY = 'coscientist.session_id';
const NICK_STORAGE_KEY = 'coscientist.nickname';
const SIDE_NAV_KEY = 'coscientist.side_nav';
const LANG_STORAGE_KEY = 'coscientist.lang';

const appSettings = {
  general: {
    startMode: 'planner',   // 'planner' | 'orchestrator' | 'orchestrator_planner'
    maxRetries: 3,
    hitlEnabled: false,
    hitlAutoApproveTimeout: 300,
    usePlanner: true,
    useProxy: true,
    opikEnabled: false,
    autoNamingEnabled: true,
    contextInitEnabled: true,
    knowledgeGraphEnabled: true,
    autoClearGraphEnabled: false,
    researchGraphEnabled: true,
  },
  researchAgent: {
    maxSearches: 2,
  },
  taskExecutorAgent: {
    keepScore: 0.3,
    abstainScore: 0.2,
  },
  coderAgent: {
    sandboxUrl: 'http://localhost:8884',
    workspaceId: '',
    mode: 'local',
  },
  orchestratorAgent: {},
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
  medicalAgent: {},
  experimentAgent: {},
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

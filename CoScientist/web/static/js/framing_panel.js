/* Structured sidebar for the research Technical specification card.
 * Loaded after graph.html's shared rendering helpers. */
const framingCache = new Map();
let framingRequestSequence = 0;

function framingText(value) {
  if (value && typeof value === "object") {
    const lang = (typeof currentLang === "string" && currentLang === "en") ? "en" : "ru";
    return String(value[lang] || value.ru || value.en || "");
  }
  return String(value || "");
}

function framingUi(ru, en) {
  return (typeof currentLang === "string" && currentLang === "en") ? en : ru;
}

function framingStudyId() {
  if (lastGraph && lastGraph.study_id) return String(lastGraph.study_id);
  const selected = document.getElementById("turn");
  return selected && selected.value ? String(selected.value) : "active";
}

function framingPanelKey(node) {
  const q = new URLSearchParams(location.search);
  return [q.get("user_id") || "", q.get("session_id") || "", framingStudyId(),
          node.framing_revision || "legacy",
          (typeof currentLang === "string" ? currentLang : "ru")].join("\u0000");
}

function framingOpenKey(studyId, sectionId) {
  const q = new URLSearchParams(location.search);
  return ["graph.framing.open", q.get("user_id") || "", q.get("session_id") || "",
          studyId, sectionId].join(":");
}

function framingIsOpen(studyId, section) {
  try {
    const remembered = localStorage.getItem(framingOpenKey(studyId, section.id));
    if (remembered !== null) return remembered === "1";
  } catch (e) { /* private browsing: use the declared default */ }
  return !!section.open_by_default;
}

function framingValue(field) {
  const value = String(field.value || "");
  if (!value) return `<div class="tz-field-value">${esc(framingUi("Не задано", "Not specified"))}</div>`;
  if (field.value_format === "markdown") {
    const rendered = markdown(value);
    if (rendered) return `<div class="tz-field-value markdown">${rendered}</div>`;
  }
  return `<div class="tz-field-value">${esc(value)}</div>`;
}

function framingField(field) {
  const source = String(field.source || "");
  return `<div class="tz-field${field.empty ? " empty" : ""}" data-field="${esc(field.id || "")}">`
    + `<div class="tz-field-label">${esc(framingText(field.label))}</div>`
    + `<div class="tz-field-help">${esc(framingText(field.help))}</div>`
    + framingValue(field)
    + `<div class="tz-field-origin"><span class="tz-origin-status">${esc(framingText(field.status_label))}</span>`
    + (source && !["human", "agent", "frame", "request"].includes(source)
        ? `<span>${esc(source)}</span>` : "")
    + `</div></div>`;
}

function framingFieldHasValue(field) {
  if (!field || field.empty) return false;
  const value = String(field.value || "").trim().toLocaleLowerCase();
  if (!value) return false;
  const missing = new Set(["", "-", "—", "не задано", "not specified", "none", "null"]);
  return !value.split(/\s*[\/|]\s*/).every(part => missing.has(part.trim()));
}

function framingVisibleSections(data) {
  return (data.sections || []).map(section => ({
    ...section,
    fields: (section.fields || []).filter(framingFieldHasValue),
    documents: section.documents || [],
  })).filter(section => section.fields.length || section.documents.length);
}

function framingSafeHref(value) {
  const href = String(value || "").trim();
  if (!href) return "";
  try {
    const url = new URL(href, location.origin);
    return ["http:", "https:"].includes(url.protocol) ? href : "";
  } catch (e) {
    return "";
  }
}

function framingDocument(document) {
  const href = framingSafeHref(document.href);
  const content = String(document.content || "");
  const rendered = content ? (markdown(content) || `<pre>${esc(content)}</pre>`) : "";
  return `<div class="tz-document">`
    + `<div class="tz-document-row"><span class="tz-document-title">${esc(framingText(document.title))}</span>`
    + (document.format ? `<span class="tz-document-format">${esc(document.format)}</span>` : "")
    + (href ? `<a href="${esc(href)}" target="_blank" rel="noopener noreferrer">${esc(framingUi("Открыть", "Open"))}</a>` : "")
    + `</div>`
    + (rendered ? `<div class="tz-document-content report">${rendered}</div>` : "")
    + `</div>`;
}

function framingSection(studyId, section) {
  const fields = (section.fields || []).map(framingField).join("");
  const documents = (section.documents || []).map(framingDocument).join("");
  return `<details class="tz-section" id="tz-section-${esc(section.id)}" data-section="${esc(section.id)}"`
    + (framingIsOpen(studyId, section) ? " open" : "") + `>`
    + `<summary>${esc(framingText(section.title))}</summary>`
    + fields + documents + `</details>`;
}

function framingAdditional(items) {
  if (!Array.isArray(items) || !items.length) return "";
  return `<details class="tz-section"><summary>${esc(framingUi("Дополнительные параметры", "Additional parameters"))}</summary>`
    + items.map(item => `<div class="tz-extra-note"><strong>${esc(framingText(item.label))}</strong>`
      + `<div>${esc(String(item.value || ""))}</div>`
      + `<div class="tz-field-help">${esc(framingText(item.note))}</div></div>`).join("")
    + `</details>`;
}

function framingHeader(data, filled) {
  return `<div class="dhead"><span class="close" onclick="closeDetail()">✕</span>`
    + `<h3>${esc(framingUi("Техническое задание", "Technical specification"))}</h3>`
    + (data.title ? `<div class="tz-head-title">${esc(data.title)}</div>` : "")
    + `<div class="tz-head-meta"><span class="tz-source-note">${esc(framingText(data.source_note))}</span>`
    + `<span class="tz-completion">${esc(
        framingUi(`Указано параметров: ${filled}`, `Specified fields: ${filled}`)
      )}</span></div></div>`;
}

function bindFramingPanel(data) {
  const box = document.getElementById("detail");
  box.querySelectorAll(".tz-section[data-section]").forEach(section => {
    section.addEventListener("toggle", () => {
      try {
        localStorage.setItem(framingOpenKey(data.study_id, section.dataset.section),
                             section.open ? "1" : "0");
      } catch (e) { /* persistence is optional */ }
    });
  });
  const nav = box.querySelector(".tz-nav select");
  if (nav) nav.addEventListener("change", () => {
    const target = box.querySelector(`#tz-section-${CSS.escape(nav.value)}`);
    if (!target) return;
    target.open = true;
    target.scrollIntoView({ block: "start", behavior: "smooth" });
  });
}

function renderFramingPanel(node, data) {
  const sections = framingVisibleSections(data);
  const filled = sections.reduce((count, section) => count + section.fields.length, 0);
  const nav = `<div class="tz-nav"><label for="tz-section-picker">${esc(framingUi("Перейти к разделу", "Go to section"))}</label>`
    + `<select id="tz-section-picker"><option value="">${esc(framingUi("Выберите раздел", "Choose a section"))}</option>`
    + sections.map(section => `<option value="${esc(section.id)}">${esc(framingText(section.title))}</option>`).join("")
    + `</select></div>`;
  const body = nav + sections.map(section => framingSection(data.study_id, section)).join("")
    + framingAdditional(data.additional_context);
  paint(node, framingHeader(data, filled) + `<div class="dbody">${body}</div>`);
  bindFramingPanel(data);
}

function framingLoadingHeader(node, error) {
  const title = `<div class="dhead"><span class="close" onclick="closeDetail()">✕</span>`
    + `<h3>${esc(framingUi("Техническое задание", "Technical specification"))}</h3></div>`;
  const content = error
    ? `<div class="tz-load-error">${esc(error)}</div>`
    : `<div class="tz-loading">${esc(framingUi("Загружается структура задания…", "Loading specification…"))}</div>`;
  paint(node, title + `<div class="dbody">${content}</div>`);
}

async function fetchFramingDetails(node, key, sequence) {
  const q = new URLSearchParams(location.search);
  const userId = q.get("user_id") || "";
  const sessionId = q.get("session_id") || "";
  const studyId = framingStudyId();
  const url = `/api/users/${encodeURIComponent(userId)}/sessions/${encodeURIComponent(sessionId)}`
    + `/graph/framing?study_id=${encodeURIComponent(studyId)}`;
  try {
    const response = await fetch(url, { cache: "no-store" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || data.error || (`HTTP ${response.status}`));
    framingCache.set(key, data);
    if (sequence !== framingRequestSequence || selectedId !== node.id
        || framingStudyId() !== studyId) return;
    renderFramingPanel(nodeData[node.id] || node, data);
  } catch (error) {
    if (sequence !== framingRequestSequence || selectedId !== node.id) return;
    framingLoadingHeader(node, String(error.message || error));
  }
}

function showFramingDetail(node) {
  // The document layout benefits from a little more room. Respect a width the
  // reader has already chosen for the shared graph panel.
  try {
    if (!localStorage.getItem("graph.detailWidth")) setDetailWidth(560, false);
  } catch (e) { /* private browsing */ }

  const sequence = ++framingRequestSequence;
  const key = framingPanelKey(node);
  const cached = framingCache.get(key);
  if (cached) {
    renderFramingPanel(node, cached);
    return;
  }
  framingLoadingHeader(node, "");
  fetchFramingDetails(node, key, sequence);
}

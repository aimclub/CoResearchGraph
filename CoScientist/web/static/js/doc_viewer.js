// =========================================================================
// Document panel — the right column shows what a message points at.
//
// An agent's deliverable, an experiment plan, a work report: each runs to tens
// of kilobytes, and printed into the feed it buries every message around it. So
// the backend writes the body into the session's artifact store and the message
// carries a summary and an id; this opens that file here.
//
// One panel, not one per message. Pressing "Open" on another message replaces
// what is showing — which is the whole point of a panel over a modal.
// =========================================================================
(function () {
  'use strict';

  // artifact_id -> markdown. A document is content-addressed and immutable, so
  // what was fetched once is still true; reopening is free.
  const cache = new Map();
  let shownId = null;
  // The rail's width before a document claimed it, restored on close.
  let widthBefore = null;
  // A document needs more than the 320px a plan list is happy in.
  const DOC_MIN_WIDTH = 640;

  function el(id) { return document.getElementById(id); }

  function railWidth() {
    const rail = el('side-rail');
    return rail ? Math.round(rail.getBoundingClientRect().width) : 0;
  }

  // Opening must also undo a collapsed rail, or the button would look broken:
  // `body.rail-collapsed` hides the whole column, panel and all.
  function claimTheColumn() {
    if (document.body.classList.contains('rail-collapsed')) {
      if (typeof toggleSideRail === 'function') toggleSideRail();
    }
    if (typeof setSideRailWidth !== 'function') return;
    const current = railWidth();
    if (current >= DOC_MIN_WIDTH) return;
    if (widthBefore === null) widthBefore = current;
    setSideRailWidth(DOC_MIN_WIDTH, false);
  }

  function releaseTheColumn() {
    if (widthBefore !== null && typeof setSideRailWidth === 'function') {
      setSideRailWidth(widthBefore, false);
    }
    widthBefore = null;
  }

  // A document is stored scope-free on purpose: `cos-artifact:<id>` names the
  // file and nothing else, so the same bytes resolve under whatever session is
  // reading them — which is what lets an imported bundle work with nothing
  // rewritten. The URL is therefore built here, at the moment of rendering, and
  // never written into the file.
  const REF_RE = /cos-artifact:([A-Za-z0-9][A-Za-z0-9._-]{0,127})/g;
  const S3_RE = /s3:\/\/([A-Za-z0-9][A-Za-z0-9._-]*)\/([^\s)\]<>"']+)/g;

  function resolveRefs(markdown) {
    return String(markdown == null ? '' : markdown)
      .replace(REF_RE, (_m, id) => sessionApi('/artifacts/' + encodeURIComponent(id)))
      // An `s3://bucket/key` is addressable too, through the route that mints a
      // fresh download URL for it.
      .replace(S3_RE, (_m, bucket, key) =>
        '/api/artifact/' + encodeURIComponent(bucket) + '/'
        + key.split('/').map(encodeURIComponent).join('/'));
  }

  function show(html) {
    const body = el('doc-viewer-body');
    if (body) {
      body.innerHTML = html;
      body.scrollTop = 0;
    }
  }

  function notice(key, fallback) {
    return `<p class="text-sm text-outline-variant italic">${escHtml(t(key, fallback))}</p>`;
  }

  async function fetchDocument(artifactId) {
    if (cache.has(artifactId)) return cache.get(artifactId);
    const response = await fetch(sessionApi('/artifacts/' + encodeURIComponent(artifactId)));
    if (!response.ok) throw new Error('HTTP ' + response.status);
    // Read the bytes rather than following the link: the route may serve a
    // document as an attachment, and a download is not a panel.
    const text = await response.text();
    cache.set(artifactId, text);
    return text;
  }

  async function openDocument(artifactId, title, forRequest) {
    if (!artifactId) return;
    const panel = el('doc-viewer');
    if (!panel) return;

    panel.classList.remove('hidden');
    claimTheColumn();
    // Anything the reader opens themselves makes the panel theirs, so a later
    // answer does not close it under them.
    if (!forRequest) openedByRequest = null;
    const heading = el('doc-viewer-title');
    if (heading) heading.textContent = title || t('doc.untitled', 'Document');

    if (shownId !== artifactId) show(notice('doc.loading', 'Loading…'));
    shownId = artifactId;
    markOpenButton(artifactId);
    refreshList();

    let text;
    try {
      text = await fetchDocument(artifactId);
    } catch (error) {
      console.warn('document not loaded:', error);
      // Another button may have won the race while this one was in flight.
      if (shownId === artifactId) show(notice('doc.failed', 'Could not open this document.'));
      return;
    }
    if (shownId !== artifactId) return;
    show(renderMarkdown(resolveRefs(text)));
    panel.scrollIntoView({ block: 'nearest' });
  }

  function closeDocument() {
    openedByRequest = null;
    const panel = el('doc-viewer');
    if (panel) panel.classList.add('hidden');
    shownId = null;
    markOpenButton(null);
    releaseTheColumn();
  }

  // The button of the document on screen reads differently from the rest, so a
  // feed full of buttons still says which one you are looking at.
  function markOpenButton(artifactId) {
    document.querySelectorAll('[data-doc-id]').forEach(button => {
      const isShown = !!artifactId && button.dataset.docId === artifactId;
      button.classList.toggle('doc-btn-open', isShown);
      button.setAttribute('aria-pressed', String(isShown));
    });
  }

  // ── Session documents ───────────────────────────────────────────────────
  // The list is the way back to a document whose message is gone: `agent_output`
  // lives in memory only, so a server restart loses the message while the file
  // it pointed at is still on disk.
  async function refreshList() {
    const panel = el('doc-viewer');
    const box = el('doc-viewer-items');
    const count = el('doc-viewer-count');
    // Nothing is looking at the list while the panel is shut, and every
    // message would otherwise cost a round trip to redraw something invisible.
    if (!box || !panel || panel.classList.contains('hidden')) return;
    let items = [];
    try {
      const data = await apiJson(sessionApi('/artifacts'));
      items = (data.artifacts || []).filter(
        a => a.href && String(a.source_kind || '').startsWith('doc:')
      );
    } catch (error) {
      console.warn('session documents not listed:', error);
      return;
    }
    if (count) count.textContent = items.length ? String(items.length) : '';
    if (!items.length) {
      box.innerHTML = `<li>${notice('doc.empty', 'No documents yet.')}</li>`;
      return;
    }
    box.innerHTML = items.map(item => {
      const title = item.label || item.name || item.artifact_id;
      const active = item.artifact_id === shownId;
      return `
        <li>
          <button type="button" onclick="openDocument('${escJs(item.artifact_id)}', '${escJs(title)}')"
            class="w-full text-left flex items-baseline gap-2 px-1.5 py-1 rounded transition-colors
                   ${active ? 'bg-primary/10 text-on-surface' : 'text-on-surface-variant hover:bg-surface-container-high/50'}">
            <span class="text-[11px] leading-snug flex-1 min-w-0 break-words">${escHtml(title)}</span>
            ${item.agent ? `<span class="text-[10px] text-outline-variant shrink-0" title="${escHtml(item.agent)}">${escHtml((window.StatusIndicator && StatusIndicator.agentName) ? (StatusIndicator.agentName(item.agent) || item.agent) : item.agent)}</span>` : ''}
          </button>
        </li>`;
    }).join('');
  }

  // A session switch invalidates everything: different store, different ids.
  function resetDocuments() {
    cache.clear();
    shownId = null;
    widthBefore = null;
    const panel = el('doc-viewer');
    if (panel) panel.classList.add('hidden');
    const box = el('doc-viewer-items');
    if (box) box.innerHTML = '';
    const count = el('doc-viewer-count');
    if (count) count.textContent = '';
  }

  // A review is a question with a document behind it, so the document opens
  // itself — being asked to approve something is exactly the moment to read it.
  // `auto` marks it as not the reader's own choice: answering closes what the
  // request opened, and leaves alone anything opened by hand.
  let openedByRequest = null;

  function openForRequest(requestId, document_) {
    if (!document_ || !document_.artifact_id) return;
    openedByRequest = requestId || null;
    openDocument(document_.artifact_id, document_.title, true);
  }

  function closeForRequest(requestId) {
    if (!openedByRequest || (requestId && requestId !== openedByRequest)) return;
    openedByRequest = null;
    closeDocument();
  }

  window.openDocument = openDocument;
  window.closeDocument = closeDocument;
  window.openDocumentForRequest = openForRequest;
  window.closeDocumentForRequest = closeForRequest;
  window.refreshSessionDocuments = refreshList;
  window.resetDocuments = resetDocuments;
})();

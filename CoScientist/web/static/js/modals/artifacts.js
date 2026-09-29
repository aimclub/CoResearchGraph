// =========================================================================
// Sandbox Artifacts panel — what the run actually left behind
// =========================================================================
// The sandbox agent publishes what it chooses to publish. Everything else it
// made sat on a machine only the sandbox could read, and the person watching
// the run had no way to look. This lists the workspace and pulls a path across
// on demand — the same transfer the agents use, so a file fetched here and a
// file fetched by an agent are one object under one link.

    const SANDBOX_ROOT = '/workspace';
    let artifactsPath = SANDBOX_ROOT;
    // Which workspace is being read: '' means the one bound to this session,
    // anything else is a task id the sandbox still holds. The sandbox addresses
    // containers by task id, so a neighbouring run is one parameter away.
    let artifactsWorkspace = '';
    // Fetched during this visit: workspace + path → durable link, so a second
    // click opens the file instead of copying it across again.
    const artifactsFetched = new Map();

    function fetchedKey(path) {
      return `${artifactsWorkspace}\u0000${path}`;
    }

    async function openArtifactsModal() {
      if (typeof toggleAttachMenu === 'function') toggleAttachMenu(false);
      document.getElementById('artifacts-modal').classList.remove('hidden');
      // The list first, then the listing: which workspace to read comes out of
      // the list, and firing both at once read the session's binding — which,
      // for a session whose sandbox was started elsewhere, is nothing at all.
      await loadWorkspaces();
      await loadArtifacts(SANDBOX_ROOT);
    }

    // What the workspace list says about a task, in the words a reader uses.
    const WORKSPACE_STATUS = {
      running: 'работает', cooldown: 'остывает', soft_stopping: 'останавливается',
      queued: 'в очереди', completed: 'завершён', failed: 'упал',
      stopped: 'остановлен', error: 'ошибка',
    };

    function workspaceLabel(workspace) {
      const id = String(workspace.sandbox_id || '');
      const status = WORKSPACE_STATUS[workspace.status] || workspace.status || '';
      // The prompt arrives as the markdown the agent was handed; its heading
      // marks say nothing here and eat the width the id and status need.
      const task = String(workspace.task || '')
        .replace(/[#*`>]/g, ' ').replace(/\s+/g, ' ').trim();
      const head = workspace.current ? `текущая (${id.slice(0, 8)})` : id.slice(0, 8);
      const tail = task ? ` — ${task.slice(0, 40)}` : '';
      return `${head} · ${status}${tail}`;
    }

    async function loadWorkspaces() {
      const select = document.getElementById('artifacts-workspace');
      const base = artifactsSession();
      if (!base) { select.innerHTML = ''; return; }
      select.innerHTML = '<option value="">песочница этой сессии</option>';

      let payload;
      try {
        const response = await fetch(`${base}/tasks`, { cache: 'no-store' });
        payload = await response.json();
      } catch (error) {
        return;  // The listing below reports the real trouble; one message is enough.
      }
      if (payload.status !== 'ok') return;

      const workspaces = payload.workspaces || [];
      if (!workspaces.length) return;

      // Read something on open rather than nothing: this session's own sandbox
      // if it has one, otherwise whatever the sandbox says is still openable.
      // Nothing is "the session's sandbox" implicitly any more — the request
      // always names a task id, because the session may not have a binding.
      if (!workspaces.some(w => w.sandbox_id === artifactsWorkspace)) {
        const chosen = workspaces.find(w => w.current && w.browsable)
          || workspaces.find(w => w.browsable) || workspaces[0];
        artifactsWorkspace = chosen.sandbox_id;
      }

      select.innerHTML = workspaces.map(workspace => {
        const id = escAttr(workspace.sandbox_id);
        const selected = workspace.sandbox_id === artifactsWorkspace ? ' selected' : '';
        // A container the sandbox has already torn down cannot be listed, but
        // saying so is the sandbox's call, not a guess made here: the row stays
        // selectable and the refusal, when it comes, is shown in full.
        const note = workspace.browsable ? '' : ' · контейнер остановлен';
        return `<option value="${id}"${selected}>`
             + `${escHtml(workspaceLabel(workspace))}${note}</option>`;
      }).join('');
    }

    function pickWorkspace(sandboxId) {
      artifactsWorkspace = sandboxId || '';
      loadArtifacts(SANDBOX_ROOT);
    }

    function closeArtifactsModal() {
      document.getElementById('artifacts-modal').classList.add('hidden');
      // Reopening should land on the listing, not on whatever was last read.
      if (artifactsViewing) closeArtifactFile();
    }

    function artifactsSession() {
      if (!activeUser || !activeSession) return null;
      return `/api/users/${encodeURIComponent(activeUser.id)}`
           + `/sessions/${encodeURIComponent(activeSession.id)}/sandbox`;
    }

    function artifactsNote(message, kind) {
      const node = document.getElementById('artifacts-note');
      node.textContent = message || '';
      node.className = 'text-[11px] mt-3 ' + (
        kind === 'error' ? 'text-error' : 'text-outline-variant');
      node.classList.toggle('hidden', !message);
    }

    function humanSize(bytes) {
      const n = Number(bytes);
      if (!Number.isFinite(n) || n < 0) return '';
      if (n < 1024) return `${n} Б`;
      if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} КБ`;
      if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} МБ`;
      return `${(n / 1024 ** 3).toFixed(2)} ГБ`;
    }

    async function loadArtifacts(path) {
      const base = artifactsSession();
      const body = document.getElementById('artifacts-body');
      if (!base) {
        body.innerHTML = '';
        artifactsNote('Откройте сессию — песочница принадлежит ей.', 'error');
        return;
      }
      artifactsPath = path || SANDBOX_ROOT;
      document.getElementById('artifacts-path').textContent = artifactsPath;
      body.innerHTML = '<div class="text-[11px] text-outline-variant py-6 text-center">Смотрю…</div>';
      artifactsNote('');

      let payload;
      try {
        const query = `path=${encodeURIComponent(artifactsPath)}`
          + (artifactsWorkspace ? `&sandbox_id=${encodeURIComponent(artifactsWorkspace)}` : '');
        const response = await fetch(`${base}/files?${query}`, { cache: 'no-store' });
        payload = await response.json();
      } catch (error) {
        body.innerHTML = '';
        artifactsNote('Песочница не ответила: ' + error, 'error');
        return;
      }
      if (payload.status !== 'ok') {
        body.innerHTML = '';
        // The usual causes are honest ones — no sandbox started for this
        // session, or a container already torn down — so say which, rather
        // than "failed".
        const message = String(payload.message || 'Песочница недоступна.');
        artifactsNote(/\b409\b/.test(message)
          ? 'Контейнер этой задачи уже остановлен — файлы из него недоступны.'
          : message, 'error');
        return;
      }
      renderArtifacts(payload.entries || []);
    }

    // The listing last drawn, so a redraw after a fetch shows the same folder.
    let lastArtifactEntries = [];

    // What each kind of file is worth showing as. The server decides the kind
    // from the name and serves the bytes with a matching content type, so this
    // only chooses the element to put them in.
    const KIND_ICONS = {
      dir: 'folder', text: 'description', image: 'image',
      pdf: 'picture_as_pdf', binary: 'deployed_code_alert',
    };

    function renderArtifacts(entries) {
      if (entries) lastArtifactEntries = entries;
      entries = lastArtifactEntries;
      const body = document.getElementById('artifacts-body');
      const up = artifactsPath !== SANDBOX_ROOT
        ? `<div class="art-row" onclick="loadArtifacts('${escAttr(parentPath(artifactsPath))}')">
             <span class="material-symbols-outlined text-sm text-outline-variant">arrow_upward</span>
             <span class="art-name">..</span></div>`
        : '';

      if (!entries.length) {
        body.innerHTML = up + '<div class="text-[11px] text-outline-variant py-6 text-center">Пусто.</div>';
        return;
      }
      // Folders first, then files, each alphabetically: a workspace after a
      // training run is a hundred checkpoints and three things you want.
      const sorted = entries.slice().sort((a, b) => {
        const da = isDirEntry(a), db = isDirEntry(b);
        if (da !== db) return da ? -1 : 1;
        return String(a.name || '').localeCompare(String(b.name || ''));
      });

      body.innerHTML = up + sorted.map(entry => {
        const isDir = isDirEntry(entry);
        const full = entry.path || `${artifactsPath}/${entry.name}`;
        const kind = isDir ? 'dir' : (entry.kind || 'binary');
        const link = artifactsFetched.get(fetchedKey(full));
        const icon = KIND_ICONS[kind] || 'description';
        // Three different wants, three actions. Reading and saving go straight
        // through us; only the durable link needs storage, because only it has
        // to outlive the container.
        const look = (kind === 'binary' || isDir) ? ''
          : `<button class="art-get" onclick="event.stopPropagation();openArtifactFile('${escAttr(full)}')">смотреть</button>`;
        const save = `<a class="art-get" download onclick="event.stopPropagation()"
             href="${escAttr(viewUrl(full, true, isDir))}">скачать</a>`;
        const share = link
          ? `<a class="art-open" href="${escAttr(link)}" target="_blank" rel="noopener">ссылка ↗</a>`
          : `<button class="art-get art-share" onclick="event.stopPropagation();fetchArtifact('${escAttr(full)}', this)"
               title="Скопировать в хранилище и получить ссылку, которую можно вставить в отчёт">ссылку</button>`;
        const onclick = isDir
          ? ` onclick="loadArtifacts('${escAttr(full)}')"`
          : (kind === 'binary' ? '' : ` onclick="openArtifactFile('${escAttr(full)}')"`);
        return `<div class="art-row"${onclick}>
            <span class="material-symbols-outlined text-sm text-outline-variant">${icon}</span>
            <span class="art-name">${escHtml(entry.name || '')}</span>
            <span class="art-size">${isDir ? '' : humanSize(entry.size)}</span>
            ${look}
            ${save}
            ${share}
          </div>`;
      }).join('');
    }

    // The sandbox says "dir"; earlier drafts of this panel looked for
    // "directory" and every folder in the workspace came out a file nobody
    // could open. Accept both and be done with it.
    function isDirEntry(entry) {
      const type = String(entry && entry.type || '').toLowerCase();
      return type === 'dir' || type === 'directory';
    }

    function parentPath(path) {
      const trimmed = String(path || '').replace(/\/+$/, '');
      const cut = trimmed.lastIndexOf('/');
      return cut > 0 ? trimmed.slice(0, cut) : SANDBOX_ROOT;
    }

    async function fetchArtifact(path, button) {
      const base = artifactsSession();
      if (!base) return;
      button.disabled = true;
      button.textContent = 'беру…';
      artifactsNote('');
      try {
        const response = await fetch(`${base}/fetch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path, sandbox_id: artifactsWorkspace || null }),
        });
        const result = await response.json();
        if (result.status !== 'success') {
          button.disabled = false;
          button.textContent = 'ссылку';
          artifactsNote(result.message || 'Скопировать не удалось.', 'error');
          return;
        }
        artifactsFetched.set(fetchedKey(path), result.url);
        // Redraw rather than patch the row: the button becomes a link, and the
        // next visit to this folder should already show it as fetched.
        renderArtifacts();
        artifactsNote(`Ссылка на ${path} готова (${humanSize(result.size_bytes)}) и не протухает — её можно вставить в отчёт.`);
      } catch (error) {
        button.disabled = false;
        button.textContent = 'ссылку';
        artifactsNote('Скопировать не удалось: ' + error, 'error');
      }
    }

    function escAttr(value) {
      return String(value == null ? '' : value)
        .replace(/&/g, '&amp;').replace(/'/g, '&#39;').replace(/"/g, '&quot;');
    }

// =========================================================================
// Looking at one file
// =========================================================================
// A workspace is read far more often than it is harvested. The transfer
// through storage answers "I want a link to this in the report"; this answers
// "what is in it?", which is the commoner question and should not need a
// bucket, a signature or a round trip through S3 to get an answer.

    // The file currently open, so the viewer survives a redraw and «назад»
    // knows which folder to return to.
    let artifactsViewing = null;

    function viewUrl(path, download, isDir) {
      const base = artifactsSession();
      return `${base}/view?path=${encodeURIComponent(path)}`
        + (artifactsWorkspace ? `&sandbox_id=${encodeURIComponent(artifactsWorkspace)}` : '')
        + (download ? '&download=1' : '')
        // The sandbox serves a directory as a ZIP; saying so up front is what
        // gives the saved file a name the operating system understands.
        + (isDir ? '&dir=1' : '');
    }

    function closeArtifactFile() {
      artifactsViewing = null;
      document.getElementById('artifacts-view').classList.add('hidden');
      document.getElementById('artifacts-browse').classList.remove('hidden');
    }

    async function openArtifactFile(path) {
      const base = artifactsSession();
      if (!base) return;
      artifactsViewing = path;
      const name = String(path).split('/').pop();
      document.getElementById('artifacts-browse').classList.add('hidden');
      document.getElementById('artifacts-view').classList.remove('hidden');
      document.getElementById('artifacts-view-name').textContent = name;
      document.getElementById('artifacts-view-save').href = viewUrl(path, true);
      const body = document.getElementById('artifacts-view-body');
      body.innerHTML = '<div class="text-[11px] text-outline-variant py-6 text-center">Читаю…</div>';

      // An image or a PDF is handed to the browser by URL — fetching the bytes
      // here would only be to hand them back.
      const kind = kindOfName(name);
      if (kind === 'image') {
        body.innerHTML = `<img class="art-image" src="${escAttr(viewUrl(path))}" alt="${escAttr(name)}">`;
        return;
      }
      if (kind === 'pdf') {
        body.innerHTML = `<iframe class="art-pdf" src="${escAttr(viewUrl(path))}"></iframe>`
          + `<div class="text-[11px] mt-2"><a class="art-open" target="_blank" rel="noopener"
               href="${escAttr(viewUrl(path))}">открыть в новой вкладке ↗</a></div>`;
        return;
      }

      let response;
      try {
        response = await fetch(viewUrl(path), { cache: 'no-store' });
      } catch (error) {
        body.innerHTML = `<div class="text-[11px] text-error py-6 text-center">Не прочиталось: ${escHtml(String(error))}</div>`;
        return;
      }
      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try { detail = (await response.json()).detail || detail; } catch (e) { /* not JSON */ }
        body.innerHTML = `<div class="text-[11px] text-error py-6 text-center">${escHtml(detail)}</div>`;
        return;
      }
      const text = await response.text();
      const cut = response.headers.get('X-Preview-Truncated') === '1';
      body.innerHTML = renderFileBody(name, text)
        + (cut ? '<div class="text-[11px] text-outline-variant mt-2">Показано начало файла — он длиннее.</div>' : '');
    }

    // Kept in step with CoScientist/web/preview.py, which decides the same
    // thing server-side; this copy only picks the element.
    const VIEW_IMAGE = /\.(png|jpe?g|gif|webp|bmp|ico|svg|avif|tiff?)$/i;

    function kindOfName(name) {
      if (VIEW_IMAGE.test(name)) return 'image';
      if (/\.pdf$/i.test(name)) return 'pdf';
      return 'text';
    }

    function renderFileBody(name, text) {
      // Markdown is what reports and AGENTS.md are written in, and reading it
      // raw is reading the syntax instead of the text.
      if (/\.(md|markdown)$/i.test(name) && typeof marked !== 'undefined') {
        return `<div class="md-body art-text-rendered">${
          DOMPurify.sanitize(marked.parse(text), { ADD_ATTR: ['target'] })}</div>`;
      }
      if (/\.(json|ipynb|jsonl?)$/i.test(name)) {
        try {
          return `<pre class="art-text">${escHtml(JSON.stringify(JSON.parse(text), null, 2))}</pre>`;
        } catch (error) { /* not valid JSON — show it as written */ }
      }
      if (/\.(csv|tsv)$/i.test(name)) return renderTable(name, text);
      return `<pre class="art-text">${escHtml(text)}</pre>`;
    }

    // A results CSV is the point of most runs; a wall of commas is not how to
    // read one. Only the head is laid out — the rest stays as text below.
    const TABLE_ROWS = 200;

    function renderTable(name, text) {
      const sep = /\.tsv$/i.test(name) ? '\t' : ',';
      const lines = text.split(/\r?\n/).filter(line => line.length);
      if (lines.length < 2) return `<pre class="art-text">${escHtml(text)}</pre>`;
      const rows = lines.slice(0, TABLE_ROWS).map(line => line.split(sep));
      const head = rows.shift();
      const more = lines.length > TABLE_ROWS
        ? `<div class="text-[11px] text-outline-variant mt-2">Показаны первые ${TABLE_ROWS} строк из ${lines.length}.</div>`
        : '';
      return `<div class="art-table-wrap"><table class="art-table">
          <thead><tr>${head.map(cell => `<th>${escHtml(cell)}</th>`).join('')}</tr></thead>
          <tbody>${rows.map(row =>
            `<tr>${row.map(cell => `<td>${escHtml(cell)}</td>`).join('')}</tr>`).join('')}</tbody>
        </table></div>${more}`;
    }

// =========================================================================
// Dataset Modal & Sandbox Upload Widget
// =========================================================================
    function openDatasetModal() {
      toggleAttachMenu(false);
      const input = document.getElementById('dataset-url-input');
      input.value = datasetUrl;
      showDatasetError('');
      document.getElementById('dataset-remove-btn').classList.toggle('hidden', !datasetUrl);
      document.getElementById('dataset-modal').classList.remove('hidden');
      input.focus();
      input.select();
    }

    function closeDatasetModal() {
      document.getElementById('dataset-modal').classList.add('hidden');
    }

    function showDatasetError(message) {
      const node = document.getElementById('dataset-error');
      node.textContent = message || '';
      node.classList.toggle('hidden', !message);
    }

    function datasetUrlError(raw) {
      // Mirrors the server-side check, so a typo is caught before it travels.
      const url = String(raw || '').trim();
      if (!url) return 'Enter a link to a .zip archive.';
      let parsed;
      try {
        parsed = new URL(url);
      } catch (error) {
        return 'That is not a valid URL.';
      }
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        return 'The link must be an http(s) URL.';
      }
      if (!parsed.pathname.toLowerCase().endsWith('.zip')) {
        return 'The link must point to a .zip archive.';
      }
      return '';
    }

    function sendDatasetUrl(url) {
      if (!ws || ws.readyState !== 1) {
        showDatasetError('Not connected — reconnect and try again.');
        return false;
      }
      ws.send(JSON.stringify({ type: 'set_dataset_url', dataset_url: url }));
      return true;
    }

    function saveDatasetLink() {
      const url = document.getElementById('dataset-url-input').value.trim();
      const error = datasetUrlError(url);
      if (error) {
        showDatasetError(error);
        return;
      }
      if (sendDatasetUrl(url)) closeDatasetModal();
    }

    function clearDatasetLink() {
      if (sendDatasetUrl('')) closeDatasetModal();
    }

    // =========================================================================
    // Sandbox Dataset Upload Progress Widget
    // =========================================================================
    let datasetLogEventSource = null;
    let activeUploadData = null;

    function updateDatasetUploadUI(d) {
      if (!d) return;
      activeUploadData = d;
      const widget = document.getElementById('dataset-upload-widget');
      const arc = document.getElementById('dataset-upload-arc');
      const pctEl = document.getElementById('dataset-upload-pct');
      const textEl = document.getElementById('dataset-upload-text');

      const p = d.progress || {};
      const total = Number(p.total_mb) || 0;
      const done = Number(p.downloaded_mb) || 0;
      let pctRaw = p.percent != null ? Number(p.percent) : (total > 0 ? (done / total) * 100 : 0);
      const pct = Math.min(100, Math.max(0, isFinite(pctRaw) ? pctRaw : 0));

      const circumference = 113.097; // 2 * PI * 18
      const dashoffset = circumference * (1 - pct / 100);

      if (arc) arc.style.strokeDashoffset = dashoffset;
      if (pctEl) pctEl.textContent = Math.round(pct) + '%';

      if (textEl) {
        if (total > 0) {
          textEl.textContent = `Uploading: ${formatDatasetSize(done)} / ${formatDatasetSize(total)}`;
        } else if (done > 0) {
          textEl.textContent = `Uploading: ${formatDatasetSize(done)}`;
        } else if (d.filename) {
          textEl.textContent = `Uploading: ${d.filename}`;
        } else {
          textEl.textContent = `Uploading dataset to sandbox...`;
        }
      }

      const filenameEl = document.getElementById('dataset-upload-filename');
      const speedEl = document.getElementById('dataset-upload-speed');
      const etaEl = document.getElementById('dataset-upload-eta');
      const statusEl = document.getElementById('dataset-upload-status');

      if (filenameEl) filenameEl.textContent = d.filename || d.download_id || '—';
      if (speedEl) speedEl.textContent = p.speed_mb_s ? `${Number(p.speed_mb_s).toFixed(1)} MB/s` : '—';
      if (etaEl) etaEl.textContent = p.eta_seconds ? `${Math.round(p.eta_seconds)}s` : '—';
      if (statusEl) statusEl.textContent = d.status || p.status || 'in_progress';

      if (widget) widget.classList.remove('hidden');
    }

    async function cancelDatasetUpload() {
      const downloadId = activeUploadData ? activeUploadData.download_id : null;
      const userId = activeUser ? activeUser.id : null;
      const sessionId = activeSession ? activeSession.id : null;
      try {
        await fetch('/api/v1/downloads/cancel', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            download_id: downloadId,
            user_id: userId,
            session_id: sessionId
          })
        });
      } catch (err) {
        console.warn('Failed to send download cancel request:', err);
      }
      const widget = document.getElementById('dataset-upload-widget');
      if (widget) widget.classList.add('hidden');
    }

    function toggleDatasetUploadDetails() {
      const drawer = document.getElementById('dataset-upload-details');
      if (drawer) drawer.classList.toggle('hidden');
    }

    function connectDatasetLogsSSE() {
      if (datasetLogEventSource) return;
      try {
        const url = new URL('/api/downloads/logs', location.href).toString();
        datasetLogEventSource = new EventSource(url);
        datasetLogEventSource.addEventListener('download', (e) => {
          try {
            const d = JSON.parse(e.data);
            if (d) updateDatasetUploadUI(d);
          } catch (_) { }
        });
        datasetLogEventSource.addEventListener('status', (e) => {
          if (String(e.data).trim() === 'idle') {
            // Stream idle
          }
        });
        datasetLogEventSource.onerror = () => {
          // EventSource automatically retries
        };
      } catch (err) {
        console.warn('Dataset logs SSE connection error:', err);
      }
    }


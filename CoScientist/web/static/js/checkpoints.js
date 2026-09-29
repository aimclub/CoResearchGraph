let checkpointSignature = null;

async function refreshCheckpoints() {
  const list = document.getElementById('checkpoint-list');
  if (!list) return;
  try {
    const response = await fetch(`/api/checkpoints?_=${Date.now()}`, { cache: 'no-store' });
    if (!response.ok) {
      checkpointSignature = null;
      list.textContent = [401, 503].includes(response.status)
        ? 'Snapshot management is available through Synapse.'
        : `Failed to load snapshots (HTTP ${response.status}).`;
      return;
    }
    const body = await response.json();
    renderCheckpoints(body.checkpoints || []);
  } catch (error) {
    console.error('checkpoints refresh failed', error);
  }
}

function renderCheckpoints(checkpoints) {
  const list = document.getElementById('checkpoint-list');
  if (!list) return;
  checkpoints.sort((left, right) => (right.created_at || '').localeCompare(left.created_at || ''));
  const signature = checkpoints.map(checkpoint => checkpoint.checkpoint_id).join('|');
  if (signature === checkpointSignature) return;
  checkpointSignature = signature;
  if (!checkpoints.length) {
    list.innerHTML = '<div class="text-outline-variant text-[9px] font-mono">No snapshots yet</div>';
    return;
  }
  list.innerHTML = checkpoints.map(checkpoint => {
    const created = (checkpoint.created_at || '').slice(11, 19);
    const checkpointId = escHtml(checkpoint.checkpoint_id || '');
    return `
      <div class="bg-surface-container-high/40 rounded-md p-2 border border-outline-variant/10 flex items-center justify-between gap-2">
        <div class="min-w-0">
          <div class="text-[10px] font-bold text-on-surface-variant truncate">${escHtml(checkpoint.label || '')}</div>
          <div class="text-[8px] font-mono text-outline-variant">${escHtml(created)} · ${escHtml(checkpoint.size_kb || 0)}KB</div>
        </div>
        <button data-checkpoint-id="${checkpointId}" onclick="restoreCheckpoint(this.dataset.checkpointId)" title="Restore this checkpoint"
          class="shrink-0 flex items-center gap-1 px-2 py-1 bg-primary/15 text-primary border border-primary/30 rounded hover:bg-primary/25 transition-all text-[9px] font-bold uppercase tracking-widest">
          <span class="material-symbols-outlined text-[13px]">restore</span> Restore
        </button>
      </div>`;
  }).join('');
}

async function restoreCheckpoint(checkpointId) {
  addSystemMsg(`Restoring checkpoint …${checkpointId.slice(-8)}`);
  try {
    const response = await fetch(`/api/checkpoints/${encodeURIComponent(checkpointId)}/restore`, {
      method: 'POST',
    });
    const body = await response.json();
    if (response.ok && body.context_id) {
      addSystemMsg(
        `Checkpoint restored as context …${body.context_id.slice(-8)} ` +
        `(${body.event_count} events). ${body.resume_hint || ''}`
      );
    } else {
      const detail = body.detail
        ? (typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail))
        : `HTTP ${response.status}`;
      addSystemMsg(`Checkpoint restore rejected: ${detail}`);
    }
  } catch (error) {
    addSystemMsg(`Checkpoint restore failed: ${error}`);
  }
  refreshCheckpoints();
}

refreshCheckpoints();
setInterval(refreshCheckpoints, 5000);

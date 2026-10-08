// =========================================================================
// Application Bootstrap & Global Listeners
// =========================================================================

// Before bootstrap(), not on DOMContentLoaded: bootstrap starts restoring a
// saved session while the page is still loading, and restoring can post-render
// an agent card into a slot that does not exist until this has run.
initAgentNav();
renderActivityRail();

// Mount the live status indicator right under the chat feed.
StatusIndicator.mount(document.getElementById('status-indicator'));

applySideNavState();
bootstrap();
loadSettings();
connectDatasetLogsSSE();

// Keep-alive ping
setInterval(() => { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'ping' })); }, 30000);

// Global click-away handler for attach menu
document.addEventListener('click', (e) => {
  const menu = document.getElementById('attach-menu');
  const btn = document.getElementById('attach-btn');
  if (menu && !menu.classList.contains('hidden') && !menu.contains(e.target) && !btn.contains(e.target)) {
    menu.classList.add('hidden');
  }
});

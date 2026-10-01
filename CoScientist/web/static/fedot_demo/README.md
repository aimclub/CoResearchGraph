# Embedded FEDOT viewer

Renderer assets originate from CoScientist PR #368 (`6e18bcf4`), which vendors
`infrastructure/fedot-mas-gui/gui/static` at `v0.1.0-40-gc648a65`.

Local adaptations: `init()` is observation-only; no standalone API/key/import
controls are wired and standalone demo presets are omitted. The bottom
`_coscientistLiveConnect` bridge uses the shared
`/static/js/fedot_runs.js` picker and session-owned persisted events. It clears
graph/answer state between runs and preserves generation events when config
arrives. `index.html` adds the picker and its script.

When updating, port renderer changes selectively; do not overwrite these
adaptations or re-enable `probeBackend`/global-latest streaming. Run the FEDOT
history/tab/bridge tests and browser smoke, including successful run -> failed
before config, two sessions, MAS/MAW configs and imported session replay.

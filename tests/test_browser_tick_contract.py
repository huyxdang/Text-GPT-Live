import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class BrowserTickContractTests(unittest.TestCase):
    def test_running_browser_surfaces_do_not_suppress_empty_text_ticks(self) -> None:
        surfaces = {
            "main": ("static/app.js", "text"),
        }

        for name, (relative_path, element_name) in surfaces.items():
            with self.subTest(surface=name):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                empty_text_guard = (
                    rf"if\s*\(\s*!elements\.{element_name}\.value\.trim\(\)\s*\)\s*return"
                )
                self.assertIsNone(re.search(empty_text_guard, source))
                self.assertIn(f"text: elements.{element_name}.value", source)

    def test_fresh_session_prepares_qwen_with_a_real_tick_before_the_loop(self) -> None:
        source = (ROOT / "static/app.js").read_text(encoding="utf-8")
        begin = source.index("async function beginStreaming()")
        end = source.index("async function waitForModel()", begin)
        streaming = source[begin:end]

        self.assertIn("renderedHistory.length === 0", streaming)
        self.assertIn('setRunButton("Preparing Qwen", { loading: true })', streaming)
        self.assertLess(streaming.index("showSessionPreparing()"), streaming.index("await tick()"))
        self.assertLess(streaming.index("await tick()"), streaming.index("window.setInterval"))

    def test_meaningful_actions_use_a_temporary_presentation_stage(self) -> None:
        markup = (ROOT / "static/index.html").read_text(encoding="utf-8")
        source = (ROOT / "static/app.js").read_text(encoding="utf-8")
        styles = (ROOT / "static/styles.css").read_text(encoding="utf-8")

        self.assertIn('id="action-stage"', markup)
        self.assertIn('role="status"', markup)
        self.assertIn("const ACTION_STAGE_MS = 2800;", source)
        self.assertIn("const RESPONSE_STAGE_HOLD_MS = 3500;", source)
        self.assertIn('if (actionPanelView !== "stream" || !action?.valid || action.kind === "idle") return;', source)
        self.assertIn("stageTurnAction(turn);", source)
        self.assertIn("dismissActionStage({ resetKey: true });", source)
        self.assertIn("--action-rail:", styles)
        self.assertNotIn("inset: 0 0 0 auto;", styles)
        self.assertIn('id="action-panel"', markup)
        self.assertIn('id="action-stage-announcement"', markup)

    def test_live_participants_share_the_card_equally(self) -> None:
        markup = (ROOT / "static/index.html").read_text(encoding="utf-8")
        styles = (ROOT / "static/styles.css").read_text(encoding="utf-8")

        self.assertIn("--action-rail: 50%;", styles)
        self.assertIn("--participant-copy-color: var(--ink);", styles)
        self.assertIn("--participant-copy-size: clamp(21px, 1.7vw, 28px);", styles)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr));", styles)
        self.assertIn('<h3 class="participant-label">Human</h3>', markup)
        self.assertIn(
            '<h3 class="participant-label">Model<span class="live-dot" aria-hidden="true"></span></h3>',
            markup,
        )
        self.assertIn('aria-label="Model activity"', markup)
        self.assertNotIn("Current assistant action", markup)
        self.assertGreaterEqual(styles.count("font-size: var(--participant-copy-size);"), 3)
        self.assertGreaterEqual(styles.count("color: var(--participant-copy-color);"), 3)

    def test_responses_stream_visually_but_tools_pop_in_complete(self) -> None:
        source = (ROOT / "static/app.js").read_text(encoding="utf-8")
        styles = (ROOT / "static/styles.css").read_text(encoding="utf-8")

        self.assertIn("function streamResponseText(value, onComplete)", source)
        self.assertIn('window.matchMedia("(prefers-reduced-motion: reduce)")', source)
        self.assertIn("window.requestAnimationFrame(reveal)", source)
        self.assertIn(
            "streamResponseText(value, () => scheduleActionStageDismiss(RESPONSE_STAGE_HOLD_MS));",
            source,
        )
        self.assertIn("stream: true", source)
        self.assertIn('actionPanelView = view === "model-actions" ? "model-actions" : "stream";', source)
        self.assertIn('setActionPanelView("model-actions")', source)
        self.assertIn(
            "showActionStage({ key, state: name, value: `${verb}: ${payload || `Using ${name}`}` });",
            source,
        )
        self.assertIn('label: "Searching the web"', source)
        self.assertIn('state: "web_search"', source)
        self.assertIn("response-cursor", styles)
        self.assertIn(
            '.action-stage[data-action="respond"].is-streaming .action-stage__value::after',
            styles,
        )
        self.assertIn(".action-stage.is-dismissing .action-stage__content", styles)
        self.assertIn('.action-stage[data-action="web_search"] .action-stage__label', styles)
        self.assertIn('body[data-action-panel="model-actions"] .action-log ul', styles)

    def test_pending_search_does_not_hide_a_newer_model_response(self) -> None:
        source = (ROOT / "static/app.js").read_text(encoding="utf-8")
        begin = source.index("function renderCurrentAction(history, pendingSearches)")
        end = source.index("function renderSession(session)", begin)
        renderer = source[begin:end]

        self.assertIn("const turn = history.at(-1);", renderer)
        self.assertNotIn("if (pendingSearches > 0)", renderer)
        self.assertIn("stageTurnAction(turn);", renderer)

    def test_tool_actions_share_model_type_but_use_semantic_colors(self) -> None:
        styles = (ROOT / "static/styles.css").read_text(encoding="utf-8")
        source = (ROOT / "static/app.js").read_text(encoding="utf-8")

        self.assertIn("--tool-action: #2457ff;", styles)
        self.assertIn("--delegate-action: #7b4bc4;", styles)
        self.assertIn('data-action="highlight"', styles)
        self.assertIn('.translation-output p', styles)
        self.assertIn("color: var(--tool-action);", styles)
        self.assertIn('data-action="delegate"', styles)
        self.assertIn("color: var(--delegate-action);", styles)
        self.assertIn("font-size: var(--participant-copy-size);", styles)
        self.assertIn('item.classList.add("is-tool")', source)
        self.assertIn('item.classList.add("is-delegate")', source)

    def test_translation_uses_a_continuous_model_channel_not_a_tool_popup(self) -> None:
        markup = (ROOT / "static/index.html").read_text(encoding="utf-8")
        source = (ROOT / "static/app.js").read_text(encoding="utf-8")
        styles = (ROOT / "static/styles.css").read_text(encoding="utf-8")

        self.assertIn('id="translation-output"', markup)
        self.assertNotIn('Chinese translation', markup)
        self.assertIn('if (name === "translate_commit") return;', source)
        self.assertIn("function renderTranslation(commits = [])", source)
        self.assertIn("TRANSLATION_APPEND_MS_PER_CHARACTER", source)
        self.assertIn("renderTranslation(session.translation_commits || []);", source)
        self.assertNotIn('translate_commit: ["Live translation"', source)
        self.assertIn('.translation-output[data-state="committing"] p::after', styles)
        self.assertIn("font-size: var(--participant-copy-size);", styles)

    def test_concurrent_work_opens_an_external_workbench_with_progressive_ui(self) -> None:
        markup = (ROOT / "static/index.html").read_text(encoding="utf-8")
        source = (ROOT / "static/app.js").read_text(encoding="utf-8")
        styles = (ROOT / "static/styles.css").read_text(encoding="utf-8")

        self.assertIn('id="work-surface"', markup)
        self.assertIn('id="live-stage"', markup)
        self.assertIn('id="interaction-card"', markup)
        self.assertIn('id="external-workbench"', markup)
        model_pane_end = markup.index("</aside>", markup.index('id="action-panel"'))
        self.assertGreater(markup.index('id="work-surface"'), model_pane_end)
        self.assertNotIn('id="job-deck"', markup)
        self.assertIn("function renderWorkSurface(history, sessionJobs = [])", source)
        self.assertIn("function syncUIPanel(root, spec)", source)
        self.assertIn("function morphWorkThread(entry, update", source)
        self.assertIn("function animateWorkLayout(beforeLayout)", source)
        self.assertIn("captureWorkLayout()", source)
        self.assertIn('window.matchMedia("(prefers-reduced-motion: reduce)").matches', source)
        self.assertIn("entry.body.replaceChildren(results)", source)
        self.assertIn("const SEARCH_RESULT_HOLD_MS = 3500;", source)
        self.assertIn("window.setTimeout(() => retireWorkThread(id), SEARCH_RESULT_HOLD_MS);", source)
        self.assertNotIn("lastSpokeAt", source)
        self.assertIn("components.slice(rendered)", source)
        self.assertIn("renderWorkSurface(history, session.jobs || []);", source)
        self.assertIn(".work-surface", styles)
        self.assertIn('.work-surface[data-job-count="2"]', styles)
        self.assertIn("grid-template-rows: minmax(0, 3fr) minmax(0, 2fr);", styles)
        self.assertIn(".work-thread__body", styles)
        self.assertIn("overscroll-behavior: contain;", styles)
        self.assertIn('more.className = "job-search__more";', source)
        self.assertIn('more.setAttribute("aria-expanded", "false");', source)
        self.assertIn("item.hidden = expanded;", source)
        self.assertIn("-webkit-line-clamp: 2;", styles)
        self.assertIn(".live-stage.has-work", styles)
        self.assertIn(".external-workbench", styles)
        self.assertIn('elements.liveStage?.classList.toggle("has-work", count > 0);', source)
        self.assertIn('elements.externalWorkbench?.setAttribute("aria-hidden", String(count === 0));', source)
        self.assertIn("grid-template-columns: minmax(0, 3fr) minmax(360px, 2fr);", styles)
        self.assertIn("grid-template-columns: minmax(0, 1fr);", styles)
        self.assertIn("@media (max-width: 1100px)", styles)
        self.assertIn("clip-path: inset(0", styles)
        self.assertNotIn(".work-thread__build-state", styles)
        self.assertNotIn("Adding the next component", source)
        self.assertNotIn('"In motion"', source)
        self.assertIn(".work-thread__shimmer", styles)
        self.assertIn("#e7eaf0", styles)
        self.assertNotIn("body[data-deck-open", styles)
        self.assertIn("workThreads.clear();", source)
        self.assertIn("elements.workSurface.replaceChildren();", source)
        self.assertNotIn("dismissedJobs.clear();", source)
        self.assertNotIn("jobWindows.clear();", source)


if __name__ == "__main__":
    unittest.main()

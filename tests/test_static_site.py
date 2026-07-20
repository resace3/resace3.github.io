from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
STATIC_PAGES = [
    ROOT / "index.html",
    ROOT / "new-projects.html",
    ROOT / "causal-dag.html",
    ROOT / "agentic-prompt.html",
    ROOT / "activity-health-demo.html",
]


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "link" and attributes.get("rel") == "stylesheet":
            self.assets.append(attributes.get("href", ""))
        if tag == "script" and attributes.get("src"):
            self.assets.append(attributes["src"])
        if tag == "a" and attributes.get("href"):
            self.links.append(attributes["href"])


def resolve_asset(page, asset):
    parsed = urlsplit(asset)
    if parsed.scheme or parsed.netloc:
        return None
    path = parsed.path
    if not path:
        return None
    if path.startswith("/"):
        return ROOT / path.lstrip("/")
    return page.parent / path


def test_static_pages_exist_and_reference_existing_assets():
    for page in STATIC_PAGES:
        assert page.exists(), f"{page.name} should exist"
        parser = AssetParser()
        parser.feed(page.read_text(encoding="utf-8"))

        for asset in parser.assets:
            resolved = resolve_asset(page, asset)
            if resolved is None:
                continue
            assert resolved.exists(), f"{page.name} references missing asset {asset}"


def test_static_pages_reference_existing_internal_links():
    for page in STATIC_PAGES:
        parser = AssetParser()
        parser.feed(page.read_text(encoding="utf-8"))

        for link in parser.links:
            resolved = resolve_asset(page, link)
            if resolved is None:
                continue
            assert resolved.exists(), f"{page.name} links to missing page {link}"


def test_activity_health_demo_has_synthetic_data_disclosure_and_local_assets():
    html = (ROOT / "activity-health-demo.html").read_text(encoding="utf-8")

    assert "Synthetic people and virtual step data only" in html
    assert "Nothing is sent or stored" in html
    assert 'src="activity-health-demo.js?v=20260719.1"' in html
    assert 'href="activity-health-demo.css?v=20260719.1"' in html
    assert "Day comparison" not in html


def test_new_projects_decorations_and_mobile_header_stay_in_viewport():
    html = (ROOT / "new-projects.html").read_text(encoding="utf-8")
    css = (ROOT / "style.css").read_text(encoding="utf-8")

    assert 'href="style.css?v=20260720.2"' in html

    orb_rule = css.split(".new-projects-hero::before {", 1)[1].split("}", 1)[0]
    assert "right: 0;" in orb_rule

    mobile_rules = css.split("@media (max-width: 680px) {", 1)[1]
    assert "grid-template-columns: 1fr;" in mobile_rules
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in mobile_rules
    assert ".brand-group" in mobile_rules
    assert ".new-projects-badge" in mobile_rules


def test_static_causal_dag_snapshot_intercepts_backend_histogram_calls():
    html = (ROOT / "causal-dag.html").read_text(encoding="utf-8")

    assert "window.personalWebsiteStaticDag = true" in html
    assert 'url.pathname === "/dag/node-histogram"' in html
    assert 'url.pathname === "/dag/autosave" || url.pathname === "/settings/autosave"' in html
    assert 'url.pathname === "/entities/search"' in html
    assert 'url.pathname === "/run"' in html
    assert "input instanceof URL" in html
    assert "document.documentElement.outerHTML" not in html


def test_static_causal_dag_snapshot_contains_analysis_result_payload():
    html = (ROOT / "causal-dag.html").read_text(encoding="utf-8")

    assert "staticAnalysisResultsDocument" in html
    assert 'data-analysis-results' in html
    assert "PANAS Score" in html
    assert "PANAS Score below 33" in html
    assert "Simulated daily PANAS summed score on a 10 to 50 scale" in html
    assert "value=\"33\"" in html
    assert "5.7" not in html
    assert "Mood score" not in html
    assert "Clinician summary" in html
    assert "Recommended: G-formula / outcome model" in html
    assert "Default (recommended): G-formula / outcome model" not in html
    assert "Already selected for this demo. Change only if you want a specific causal estimator." in html
    assert "Choose from DAG" not in html
    assert "PANAS Score (t-1)" in html
    assert "Sleep may contribute to current PANAS Score changes" in html
    assert "What the DAG says may cause what" in html
    assert "Prior PANAS Score may affect Sleep" in html
    assert "Prior PANAS Score may affect Steps" in html
    assert "Steps may contribute to current PANAS Score changes" in html
    assert "Show plots and technical details" in html
    assert "95% CI -0.54 to 1.99" in html
    assert "Predicted PANAS Score" in html
    assert "Estimated PANAS Score difference" in html
    assert "Advanced results" not in html
    assert "Effect estimate distribution" not in html
    assert "effect distribution" not in html
    assert "Sensor not found" not in html
    assert "Home Assistant" not in html


def test_static_causal_dag_sidebar_advanced_toggle_contains_aggregate_controls():
    html = (ROOT / "causal-dag.html").read_text(encoding="utf-8")

    assert "Simulated variable" in html
    assert '<details class="sidebar-advanced-settings">' in html
    assert "<summary>Advanced</summary>" in html
    assert html.index("<summary>Advanced</summary>") < html.index("Daily aggregate")
    assert html.index("Daily aggregate") < html.index("From time") < html.index("To time")
    assert "binary_sensor.pantry_door_window" not in html
    assert "sensor.nutribullet_plug_current" not in html


def test_static_causal_dag_snapshot_owns_static_analysis_submit_flow():
    html = (ROOT / "causal-dag.html").read_text(encoding="utf-8")

    assert "runStaticAnalysisInPage" in html
    assert 'document.addEventListener("submit", handleStaticAnalysisEvent, true)' in html
    assert 'document.addEventListener("click", handleStaticAnalysisEvent, true)' in html
    assert "event.stopImmediatePropagation()" in html
    assert "Running analysis" in html
    assert "Saving DAG" not in html


def test_static_causal_dag_snapshot_places_guide_button_before_title():
    html = (ROOT / "causal-dag.html").read_text(encoding="utf-8")

    guide_index = html.index('class="tab guide-tab brand-guide-tab"')
    title_index = html.index("<strong>N of 1 Causal Analysis Engine</strong>")

    assert guide_index < title_index
    assert '<span class="brand-mark">CAE</span>' not in html
    assert '<nav class="tabs" aria-label="Primary">' not in html
    assert ">Analysis</a>" not in html
    assert "Exposure (Independent)" in html
    assert "Outcome (Dependent)" in html
    assert "Independent Variable" not in html
    assert "Dependent Variable" not in html
    assert 'data-onboarding-skip>Skip "Guide Me"</button>' in html

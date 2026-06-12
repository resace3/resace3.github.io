from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
STATIC_PAGES = [
    ROOT / "index.html",
    ROOT / "new-projects.html",
    ROOT / "causal-dag.html",
    ROOT / "agentic-prompt.html",
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
    assert "Daily PANAS summed score on a 10 to 50 scale" in html
    assert "value=\"33\"" in html
    assert "5.7" not in html
    assert "Mood score" not in html
    assert "Sleep duration and low mood" in html
    assert "95% CI -0.63 to -0.18" in html
    assert "Predicted probability of low mood" in html
    assert "Estimated mood difference" in html


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

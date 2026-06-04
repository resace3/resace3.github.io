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

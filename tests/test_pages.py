import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_learn_pages_links_and_images_resolve():
    pages = sorted((DOCS / "learn").glob("*.html"))
    assert len(pages) >= 30
    idx = (DOCS / "learn.html").read_text(encoding="utf-8")
    for p in pages:
        assert f"learn/{p.name}" in idx, p.name                       # every page is reachable from the index
        h = p.read_text(encoding="utf-8")
        for src in re.findall(r'src="([^"]+)"', h):
            assert (p.parent / src).resolve().exists(), (p.name, src)
        for href in re.findall(r'href="([^"#][^"]*)"', h):
            if href.startswith("http"):
                continue
            assert (p.parent / href).resolve().exists(), (p.name, href)
    for href in re.findall(r'href="(learn/[^"]+)"', idx):
        assert (DOCS / href).exists(), href


def test_interactive_pages_carry_their_script():
    for p in (DOCS / "learn").glob("*.html"):
        h = p.read_text(encoding="utf-8")
        if 'id="lad"' in h:
            assert "<script>" in h, p.name


def test_index_links_to_learn_and_has_no_stale_coming_next():
    h = (DOCS / "index.html").read_text(encoding="utf-8")
    assert 'href="learn.html"' in h and "Coming next" not in h

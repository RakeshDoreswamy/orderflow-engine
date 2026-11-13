import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_study_page_images_exist_and_cover_23_concepts():
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    imgs = re.findall(r'src="(img/[^"]+)"', html)
    assert all((ROOT / "docs" / i).exists() for i in imgs)
    nums = sorted(int(m) for m in re.findall(r"img/concepts/(\d\d)_", html))
    assert sorted(set(nums)) == list(range(1, 24))

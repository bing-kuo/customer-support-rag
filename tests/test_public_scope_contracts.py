from html.parser import HTMLParser
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


class ProductOptions(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_product = False
        self.current = None
        self.options = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "select" and attrs.get("id") == "product":
            self.in_product = True
        elif tag == "option" and self.in_product:
            self.current = [attrs.get("value", ""), ""]

    def handle_data(self, data):
        if self.current is not None:
            self.current[1] += data

    def handle_endtag(self, tag):
        if tag == "option" and self.current is not None:
            self.options.append((self.current[0], self.current[1].strip()))
            self.current = None
        elif tag == "select":
            self.in_product = False


def test_product_select_has_valid_synthetic_options_for_both_models():
    parser = ProductOptions()
    parser.feed((ROOT / "web" / "index.html").read_text())
    assert parser.options == [
        ("", "尚不確定型號"),
        ("hp-a", "Nuvquora A"),
        ("hp-b", "Nuvquora B"),
    ]


def test_cloud_setup_uses_repository_relative_env_instruction():
    setup = (ROOT / "docs" / "cloud-setup.md").read_text()
    assert "在專案根目錄建立或編輯 `.env`" in setup
    assert not re.search(r"(?:^|[\s`])/(?:Users|home|private|tmp)/", setup)
    assert not re.search(r"[A-Za-z]:\\(?:Users|home)\\", setup)

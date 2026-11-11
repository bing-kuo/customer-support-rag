"""Default app serves the repository web files. This is not browser QA."""

from pathlib import Path

from fastapi.testclient import TestClient

from customer_support_rag.api import app
from customer_support_rag.web_pages import default_web_root

WEB = Path(__file__).resolve().parents[1] / "web"
client = TestClient(app, base_url="http://127.0.0.1")


def test_default_app_serves_repository_index_and_named_assets() -> None:
    home = client.get("/")
    assert home.status_code == 200
    assert home.headers["content-type"].startswith("text/html")
    assert home.headers["x-content-type-options"] == "nosniff"
    assert home.content == (WEB / "index.html").read_bytes()
    assert b'href="/assets/style.css"' in home.content
    assert b'src="/assets/app.js"' in home.content
    assert b'value="hp-a"' in home.content
    assert b'value="hp-b"' in home.content
    css = client.get("/assets/style.css")
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    assert css.content == (WEB / "assets" / "style.css").read_bytes()
    script = client.get("/assets/app.js")
    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")
    assert script.content == (WEB / "assets" / "app.js").read_bytes()
    assert b'fetch("/ask"' in script.content
    assert b'"product_id"' in script.content
    assert b"localStorage" not in script.content
    assert b"innerHTML" not in script.content


def test_repository_web_does_not_publish_extra_or_legacy_routes() -> None:
    fixture = client.get("/assets/demo-fixtures.json")
    assert fixture.status_code == 404
    assert fixture.headers["content-type"].startswith("application/json")
    assert client.get("/index.html").status_code == 404
    legacy = client.post("/api/v1/sessions", json={})
    assert legacy.status_code == 404
    assert legacy.headers["content-type"].startswith("application/json")
    assert "text/html" not in legacy.headers["content-type"]


def test_packaged_web_root_matches_the_repository_web_directory() -> None:
    located = default_web_root()
    assert located.resolve() == WEB.resolve()
    assert (located / "index.html").is_file()
    assert (located / "assets" / "style.css").is_file()
    assert (located / "assets" / "app.js").is_file()

"""Tmp web-tree tests. The fixture pages are not the product UI."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from customer_support_rag.api import create_app
from customer_support_rag.web_pages import default_web_root


def test_homepage_and_named_assets_use_fixed_content_types(tmp_path: Path) -> None:
    root = _web_tree(tmp_path)
    client = _client(root)
    home = client.get("/")
    assert home.status_code == 200
    assert home.headers["content-type"].startswith("text/html")
    assert home.headers["x-content-type-options"] == "nosniff"
    assert home.text == "<!doctype html><title>fixture</title>"
    css = client.get("/assets/style.css")
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    assert css.text == "body{color:#111}"
    script = client.get("/assets/app.js")
    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")
    assert script.text == "window.__fixture=true;"


def test_missing_web_root_is_json_404(tmp_path: Path) -> None:
    client = _client(tmp_path / "absent-web")
    home = client.get("/")
    assert home.status_code == 404
    assert home.headers["content-type"].startswith("application/json")
    assert home.json() == {"detail": "Not Found"}
    asset = client.get("/assets/style.css")
    assert asset.status_code == 404
    assert asset.json() == {"detail": "Not Found"}


def test_index_html_is_not_an_asset_and_unknown_names_stay_json(
    tmp_path: Path,
) -> None:
    root = _web_tree(tmp_path)
    client = _client(root)
    listed = client.get("/index.html")
    assert listed.status_code == 404
    assert listed.headers["content-type"].startswith("application/json")
    assert client.get("/assets/").status_code == 404
    assert client.get("/assets/missing.js").status_code == 404
    post_home = client.post("/")
    assert post_home.status_code == 405
    assert post_home.headers["content-type"].startswith("application/json")


def test_hidden_db_fixture_and_traversal_are_not_downloadable(tmp_path: Path) -> None:
    root = _web_tree(tmp_path)
    client = _client(root)
    fixture = client.get("/assets/demo-fixtures.json")
    assert fixture.status_code == 404
    assert "secret-fixture" not in fixture.text
    assert client.get("/assets/.env").status_code == 404
    assert client.get("/.env").status_code == 404
    assert client.get("/secret.sqlite").status_code == 404
    assert client.get("/assets/../index.html").status_code == 404
    encoded = client.get("/assets/%2e%2e/index.html")
    assert encoded.status_code == 404
    assert "<!doctype html>" not in encoded.text
    escaped = client.get("/assets/%2e%2e%2findex.html")
    assert escaped.status_code == 404
    assert client.get("/assets/style.css/../../README.md").status_code == 404


def test_symlink_escape_is_not_served(tmp_path: Path) -> None:
    root = tmp_path / "web"
    assets = root / "assets"
    assets.mkdir(parents=True)
    (root / "index.html").write_text(
        "<!doctype html><title>ok</title>", encoding="utf-8"
    )
    outside = tmp_path / "outside.css"
    outside.write_text("escaped-css", encoding="utf-8")
    (assets / "style.css").symlink_to(outside)
    (assets / "app.js").write_text("window.__fixture=true;", encoding="utf-8")
    client = _client(root)
    css = client.get("/assets/style.css")
    assert css.status_code == 404
    assert "escaped-css" not in css.text
    assert client.get("/assets/app.js").status_code == 200


def test_assets_directory_symlink_is_not_followed(tmp_path: Path) -> None:
    root = tmp_path / "web"
    root.mkdir()
    (root / "index.html").write_text("<!doctype html>", encoding="utf-8")
    real_assets = tmp_path / "elsewhere"
    real_assets.mkdir()
    (real_assets / "style.css").write_text("escaped", encoding="utf-8")
    (root / "assets").symlink_to(real_assets)
    client = _client(root)
    css = client.get("/assets/style.css")
    assert css.status_code == 404
    assert "escaped" not in css.text


def test_default_web_root_is_the_repository_web_directory() -> None:
    located = default_web_root()
    assert located.name == "web"
    assert (
        located.parent.name == "local-demo-integration"
        or (located.parent / "src" / "customer_support_rag").is_dir()
    )
    assert not located.exists()


def _web_tree(tmp_path: Path) -> Path:
    root = tmp_path / "web"
    assets = root / "assets"
    assets.mkdir(parents=True)
    (root / "index.html").write_text(
        "<!doctype html><title>fixture</title>", encoding="utf-8"
    )
    (assets / "style.css").write_text("body{color:#111}", encoding="utf-8")
    (assets / "app.js").write_text("window.__fixture=true;", encoding="utf-8")
    (assets / "demo-fixtures.json").write_text(
        '{"secret-fixture":true}', encoding="utf-8"
    )
    (assets / ".env").write_text("hidden-env-file\n", encoding="utf-8")
    (root / ".env").write_text("hidden-env-file\n", encoding="utf-8")
    (root / "secret.sqlite").write_bytes(b"sqlite")
    return root


def _client(web_root: Path) -> TestClient:
    return TestClient(create_app(None, web_root=web_root), base_url="http://127.0.0.1")

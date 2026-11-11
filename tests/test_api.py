from fastapi.testclient import TestClient

from customer_support_rag import __version__
from customer_support_rag.api import app

client = TestClient(app, base_url="http://127.0.0.1")


def test_health_returns_service_status_and_identity() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "application": "customer-support-rag",
        "version": __version__,
    }


def test_health_rejects_unsupported_method() -> None:
    assert client.post("/health").status_code == 405


def test_unknown_route_returns_not_found() -> None:
    response = client.get("/missing")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Not Found"}


def test_unknown_post_is_json_not_html() -> None:
    response = client.post("/missing", json={"product_id": "hp-a"})
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert "text/html" not in response.headers["content-type"]
    assert response.json() == {"detail": "Not Found"}

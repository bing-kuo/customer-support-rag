from fastapi.testclient import TestClient

from customer_support_rag import __version__
from customer_support_rag.api import app

client = TestClient(app)


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
    assert client.get("/missing").status_code == 404

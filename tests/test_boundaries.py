from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_unapproved_release_fails_closed():
    sid = client.post('/api/v1/sessions', json={}).json()['session_id']
    response = client.post(f'/api/v1/sessions/{sid}/turns', json={'client_turn_id': 'one', 'message': 'pair', 'product_id': 'hp-a'})
    assert response.status_code == 503
    assert response.json()['error']['code'] == 'no_active_release'


def test_schema_and_origin_boundaries():
    assert client.post('/api/v1/sessions', json={'role': 'system'}).status_code == 422
    assert client.post('/api/v1/sessions', json={}, headers={'origin': 'https://evil.example'}).status_code == 400
    assert client.post('/api/v1/sessions', content='x' * 16385).status_code == 413
    assert client.get('/api/v1/sources/candidate').status_code == 404

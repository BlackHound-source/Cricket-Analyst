"""Small API smoke test; run from repository root after installing backend requirements."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi.testclient import TestClient
from app.main import app


def test_health():
    response = TestClient(app).get('/api/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


if __name__ == '__main__':
    test_health()
    print('API smoke test passed.')

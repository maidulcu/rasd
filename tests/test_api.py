def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Video Analyzer Starter" in response.text
    assert "Analyze" in response.text


def test_home_page_has_url_input(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'name="video_url"' in response.text
    assert 'type="url"' in response.text


def test_results_page_not_found(client):
    response = client.get("/results/99999")
    assert response.status_code == 404


def test_analyze_missing_url(client):
    response = client.post("/analyze", data={})
    assert response.status_code == 422

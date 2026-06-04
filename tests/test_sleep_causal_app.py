import pytest

from sleep_causal_app import create_app


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("SLEEP_ANALYSIS_CONFIG", str(tmp_path / "sleep_analysis.json"))
    monkeypatch.delenv("PERSONAL_WEBSITE_USE_HOME_ASSISTANT", raising=False)
    monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)

    flask_app = create_app()
    flask_app.config.update(TESTING=True)
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.mark.parametrize(
    "path, expected_text",
    [
        ("/", b"N of 1 Causal Analysis Engine"),
        ("/dag", b"N of 1 Causal Analysis Engine"),
        ("/sensor-maps", b"Sensor"),
        ("/saved-analyses", b"Saved"),
    ],
)
def test_core_pages_render(client, path, expected_text):
    response = client.get(path, follow_redirects=True)

    assert response.status_code == 200
    assert expected_text in response.data


def test_demo_histogram_routes_return_json(client):
    outcome = client.get("/outcome-histogram")
    covariate = client.get(
        "/covariate-histogram",
        query_string={
            "sensor": "sensor.nick_r_steps",
            "sleep_sensor": "sensor.nick_r_mood_score",
            "analysis_window_days": "90",
        },
    )

    assert outcome.status_code == 200
    assert outcome.is_json
    assert outcome.get_json()["ok"] is True
    assert covariate.status_code == 200
    assert covariate.is_json
    assert covariate.get_json()["ok"] is True


def test_demo_analysis_runs(client):
    response = client.post("/run")

    assert response.status_code == 200
    assert b"N of 1 Causal Analysis Engine" in response.data
    assert b"Sleep duration" in response.data


import json
import os
import re

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
        ("/sensor-maps", b"Variable"),
        ("/saved-analyses", b"Saved"),
    ],
)
def test_core_pages_render(client, path, expected_text):
    response = client.get(path, follow_redirects=True)

    assert response.status_code == 200
    assert expected_text in response.data


def test_default_dag_uses_panas_score_label(client):
    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert b"PANAS Score" in response.data
    assert b'value="33.0"' in response.data
    assert b"10 to 50" in response.data
    assert b"Mood score" not in response.data
    assert re.findall(r'<strong class="dag-node-title" data-node-title>(.*?)</strong>', html) == [
        "PANAS Score (t-1)",
        "Sleep",
        "Steps",
        "PANAS Score",
    ]


def test_refresh_resets_saved_dag_to_default_demo(client):
    config_path = os.environ["SLEEP_ANALYSIS_CONFIG"]
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "causal_dag": {
                    "nodes": [
                        {
                            "id": "sensor.edited_outcome",
                            "label": "Edited outcome",
                            "role": "outcome",
                        }
                    ],
                    "edges": [],
                }
            },
            handle,
        )

    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Edited outcome" not in html
    assert re.findall(r'<strong class="dag-node-title" data-node-title>(.*?)</strong>', html) == [
        "PANAS Score (t-1)",
        "Sleep",
        "Steps",
        "PANAS Score",
    ]
    edges = set(
        re.findall(
            r'name="edge_source" value="([^"]+)" data-edge-source>\s+<input name="edge_target" value="([^"]+)"',
            html,
        )
    )
    assert edges == {
        ("derived.nick_r_mood_score_lag1", "sensor.nick_r_sleep_minutes_asleep"),
        ("derived.nick_r_mood_score_lag1", "sensor.nick_r_steps"),
        ("sensor.nick_r_sleep_minutes_asleep", "sensor.nick_r_mood_score"),
        ("sensor.nick_r_steps", "sensor.nick_r_mood_score"),
    }

    with open(config_path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert [node["label"] for node in saved["causal_dag"]["nodes"]] == [
        "PANAS Score (t-1)",
        "Sleep",
        "Steps",
        "PANAS Score",
    ]


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
    prior_panas = client.get(
        "/covariate-histogram",
        query_string={
            "sensor": "derived.nick_r_mood_score_lag1",
            "sleep_sensor": "sensor.nick_r_mood_score",
            "analysis_window_days": "90",
        },
    )

    assert outcome.status_code == 200
    assert outcome.is_json
    outcome_data = outcome.get_json()
    assert outcome_data["ok"] is True
    assert 10 <= outcome_data["minimum"] <= outcome_data["maximum"] <= 50
    assert covariate.status_code == 200
    assert covariate.is_json
    assert covariate.get_json()["ok"] is True
    assert prior_panas.status_code == 200
    assert prior_panas.is_json
    assert prior_panas.get_json()["ok"] is True


def test_demo_analysis_runs(client):
    response = client.post("/run")

    assert response.status_code == 200
    assert b"N of 1 Causal Analysis Engine" in response.data
    assert b"Clinician summary" in response.data
    assert b"What the DAG says may cause what" in response.data
    assert b"Prior PANAS Score may affect Sleep" in response.data
    assert b"Prior PANAS Score may affect Steps" in response.data
    assert b"Sleep may contribute to current PANAS Score changes" in response.data
    assert b"Steps may contribute to current PANAS Score changes" in response.data
    assert b"Reverse causality or unmeasured factors" in response.data
    assert b"direction is uncertain" not in response.data
    assert b"Advanced results" not in response.data
    assert b"Effect estimate distribution" not in response.data
    assert b"effect distribution" not in response.data
    assert b"Show plots and technical details" in response.data
    assert b"Treatment effect estimate" in response.data
    assert b"When Sleep is above 420" in response.data
    assert b"95% CI" in response.data
    assert b"Observed vs predicted outcome plot" in response.data
    assert b"Run the model from the setup panel" not in response.data


def test_analysis_setup_uses_run_analysis_button_label(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b'data-run-analysis-button' in response.data
    assert b"Run Analysis" in response.data
    assert b"Recommended: G-formula / outcome model" in response.data
    assert b"Default (recommended): G-formula / outcome model" not in response.data
    assert b"Already selected for this demo. Change only if you want a specific causal estimator." in response.data
    assert b"Choose from DAG" not in response.data


def test_demo_is_simulated_only_even_with_home_assistant_env(client, monkeypatch):
    monkeypatch.setenv("PERSONAL_WEBSITE_USE_HOME_ASSISTANT", "1")
    monkeypatch.setenv("SUPERVISOR_TOKEN", "fake-token")

    search_response = client.get("/entities/search", query_string={"q": "sensor."})
    analysis_response = client.post("/run")

    assert search_response.status_code == 200
    entities = search_response.get_json()["entities"]
    entity_ids = {row["entity_id"] for row in entities}
    assert entity_ids == {
        "sensor.nick_r_mood_score",
        "sensor.nick_r_sleep_minutes_asleep",
        "sensor.nick_r_steps",
    }
    assert analysis_response.status_code == 200
    assert b"Clinician summary" in analysis_response.data
    assert b"Home Assistant" not in analysis_response.data
    assert b"Simulated" in analysis_response.data or b"simulated" in analysis_response.data


def test_sidebar_uses_simulated_variable_and_advanced_toggle(client):
    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Simulated variable" in html
    assert "Sensor not found" not in html
    assert '<details class="sidebar-advanced-settings">' in html
    assert "<summary>Advanced</summary>" in html
    assert html.index("<summary>Advanced</summary>") < html.index("Daily aggregate")
    assert html.index("Daily aggregate") < html.index("From time") < html.index("To time")


def test_autosave_routes_return_ok_json(client):
    dag_response = client.post("/dag/autosave")
    settings_response = client.post("/settings/autosave")

    assert dag_response.status_code == 200
    assert dag_response.is_json
    assert dag_response.get_json()["ok"] is True
    assert settings_response.status_code == 200
    assert settings_response.is_json
    assert settings_response.get_json()["ok"] is True

import copy
import json
from datetime import datetime, timezone

import pytest

from sleep_causal_app.main import (
    DEFAULT_CONFIG,
    PRIOR_PANAS_ENTITY,
    analysis_report_snapshot,
    build_dag_svg,
    build_saved_analysis_pdf,
    build_saved_analysis_snapshot,
    find_saved_analysis,
    humanize_report_label,
    normalize_saved_analyses,
)


def fake_result():
    return {
        "ok": True,
        "analysis_kind": "dynamic_causal",
        "rows": 90,
        "date_start": "2026-01-01",
        "date_end": "2026-03-31",
        "outcome_label": "sensor.nick_r_mood_score",
        "exposure_label": "last(sensor.nick_r_sleep_minutes_asleep) more than 420",
        "analysis_method": "g_formula",
        "analysis_method_label": "G-formula / outcome model",
        "covariate_count": 2,
        "latest_effect": {
            "date": "2026-03-31",
            "contemporaneous": 0.73,
            "lower": -0.54,
            "upper": 1.99,
            "lag1": 0.12,
            "cumulative": 0.91,
        },
        "clinician_summary": {
            "headline": "Sleep may contribute to current PANAS Score changes",
            "answer": "When Sleep is above 420 minutes, PANAS Score is estimated to be higher.",
            "confidence": "Suggestive, not definitive",
            "cause_chain": [
                "Prior PANAS Score may affect Sleep",
                "Prior PANAS Score may affect Steps",
                "Sleep may contribute to current PANAS Score changes",
                "Steps may contribute to current PANAS Score changes",
            ],
            "caveat": "This uses simulated observational data.",
        },
        "method_outputs": {
            "method": "g_formula",
            "method_label": "G-formula / outcome model",
            "primary": [
                {
                    "id": "effect_estimate",
                    "kind": "effect_estimate",
                    "title": "Treatment effect estimate",
                    "estimate": 0.73,
                    "lower": -0.54,
                    "upper": 1.99,
                }
            ],
            "main": [
                {
                    "id": "predicted_outcome_curves",
                    "kind": "line_chart",
                    "title": "Predicted PANAS Score",
                    "subtitle": "Counterfactual PANAS Score prediction.",
                    "chart": {
                        "points": [{"date": "2026-01-01"}],
                        "series": [{"key": "exposure", "label": "Exposure (Independent)"}],
                    },
                },
                {
                    "id": "counterfactual_summary",
                    "kind": "counterfactual_table",
                    "title": "Estimated PANAS Score difference",
                    "rows": [{"component": "Contrast", "mean": 0.73, "detail": "Synthetic example"}],
                },
            ],
            "diagnostics": [],
            "advanced": [],
            "validation_errors": [],
        },
        "result_validation_errors": [],
        "positivity": [],
        "model_terms": [],
    }


def test_saved_analysis_snapshot_contains_report_payload_without_advanced_outputs():
    config = copy.deepcopy(DEFAULT_CONFIG)
    created_at = datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc)

    snapshot = build_saved_analysis_snapshot(config, fake_result(), created_at)

    assert snapshot["id"] == "20260331120000"
    assert snapshot["immutable"] is True
    assert snapshot["metadata"]["analysis_method_label"] == "G-formula / outcome model"
    assert snapshot["settings"]["sleep_sensor"] == "sensor.nick_r_mood_score"
    assert [node["label"] for node in snapshot["variable_settings"]] == [
        "PANAS Score (t-1)",
        "Sleep",
        "Steps",
        "PANAS Score",
    ]
    assert snapshot["plots"]["advanced"] == []
    assert "effect_distribution" not in str(snapshot)
    assert "Influence diagnostics" not in str(snapshot)


def test_analysis_report_snapshot_returns_embedded_snapshot_when_present():
    snapshot = {"id": "saved", "immutable": True, "metadata": {"rows": 90}}

    assert analysis_report_snapshot({"snapshot": snapshot}) is snapshot


def test_analysis_report_snapshot_reconstructs_legacy_saved_analysis():
    analysis = {
        "id": "legacy",
        "name": "Legacy analysis",
        "created_at": "2026-03-31 12:00 UTC",
        "settings": copy.deepcopy(DEFAULT_CONFIG),
        "result": fake_result(),
    }

    report = analysis_report_snapshot(analysis)

    assert report["id"] == "legacy"
    assert report["name"] == "Legacy analysis"
    assert report["metadata"]["rows"] == 90
    assert report["metadata"]["analysis_method_label"] == "G-formula / outcome model"
    assert "PANAS Score" in report["dag_svg"]


def test_build_dag_svg_contains_default_graph_labels_and_arrows():
    svg = build_dag_svg(DEFAULT_CONFIG["causal_dag"])

    assert svg.startswith('<svg viewBox=')
    assert "PANAS Score (t-1)" in svg
    assert "Sleep" in svg
    assert "Steps" in svg
    assert "PANAS Score" in svg
    assert svg.count('marker-end="url(#report-arrow)"') == 4


def test_saved_analysis_pdf_is_generated_without_advanced_section_label():
    snapshot = build_saved_analysis_snapshot(
        copy.deepcopy(DEFAULT_CONFIG),
        fake_result(),
        datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
    )

    pdf = build_saved_analysis_pdf(snapshot)

    assert pdf.startswith(b"%PDF-1.4")
    assert b"Causal Analysis Engine report" in pdf
    assert b"PANAS Score" in pdf
    assert b"Advanced results" not in pdf
    assert b"Effect estimate distribution" not in pdf


@pytest.mark.parametrize(
    "value, expected",
    [
        (PRIOR_PANAS_ENTITY, "Mood Score Lag1"),
        ("sensor.nick_r_steps", "Steps"),
        ("binary_sensor.pantry_door_window", "Pantry Door Window"),
        ("sensor.nick_r_sleep_minutes_asleep", "Sleep Minutes Asleep"),
        ("", "Variable"),
    ],
)
def test_humanize_report_label(value, expected):
    assert humanize_report_label(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ([{"id": "a"}, "bad", {"id": "b"}], [{"id": "a"}, {"id": "b"}]),
        ("bad", []),
        (None, []),
    ],
)
def test_normalize_saved_analyses(value, expected):
    assert normalize_saved_analyses(value) == expected


def test_find_saved_analysis_matches_stringified_ids():
    config = {"saved_analyses": [{"id": 123, "name": "A"}, {"id": "456", "name": "B"}]}

    assert find_saved_analysis(config, "123")["name"] == "A"
    assert find_saved_analysis(config, 456)["name"] == "B"
    assert find_saved_analysis(config, "missing") is None


def test_save_report_pdf_and_delete_routes(isolated_client, config_path):
    save_response = isolated_client.post("/analyses/save", follow_redirects=False)
    assert save_response.status_code == 302

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    saved = config["saved_analyses"][0]
    analysis_id = saved["id"]

    report_response = isolated_client.get(f"/saved-analyses/{analysis_id}")
    pdf_response = isolated_client.get(f"/saved-analyses/{analysis_id}/pdf")
    delete_response = isolated_client.post(f"/saved-analyses/{analysis_id}/delete", follow_redirects=False)

    assert report_response.status_code == 200
    assert b"Clinician summary" in report_response.data
    assert b"Advanced results" not in report_response.data
    assert pdf_response.status_code == 200
    assert pdf_response.mimetype == "application/pdf"
    assert pdf_response.data.startswith(b"%PDF-1.4")
    assert delete_response.status_code == 302

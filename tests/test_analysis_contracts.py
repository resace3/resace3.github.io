import copy

import numpy as np
import pandas as pd
import pytest

import sleep_causal_app.main as main
from sleep_causal_app.main import (
    DEFAULT_CONFIG,
    METHOD_RESULT_SPECS,
    PRIOR_PANAS_ENTITY,
    build_clinician_summary,
    build_counterfactual_predictions,
    build_counterfactual_summary_output,
    build_effective_sample_size_output,
    build_method_result_outputs,
    build_multi_series_chart,
    build_numeric_histogram_bins,
    build_observed_predicted_output,
    build_predicted_outcome_curves_output,
    build_propensity_diagnostics,
    build_treatment_effect_interpretation,
    causal_covariate_columns,
    confidence_interval_phrase,
    exposure_state_text,
    format_interpretation_number,
    format_signed_value,
    format_threshold_value,
    is_result_output_non_empty,
    resolve_result_method,
    result_method_spec,
    result_output_failure_reason,
    run_dynamic_causal_analysis,
    standardized_mean_difference,
    validate_method_outputs,
    weighted_mean,
    weighted_variance,
)


def sample_dataset(rows=28):
    dates = pd.date_range("2026-01-01", periods=rows, freq="D")
    exposure = np.array([0, 1] * ((rows + 1) // 2), dtype=float)[:rows]
    return pd.DataFrame(
        {
            "outcome": 32 + np.sin(np.arange(rows) / 3.0) + exposure * 0.8,
            "exposure": exposure,
            "exposure_raw": 390 + exposure * 70 + np.arange(rows) % 4,
            "sensor.nick_r_steps": 7600 + np.arange(rows) * 25,
            PRIOR_PANAS_ENTITY: 34 + np.cos(np.arange(rows) / 4.0),
        },
        index=dates,
    )


def sample_effect_model(rows=28, lower=-0.25, upper=2.1):
    dates = pd.date_range("2026-01-01", periods=rows, freq="D")
    effects = [
        {
            "date": date.strftime("%Y-%m-%d"),
            "effect": round(0.8 + index * 0.01, 4),
            "lower": lower,
            "upper": upper,
        }
        for index, date in enumerate(dates)
    ]
    return {
        "effects": effects,
        "latest_effect": {
            "date": dates[-1].strftime("%Y-%m-%d"),
            "contemporaneous": effects[-1]["effect"],
            "lower": lower,
            "upper": upper,
            "lag1": 0.2,
            "cumulative": 1.4,
        },
    }


def test_method_result_specs_do_not_include_advanced_effect_distribution():
    assert set(METHOD_RESULT_SPECS) == {"ipw", "g_formula", "doubly_robust"}
    for spec in METHOD_RESULT_SPECS.values():
        assert spec["advanced"] == []
        assert "effect_distribution" not in spec["expected"]
        assert "influence_diagnostics" not in spec["expected"]


@pytest.mark.parametrize(
    "method, dataset, expected",
    [
        ("g_formula", pd.DataFrame(), "g_formula"),
        ("ipw", pd.DataFrame(), "ipw"),
        ("doubly_robust", pd.DataFrame(), "doubly_robust"),
        ("unknown", pd.DataFrame(), "g_formula"),
        ("dag_auto", pd.DataFrame({"outcome": [], "exposure": []}), "g_formula"),
        ("dag_auto", pd.DataFrame({"outcome": [], "exposure": [], "sensor.x": []}), "doubly_robust"),
    ],
)
def test_resolve_result_method(method, dataset, expected):
    assert resolve_result_method(method, dataset) == expected


@pytest.mark.parametrize(
    "method, expected_primary, expected_main, expected_diagnostics",
    [
        ("g_formula", ["effect_estimate"], ["predicted_outcome_curves", "observed_predicted_plot", "counterfactual_summary"], []),
        ("ipw", ["effect_estimate"], [], ["love_plot", "weight_histogram", "effective_sample_size"]),
        ("doubly_robust", ["effect_estimate"], ["predicted_outcome_curves"], ["love_plot", "weight_histogram", "effective_sample_size"]),
    ],
)
def test_build_method_result_outputs_returns_expected_sections(method, expected_primary, expected_main, expected_diagnostics):
    outputs = build_method_result_outputs(sample_dataset(), sample_effect_model(), copy.deepcopy(DEFAULT_CONFIG), method)

    assert outputs["method"] == method
    assert [row["id"] for row in outputs["primary"]] == expected_primary
    assert [row["id"] for row in outputs["main"]] == expected_main
    assert [row["id"] for row in outputs["diagnostics"]] == expected_diagnostics
    assert outputs["advanced"] == []
    assert outputs["validation_errors"] == []


@pytest.mark.parametrize(
    "method, expected_ids",
    [
        ("g_formula", ["effect_estimate", "predicted_outcome_curves", "observed_predicted_plot", "counterfactual_summary"]),
        ("ipw", ["effect_estimate", "love_plot", "weight_histogram", "effective_sample_size"]),
        ("doubly_robust", ["effect_estimate", "love_plot", "weight_histogram", "predicted_outcome_curves", "effective_sample_size"]),
        ("dag_auto", ["effect_estimate", "predicted_outcome_curves", "observed_predicted_plot", "counterfactual_summary"]),
    ],
)
def test_result_method_spec_expected_ids(method, expected_ids):
    assert result_method_spec(method)["expected"] == expected_ids


@pytest.mark.parametrize(
    "output, expected",
    [
        (None, False),
        ({}, False),
        ({"kind": "effect_estimate", "estimate": 1, "lower": 0, "upper": 2}, True),
        ({"kind": "effect_estimate", "estimate": 1, "lower": None, "upper": 2}, False),
        ({"kind": "love_plot", "rows": [{"label": "x"}]}, True),
        ({"kind": "love_plot", "rows": []}, False),
        ({"kind": "histogram", "bins": [{"count": 0}]}, False),
        ({"kind": "histogram", "bins": [{"count": 3}]}, True),
        ({"kind": "ess_summary", "rows": [{"label": "Effective sample size", "value": 0}]}, False),
        ({"kind": "ess_summary", "rows": [{"label": "Effective sample size", "value": 12.5}]}, True),
        ({"kind": "line_chart", "chart": {"points": [{"x": 1}], "series": [{"key": "x"}]}}, True),
        ({"kind": "line_chart", "chart": {"points": [], "series": [{"key": "x"}]}}, False),
        ({"kind": "counterfactual_table", "rows": [{"component": "Contrast"}]}, True),
        ({"kind": "influence_diagnostics", "rows": []}, False),
    ],
)
def test_is_result_output_non_empty(output, expected):
    assert is_result_output_non_empty(output) is expected


@pytest.mark.parametrize(
    "output_id, output, expected_fragment",
    [
        ("effect_estimate", {}, "estimate or confidence interval"),
        ("love_plot", {}, "no covariates"),
        ("weight_histogram", {}, "no finite inverse probability weights"),
        ("effective_sample_size", {}, "empty or zero effective sample size"),
        ("predicted_outcome_curves", {}, "plottable predicted values"),
        ("observed_predicted_plot", {}, "plottable predicted values"),
        ("counterfactual_summary", {}, "no rows"),
        ("unknown", None, "not created"),
    ],
)
def test_result_output_failure_reason_is_specific(output_id, output, expected_fragment):
    assert expected_fragment in result_output_failure_reason(output_id, output)


def test_validate_method_outputs_names_missing_outputs():
    errors = validate_method_outputs("g_formula", ["effect_estimate", "predicted_outcome_curves"], {})

    assert [row["output_id"] for row in errors] == ["effect_estimate", "predicted_outcome_curves"]
    assert "G-formula" in errors[0]["message"]


@pytest.mark.parametrize(
    "values, bins, expected_counts",
    [
        (pd.Series([], dtype=float), 10, []),
        (pd.Series([5, 5, 5]), 10, [3]),
        (pd.Series([1, 2, 3, 4]), 10, [1, 1, 1, 1]),
        (pd.Series([1, np.inf, np.nan, 2]), 10, [1, 0, 0, 1]),
    ],
)
def test_build_numeric_histogram_bins(values, bins, expected_counts):
    rows = build_numeric_histogram_bins(values, bins=bins)

    assert [row["count"] for row in rows] == expected_counts
    if rows:
        assert max(row["height"] for row in rows) == 100


@pytest.mark.parametrize(
    "values, weights, expected",
    [
        (pd.Series([1.0, 3.0]), pd.Series([1.0, 1.0]), 2.0),
        (pd.Series([1.0, 3.0]), pd.Series([0.0, 0.0]), 2.0),
        (pd.Series([1.0, 3.0]), pd.Series([1.0, 3.0]), 2.5),
    ],
)
def test_weighted_mean(values, weights, expected):
    assert weighted_mean(values, weights) == expected


@pytest.mark.parametrize(
    "values, weights, mean, expected",
    [
        (pd.Series([1.0, 3.0]), pd.Series([1.0, 1.0]), 2.0, 1.0),
        (pd.Series([1.0, 3.0]), pd.Series([0.0, 0.0]), 2.0, 1.0),
        (pd.Series([1.0, 3.0]), pd.Series([1.0, 3.0]), 2.5, 0.75),
    ],
)
def test_weighted_variance(values, weights, mean, expected):
    assert weighted_variance(values, weights, mean) == expected


@pytest.mark.parametrize(
    "values, exposure, expected",
        [
            (pd.Series([1.0, 1.0, 1.0, 1.0]), pd.Series([0, 0, 1, 1]), 0.0),
            (pd.Series([1.0, 2.0, 5.0, 6.0]), pd.Series([0, 0, 1, 1]), 8.0),
            (pd.Series([1.0, 2.0]), pd.Series([1, 1]), 0.0),
        ],
)
def test_standardized_mean_difference(values, exposure, expected):
    assert standardized_mean_difference(values, exposure) == expected


def test_propensity_diagnostics_and_effective_sample_size_are_nonempty():
    propensity = build_propensity_diagnostics(sample_dataset())
    ess = build_effective_sample_size_output(propensity, len(sample_dataset()))

    assert set(propensity) == {"covariates", "probabilities", "weights"}
    assert propensity["probabilities"].between(0.03, 0.97).all()
    assert is_result_output_non_empty(ess) is True


def test_causal_covariate_columns_excludes_core_analysis_columns():
    assert causal_covariate_columns(sample_dataset()) == ["sensor.nick_r_steps", PRIOR_PANAS_ENTITY]


def test_counterfactual_prediction_outputs_are_plottable():
    points = build_counterfactual_predictions(sample_dataset(), sample_effect_model())
    curves = build_predicted_outcome_curves_output(points)
    observed = build_observed_predicted_output(points)
    summary = build_counterfactual_summary_output(points, sample_effect_model())

    assert len(points) == 28
    assert is_result_output_non_empty(curves)
    assert is_result_output_non_empty(observed)
    assert is_result_output_non_empty(summary)
    assert [row["component"] for row in summary["rows"]] == ["No exposure", "Exposure (Independent)", "Contrast"]


def test_multi_series_chart_handles_empty_and_populated_points():
    assert build_multi_series_chart([], [{"key": "x", "label": "X"}]) == {
        "points": [],
        "series": [],
        "x_ticks": [],
        "y_ticks": [],
        "minimum": 0,
        "maximum": 0,
    }

    chart = build_multi_series_chart(
        [{"date": "2026-01-01", "a": 1, "b": 3}, {"date": "2026-01-02", "a": 2, "b": 4}],
        [{"key": "a", "label": "A"}, {"key": "b", "label": "B"}],
    )

    assert len(chart["points"]) == 2
    assert len(chart["series"]) == 2
    assert chart["minimum"] == 1
    assert chart["maximum"] == 4


def test_clinician_summary_is_plain_language_and_professional_about_uncertainty():
    summary = build_clinician_summary(
        copy.deepcopy(DEFAULT_CONFIG),
        sample_effect_model(lower=-0.2, upper=2.1),
        sample_dataset(),
        "g_formula",
    )

    assert summary["headline"] == "Sleep may contribute to current PANAS Score changes"
    assert "after accounting for Steps and PANAS Score (t-1)" in summary["answer"]
    assert summary["confidence"] == "Suggestive, not definitive"
    assert "Reverse causality or unmeasured factors" in summary["caveat"]
    assert "direction is uncertain" not in str(summary)
    assert summary["cause_chain"] == [
        "Prior PANAS Score may affect Sleep",
        "Prior PANAS Score may affect Steps",
        "Sleep may contribute to current PANAS Score changes",
        "Steps may contribute to current PANAS Score changes",
    ]


@pytest.mark.parametrize(
    "comparison, expected",
    [
        ("exposed", "Sleep is above 420"),
        ("reference", "Sleep is at or below 420"),
    ],
)
def test_exposure_state_text_uses_dag_label_and_threshold(comparison, expected):
    assert exposure_state_text(DEFAULT_CONFIG, comparison) == expected


@pytest.mark.parametrize(
    "latest, expected_fragment",
    [
        ({"contemporaneous": 1.2, "lower": -0.2, "upper": 2.1}, "95% confidence interval"),
        ({"contemporaneous": None, "lower": -0.2, "upper": 2.1}, ""),
    ],
)
def test_treatment_effect_interpretation_requires_estimate(latest, expected_fragment):
    interpretation = build_treatment_effect_interpretation(latest, copy.deepcopy(DEFAULT_CONFIG), "g_formula")

    assert expected_fragment in interpretation


@pytest.mark.parametrize(
    "func, value, expected",
    [
        (format_interpretation_number, 0.123, "0.12"),
        (format_interpretation_number, 12.34, "12.3"),
        (format_interpretation_number, 1234, "1,234"),
        (format_threshold_value, 420.0, "420"),
        (format_threshold_value, 420.5, "420.5"),
        (format_signed_value, -0.123, "-0.12"),
        (format_signed_value, 1234, "1,234"),
    ],
)
def test_number_formatters(func, value, expected):
    assert func(value) == expected


@pytest.mark.parametrize(
    "lower, upper, expected",
    [
        (-0.2, 2.1, "about 0.20 outcome units lower to about 2.10 outcome units higher"),
        (0.5, 2.1, "0.50 to 2.10 outcome units higher"),
        (-2.1, -0.5, "0.50 to 2.10 outcome units lower"),
        ("bad", 2.1, ""),
    ],
)
def test_confidence_interval_phrase(lower, upper, expected):
    assert confidence_interval_phrase(lower, upper, "outcome units") == expected


def test_default_dynamic_analysis_runs_with_clinician_facing_output():
    result = run_dynamic_causal_analysis(copy.deepcopy(DEFAULT_CONFIG))

    assert result["ok"] is True
    assert result["analysis_method"] == "g_formula"
    assert result["outcome_label"] == "sensor.nick_r_mood_score"
    assert result["covariate_count"] == 2
    assert result["method_outputs"]["advanced"] == []
    assert result["method_outputs"]["validation_errors"] == []
    assert result["clinician_summary"]["headline"] == "Sleep may contribute to current PANAS Score changes"
    assert "simulated observational data" in result["clinician_summary"]["caveat"]
    assert "Advanced results" not in str(result)
    assert "Effect estimate distribution" not in str(result)


@pytest.mark.parametrize(
    "overrides, expected_message",
    [
        ({"sleep_sensor": ""}, "Choose an outcome variable"),
        ({"exposure_sensor": ""}, "Choose a time-varying exposure variable"),
    ],
)
def test_dynamic_analysis_reports_missing_required_variables(overrides, expected_message):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(overrides)

    result = run_dynamic_causal_analysis(config)

    assert result["ok"] is False
    assert expected_message in result["message"]


def test_dynamic_analysis_reports_short_dataset(monkeypatch):
    config = copy.deepcopy(DEFAULT_CONFIG)
    short = sample_dataset(rows=10)
    monkeypatch.setattr(main, "fetch_home_assistant_history", lambda _config: pd.DataFrame({"state": [1.0]}))
    monkeypatch.setattr(main, "prepare_causal_dataset", lambda _history, _config: short)

    result = run_dynamic_causal_analysis(config)

    assert result["ok"] is False
    assert "More usable daily observations" in result["message"]


def test_dynamic_analysis_reports_one_class_exposure(monkeypatch):
    config = copy.deepcopy(DEFAULT_CONFIG)
    one_class = sample_dataset(rows=40)
    one_class["exposure"] = 1.0
    one_class["exposure_raw"] = 420.0
    monkeypatch.setattr(main, "fetch_home_assistant_history", lambda _config: pd.DataFrame({"state": [1.0]}))
    monkeypatch.setattr(main, "prepare_causal_dataset", lambda _history, _config: one_class)

    result = run_dynamic_causal_analysis(config)

    assert result["ok"] is False
    assert "exposure threshold creates only one exposure class" in result["message"]

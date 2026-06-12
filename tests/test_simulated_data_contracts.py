import copy

import pandas as pd
import pytest

from sleep_causal_app.main import (
    DEFAULT_CONFIG,
    PRIOR_PANAS_ENTITY,
    PRIOR_PANAS_SOURCE_ENTITY,
    apply_balanced_exposure_threshold,
    balanced_binary_threshold,
    build_binary_exposure,
    build_binary_level_histogram,
    build_binary_outcome,
    build_covariate_histogram,
    build_demo_history,
    build_histogram_association,
    build_outcome_histogram,
    build_series_histogram,
    daily_sensor_values,
    daily_values_for_entity,
    fetch_home_assistant_history,
    prepare_causal_dataset,
    prepare_dataset,
    search_home_assistant_entities,
)


def make_repeated_daily_history():
    return pd.DataFrame(
        [
            {"entity_id": "sensor.a", "last_changed": pd.Timestamp("2026-01-01 01:00Z"), "state": 1.0, "raw_state": "1"},
            {"entity_id": "sensor.a", "last_changed": pd.Timestamp("2026-01-01 20:00Z"), "state": 3.0, "raw_state": "3"},
            {"entity_id": "sensor.a", "last_changed": pd.Timestamp("2026-01-02 01:00Z"), "state": 2.0, "raw_state": "2"},
            {"entity_id": "sensor.a", "last_changed": pd.Timestamp("2026-01-02 20:00Z"), "state": 6.0, "raw_state": "6"},
            {"entity_id": "sensor.b", "last_changed": pd.Timestamp("2026-01-01 20:00Z"), "state": 10.0, "raw_state": "10"},
        ]
    )


@pytest.mark.parametrize("analysis_window_days", [14, 90, 365, 730])
def test_demo_history_contains_only_simulated_source_entities(analysis_window_days):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["analysis_window_days"] = analysis_window_days

    history = build_demo_history(config)

    expected_days = max(120, analysis_window_days + 14)
    assert len(history) == expected_days * 3
    assert set(history["entity_id"]) == {
        PRIOR_PANAS_SOURCE_ENTITY,
        "sensor.nick_r_sleep_minutes_asleep",
        "sensor.nick_r_steps",
    }
    assert PRIOR_PANAS_ENTITY not in set(history["entity_id"])


@pytest.mark.parametrize(
    "entity, minimum, maximum",
    [
        (PRIOR_PANAS_SOURCE_ENTITY, 10, 50),
        ("sensor.nick_r_sleep_minutes_asleep", 300, 560),
        ("sensor.nick_r_steps", 4000, 12000),
    ],
)
def test_demo_history_value_ranges_match_default_example(entity, minimum, maximum):
    history = build_demo_history(DEFAULT_CONFIG)
    values = history.loc[history["entity_id"] == entity, "state"]

    assert not values.empty
    assert values.min() >= minimum
    assert values.max() <= maximum


def test_fetch_history_ignores_home_assistant_environment(monkeypatch):
    monkeypatch.setenv("PERSONAL_WEBSITE_USE_HOME_ASSISTANT", "1")
    monkeypatch.setenv("SUPERVISOR_TOKEN", "fake-token")

    history = fetch_home_assistant_history(DEFAULT_CONFIG)

    assert set(history["entity_id"]) == {
        PRIOR_PANAS_SOURCE_ENTITY,
        "sensor.nick_r_sleep_minutes_asleep",
        "sensor.nick_r_steps",
    }


@pytest.mark.parametrize(
    "aggregation, expected",
    [
        ("last", [3.0, 6.0]),
        ("sum", [4.0, 8.0]),
        ("min", [1.0, 2.0]),
        ("max", [3.0, 6.0]),
        ("mean", [2.0, 4.0]),
        ("bad", [3.0, 6.0]),
    ],
)
def test_daily_values_for_entity_applies_aggregation(aggregation, expected):
    values = daily_values_for_entity(make_repeated_daily_history(), "sensor.a", aggregation)

    assert values.tolist() == expected


@pytest.mark.parametrize(
    "time_start, time_end, expected",
    [
        ("00:00", "12:00", [1.0, 2.0]),
        ("18:00", "23:00", [3.0, 6.0]),
        ("21:00", "02:00", [1.0, 2.0]),
    ],
)
def test_daily_values_for_entity_filters_time_windows(time_start, time_end, expected):
    values = daily_values_for_entity(
        make_repeated_daily_history(),
        "sensor.a",
        "last",
        0,
        time_start,
        time_end,
    )

    assert values.tolist() == expected


def test_daily_values_for_entity_day_offset_shifts_to_prior_day():
    values = daily_values_for_entity(make_repeated_daily_history(), "sensor.a", "last", day_offset=1)

    assert pd.isna(values.iloc[0])
    assert values.iloc[1] == 3.0


def test_prior_panas_entity_uses_shifted_current_panas_source():
    history = build_demo_history(DEFAULT_CONFIG)
    current = daily_values_for_entity(history, PRIOR_PANAS_SOURCE_ENTITY, "last", 0)
    prior = daily_values_for_entity(history, PRIOR_PANAS_ENTITY, "last", 1)

    pd.testing.assert_series_equal(prior.dropna(), current.shift(1).dropna(), check_names=False)


@pytest.mark.parametrize("days", [14, 30, 90])
def test_daily_sensor_values_limits_to_requested_window(days):
    history = build_demo_history({**DEFAULT_CONFIG, "analysis_window_days": 120})

    values = daily_sensor_values(history, PRIOR_PANAS_SOURCE_ENTITY, days)

    assert len(values) == days
    assert values.index.is_monotonic_increasing


def test_prepare_causal_dataset_matches_default_panas_sleep_steps_example():
    config = copy.deepcopy(DEFAULT_CONFIG)
    history = build_demo_history(config)

    dataset = prepare_causal_dataset(history, config)

    assert 80 <= len(dataset) <= 90
    assert {"outcome", "exposure", "exposure_raw", "sensor.nick_r_steps", PRIOR_PANAS_ENTITY}.issubset(dataset.columns)
    assert dataset["outcome"].between(10, 50).all()
    assert set(dataset["exposure"].unique()).issubset({0.0, 1.0})
    assert dataset["exposure"].nunique() == 2
    assert dataset.isna().sum().sum() == 0


def test_prepare_dataset_uses_default_outcome_and_predictors():
    config = copy.deepcopy(DEFAULT_CONFIG)
    history = build_demo_history(config)

    dataset = prepare_dataset(history, config)

    assert "sensor.nick_r_mood_score" in dataset.columns
    assert "sensor.nick_r_steps" in dataset.columns
    assert PRIOR_PANAS_ENTITY in dataset.columns
    assert dataset.isna().sum().sum() == 0


@pytest.mark.parametrize(
    "operator, threshold, expected",
    [
        ("less_than", 33, [True, False, False]),
        ("more_than", 33, [False, False, True]),
    ],
)
def test_build_binary_outcome_respects_threshold_direction(operator, threshold, expected):
    config = {**DEFAULT_CONFIG, "outcome_operator": operator, "outcome_threshold": threshold}

    outcome = build_binary_outcome(pd.Series([20, 33, 40]), config)

    assert outcome.tolist() == expected


@pytest.mark.parametrize(
    "operator, threshold, expected",
    [
        ("more_than", 420, [False, False, True]),
        ("less_than", 420, [True, False, False]),
    ],
)
def test_build_binary_exposure_respects_threshold_direction(operator, threshold, expected):
    config = {**DEFAULT_CONFIG, "exposure_operator": operator, "exposure_threshold": threshold}

    exposure = build_binary_exposure(pd.Series([300, 420, 500]), config)

    assert exposure.tolist() == expected


@pytest.mark.parametrize(
    "values, expected",
    [
        (pd.Series([1, 2, 3, 4]), 2.5),
        (pd.Series([5, 5, 5]), None),
        (pd.Series([], dtype=float), None),
    ],
)
def test_balanced_binary_threshold(values, expected):
    assert balanced_binary_threshold(values) == expected


def test_apply_balanced_exposure_threshold_updates_config_when_possible():
    config = copy.deepcopy(DEFAULT_CONFIG)
    dataset = pd.DataFrame({"exposure_raw": [1, 2, 3, 4]})

    changed = apply_balanced_exposure_threshold(dataset, config)

    assert changed is True
    assert config["exposure_operator"] == "more_than"
    assert config["exposure_threshold"] == 2.5


def test_outcome_histogram_uses_panas_scale_and_no_real_sensor_error():
    histogram = build_outcome_histogram(copy.deepcopy(DEFAULT_CONFIG))

    assert histogram["ok"] is True
    assert 10 <= histogram["minimum"] <= histogram["maximum"] <= 50
    assert histogram["requested_days"] == 90
    assert "Sensor not found" not in str(histogram)
    assert "Home Assistant" not in str(histogram)


@pytest.mark.parametrize(
    "entity, aggregation, day_offset",
    [
        ("sensor.nick_r_steps", "max", 1),
        (PRIOR_PANAS_ENTITY, "last", 1),
        ("sensor.nick_r_sleep_minutes_asleep", "last", 1),
    ],
)
def test_covariate_histograms_render_for_default_simulated_variables(entity, aggregation, day_offset):
    config = copy.deepcopy(DEFAULT_CONFIG)
    histogram = build_covariate_histogram(
        config,
        {"entity": entity, "aggregation": aggregation, "day_offset": day_offset},
    )

    assert histogram["ok"] is True
    assert histogram["rows"] > 0
    assert histogram["requested_days"] == 90
    assert histogram["association"]["available_days"] > 0
    assert "Sensor not found" not in str(histogram)


def test_series_histogram_reports_continuous_distribution_and_association():
    values = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2026-01-01", periods=5))
    outcome = pd.Series([False, False, True, True, True], index=values.index)

    histogram = build_series_histogram(values, "sensor.a", 25, outcome=outcome)

    assert histogram["ok"] is True
    assert histogram["rows"] == 5
    assert histogram["association"]["available_days"] == 5
    assert histogram["association"]["correlation"] is not None
    assert sum(row["count"] for row in histogram["bins"]) == 5


def test_binary_level_histogram_reports_two_levels():
    values = pd.Series([0, 1, 1, 0], index=pd.date_range("2026-01-01", periods=4))
    outcome = pd.Series([False, True, True, False], index=values.index)

    histogram = build_binary_level_histogram(values, "sensor.binary", 4, outcome=outcome)

    assert histogram["kind"] == "binary"
    assert [row["label"] for row in histogram["bins"]] == ["Off / closed / 0", "On / open / 1"]
    assert [row["count"] for row in histogram["bins"]] == [2, 2]
    assert histogram["statistics"]["on_percent"] == 50.0


@pytest.mark.parametrize(
    "values, outcome, expected_days, expected_correlation",
    [
        (pd.Series([1, 2, 3]), pd.Series([0, 1, 1]), 3, pytest.approx(0.866, abs=0.01)),
        (pd.Series([1, 1, 1]), pd.Series([0, 1, 1]), 3, None),
        (pd.Series([1]), pd.Series([1]), 1, None),
        (pd.Series([1, 2]), None, 0, None),
    ],
)
def test_histogram_association_handles_available_and_unavailable_correlations(values, outcome, expected_days, expected_correlation):
    association = build_histogram_association(values, outcome)

    assert association["available_days"] == expected_days
    if expected_correlation is None:
        assert association["correlation"] is None
    else:
        assert association["correlation"] == expected_correlation


def test_search_home_assistant_entities_is_disabled_for_simulated_demo():
    assert search_home_assistant_entities("sensor") == []


def test_dag_node_histogram_route_never_returns_sensor_not_found_for_unknown_real_sensor(isolated_client):
    response = isolated_client.get(
        "/dag/node-histogram",
        query_string={"sensor": "sensor.real_blood_pressure", "analysis_window_days": "90"},
    )

    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert data["ok"] is False
    assert "Sensor not found" not in data["message"]
    assert "Home Assistant" not in data["message"]

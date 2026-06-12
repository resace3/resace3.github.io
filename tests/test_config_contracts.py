import copy

import pandas as pd
import pytest

from sleep_causal_app.main import (
    DEFAULT_CONFIG,
    PRIOR_PANAS_ENTITY,
    PRIOR_PANAS_SOURCE_ENTITY,
    SIMULATED_SENSOR_MAPS,
    build_analysis_preview,
    build_distribution_statistics,
    default_demo_causal_dag,
    ensure_list,
    filter_history_time_window,
    histogram_bin_count,
    is_derived_entity,
    normalize_aggregation,
    normalize_analysis_method,
    normalize_config,
    normalize_dag_role,
    normalize_day_offset,
    normalize_missing_strategy,
    normalize_node_value_type,
    normalize_outcome_operator,
    normalize_sensor_maps,
    normalize_sensor_role,
    normalize_threshold_operator,
    normalize_time_text,
    normalize_user_sensor_maps,
    parse_float,
    parse_int,
    parse_optional_float,
    parse_time_minutes,
    restore_default_demo_dag,
    source_entity_id,
    split_entities,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        ([" sensor.a ", "", "sensor.b"], ["sensor.a", "sensor.b"]),
        ("sensor.a,sensor.b", ["sensor.a", "sensor.b"]),
        ("sensor.a\nsensor.b", ["sensor.a", "sensor.b"]),
        ("  ", []),
    ],
)
def test_ensure_list_normalizes_values(value, expected):
    assert ensure_list(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("sensor.a,sensor.b", ["sensor.a", "sensor.b"]),
        ("sensor.a\nsensor.b", ["sensor.a", "sensor.b"]),
        ("sensor.a,\n sensor.b ,, sensor.c", ["sensor.a", "sensor.b", "sensor.c"]),
        ("", []),
    ],
)
def test_split_entities_accepts_commas_and_newlines(value, expected):
    assert split_entities(value) == expected


@pytest.mark.parametrize(
    "entity, source, derived",
    [
        (PRIOR_PANAS_ENTITY, PRIOR_PANAS_SOURCE_ENTITY, True),
        (PRIOR_PANAS_SOURCE_ENTITY, PRIOR_PANAS_SOURCE_ENTITY, False),
        ("sensor.nick_r_steps", "sensor.nick_r_steps", False),
    ],
)
def test_derived_source_entity_contract(entity, source, derived):
    assert source_entity_id(entity) == source
    assert is_derived_entity(entity) is derived


@pytest.mark.parametrize(
    "value, expected",
    [
        ("sum", "sum"),
        ("min", "min"),
        ("max", "max"),
        ("mean", "mean"),
        ("last", "last"),
        ("median", "last"),
    ],
)
def test_normalize_aggregation(value, expected):
    assert normalize_aggregation(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1:05", "01:05"),
        ("23:59", "23:59"),
        ("00:00", "00:00"),
        ("24:00", ""),
        ("10:61", ""),
        ("not-time", ""),
        ("07:03:99", "07:03"),
    ],
)
def test_normalize_time_text(value, expected):
    assert normalize_time_text(value) == expected


@pytest.mark.parametrize(
    "value, fallback, expected",
    [
        ("00:00", 12, 0),
        ("01:30", 12, 90),
        ("23:59", 12, 1439),
        ("bad", 12, 12),
    ],
)
def test_parse_time_minutes(value, fallback, expected):
    assert parse_time_minutes(value, fallback) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1", 1),
        (1, 1),
        ("0", 0),
        ("2", 0),
        ("", 0),
    ],
)
def test_normalize_day_offset_is_current_or_previous_day(value, expected):
    assert normalize_day_offset(value) == expected


@pytest.mark.parametrize(
    "func, value, expected",
    [
        (normalize_sensor_role, "outcome", "outcome"),
        (normalize_sensor_role, "exposure", "covariate"),
        (normalize_sensor_role, "covariate", "covariate"),
        (normalize_dag_role, "exposure", "exposure"),
        (normalize_dag_role, "outcome", "outcome"),
        (normalize_dag_role, "covariate", "covariate"),
        (normalize_dag_role, "latent", "latent"),
        (normalize_dag_role, "bad", "covariate"),
        (normalize_outcome_operator, "more_than", "more_than"),
        (normalize_outcome_operator, "less_than", "less_than"),
        (normalize_threshold_operator, "less_than", "less_than"),
        (normalize_threshold_operator, "more_than", "more_than"),
        (normalize_node_value_type, "binary_threshold", "binary_threshold"),
        (normalize_node_value_type, "free_text", "continuous"),
        (normalize_missing_strategy, "interpolate", "interpolate"),
        (normalize_missing_strategy, "forward_fill", "forward_fill"),
        (normalize_missing_strategy, "zero_fill", "drop"),
        (normalize_analysis_method, "g_formula", "g_formula"),
        (normalize_analysis_method, "ipw", "ipw"),
        (normalize_analysis_method, "doubly_robust", "doubly_robust"),
        (normalize_analysis_method, "dag_auto", "dag_auto"),
        (normalize_analysis_method, "unknown", "g_formula"),
    ],
)
def test_normalizers_keep_supported_values_and_fallbacks(func, value, expected):
    assert func(value) == expected


@pytest.mark.parametrize(
    "func, args, expected",
    [
        (parse_int, ("7", 1), 7),
        (parse_int, ("7.5", 1), 1),
        (parse_int, (None, 4), 4),
        (parse_float, ("7.5", 1), 7.5),
        (parse_float, ("bad", 1.25), 1.25),
        (parse_float, (None, 2.5), 2.5),
        (parse_optional_float, ("7.5",), 7.5),
        (parse_optional_float, ("",), None),
        (parse_optional_float, (None,), None),
        (parse_optional_float, ("bad",), None),
    ],
)
def test_numeric_parsers(func, args, expected):
    assert func(*args) == expected


@pytest.mark.parametrize(
    "requested_days, expected",
    [
        (0, 5),
        (14, 5),
        (25, 5),
        (90, 18),
        (730, 146),
    ],
)
def test_histogram_bin_count_scales_with_window(requested_days, expected):
    assert histogram_bin_count(requested_days) == expected


@pytest.mark.parametrize(
    "values, expected",
    [
        (pd.Series([1, 2, 3]), {"mean": 2.0, "median": 2.0, "minimum": 1.0, "maximum": 3.0}),
        (pd.Series([0, 1, 1, 0, 1]), {"on_count": 3, "off_count": 2, "on_percent": 60.0}),
        (pd.Series(["bad", None]), {}),
    ],
)
def test_build_distribution_statistics(values, expected):
    statistics = build_distribution_statistics(values)

    for key, value in expected.items():
        assert statistics[key] == value
    if statistics:
        assert [row["label"] for row in statistics["percentiles"]] == [
            "P5",
            "P10",
            "P25",
            "P50",
            "P75",
            "P90",
            "P95",
        ]


def test_normalize_sensor_maps_keeps_only_simulated_variables():
    maps = normalize_sensor_maps(
        [
            {"sensor": "sensor.real_world_bp", "description": "real", "role": "outcome"},
            {"sensor": "sensor.nick_r_steps", "description": "Daily steps", "role": "outcome"},
            {"sensor": PRIOR_PANAS_ENTITY, "description": "", "role": "bad"},
        ]
    )

    assert [row["sensor"] for row in maps] == [row["sensor"] for row in SIMULATED_SENSOR_MAPS]
    steps = next(row for row in maps if row["sensor"] == "sensor.nick_r_steps")
    prior_panas = next(row for row in maps if row["sensor"] == PRIOR_PANAS_ENTITY)
    assert steps == {"sensor": "sensor.nick_r_steps", "description": "Daily steps", "role": "outcome"}
    assert prior_panas["description"] == "Prior-day PANAS Score derived from the simulated PANAS Score."
    assert prior_panas["role"] == "covariate"


def test_normalize_user_sensor_maps_rejects_arbitrary_real_sensors():
    maps = normalize_user_sensor_maps(
        [
            {"sensor": "sensor.nick_r_steps", "description": "Steps", "role": "covariate"},
            {"sensor": "sensor.real_bp", "description": "Real BP", "role": "outcome"},
            {"sensor": "sensor.nick_r_steps", "description": "Duplicate", "role": "outcome"},
        ]
    )

    assert {row["sensor"] for row in maps} == {row["sensor"] for row in SIMULATED_SENSOR_MAPS}
    assert next(row for row in maps if row["sensor"] == "sensor.nick_r_steps")["description"] == "Steps"
    assert "sensor.real_bp" not in {row["sensor"] for row in maps}


def test_normalize_config_clamps_values_and_preserves_demo_contract():
    config = normalize_config(
        {
            "analysis_window_days": "4",
            "max_lag_days": "99",
            "state_forgetting": "0.2",
            "analysis_method": "unknown",
            "sensor_maps": [{"sensor": "sensor.real", "description": "Real", "role": "outcome"}],
        }
    )

    assert config["analysis_window_days"] == 14
    assert config["max_lag_days"] == 21
    assert config["state_forgetting"] == 0.90
    assert config["analysis_method"] == "g_formula"
    assert [row["sensor"] for row in config["sensor_maps"]] == [row["sensor"] for row in SIMULATED_SENSOR_MAPS]
    assert config["sleep_sensor"] == "sensor.nick_r_mood_score"
    assert config["exposure_sensor"] == "sensor.nick_r_sleep_minutes_asleep"
    assert config["predictor_entities"] == [PRIOR_PANAS_ENTITY, "sensor.nick_r_steps"]


def test_default_demo_causal_dag_has_expected_nodes_and_forward_edges():
    dag = default_demo_causal_dag()

    assert [node["label"] for node in dag["nodes"]] == [
        "PANAS Score (t-1)",
        "Sleep",
        "Steps",
        "PANAS Score",
    ]
    assert {node["role"] for node in dag["nodes"]} == {"covariate", "exposure", "outcome"}
    assert {(edge["source"], edge["target"]) for edge in dag["edges"]} == {
        (PRIOR_PANAS_ENTITY, "sensor.nick_r_sleep_minutes_asleep"),
        (PRIOR_PANAS_ENTITY, "sensor.nick_r_steps"),
        ("sensor.nick_r_sleep_minutes_asleep", "sensor.nick_r_mood_score"),
        ("sensor.nick_r_steps", "sensor.nick_r_mood_score"),
    }
    node_by_id = {node["id"]: node for node in dag["nodes"]}
    for edge in dag["edges"]:
        assert node_by_id[edge["source"]]["x"] < node_by_id[edge["target"]]["x"]


def test_restore_default_demo_dag_replaces_custom_graph_and_updates_settings():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["causal_dag"] = {
        "nodes": [{"id": "sensor.fake", "label": "Fake", "role": "outcome"}],
        "edges": [],
    }

    restored = restore_default_demo_dag(config)

    assert [node["label"] for node in restored["causal_dag"]["nodes"]] == [
        "PANAS Score (t-1)",
        "Sleep",
        "Steps",
        "PANAS Score",
    ]
    assert restored["sleep_sensor"] == "sensor.nick_r_mood_score"
    assert restored["exposure_sensor"] == "sensor.nick_r_sleep_minutes_asleep"
    assert restored["exposure_threshold"] == 420
    assert restored["outcome_threshold"] == 33


@pytest.mark.parametrize(
    "start, end, expected_states",
    [
        ("09:00", "17:00", [2.0, 3.0]),
        ("23:00", "02:00", [1.0, 4.0]),
        ("", "", [1.0, 2.0, 3.0, 4.0]),
    ],
)
def test_filter_history_time_window_handles_daytime_and_overnight_windows(start, end, expected_states):
    history = pd.DataFrame(
        {
            "last_changed": pd.to_datetime(
                [
                    "2026-01-01 01:00Z",
                    "2026-01-01 09:30Z",
                    "2026-01-01 16:30Z",
                    "2026-01-01 23:30Z",
                ]
            ),
            "state": [1.0, 2.0, 3.0, 4.0],
        }
    )

    filtered = filter_history_time_window(history, start, end)

    assert filtered["state"].tolist() == expected_states


@pytest.mark.parametrize(
    "overrides, expected_ready, expected_feature_count",
    [
        ({}, True, 2),
        ({"sleep_sensor": ""}, False, 2),
        ({"exposure_sensor": ""}, False, 2),
        ({"predictor_entities": ["sensor.nick_r_steps", "sensor.nick_r_steps"]}, True, 1),
    ],
)
def test_build_analysis_preview_reports_readiness_and_unique_features(overrides, expected_ready, expected_feature_count):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(overrides)

    preview = build_analysis_preview(config)

    assert preview["ready"] is expected_ready
    assert preview["feature_count"] == expected_feature_count

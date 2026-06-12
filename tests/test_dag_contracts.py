import copy

import pytest
from werkzeug.datastructures import MultiDict

from sleep_causal_app.main import (
    DEFAULT_CONFIG,
    PRIOR_PANAS_ENTITY,
    apply_analysis_settings_from_dag,
    find_causal_dag_cycle,
    form_predictor_settings,
    normalize_causal_dag,
    parse_dag_form,
)


def dag_form(**overrides):
    data = {
        "node_id": [" sensor.a ", "sensor.b", "sensor.c", "sensor.a", ""],
        "node_label": ["", "B label", "C label", "Duplicate", "Blank"],
        "node_description": ["A desc", "", "C desc", "", ""],
        "node_role": ["outcome", "exposure", "invalid", "latent", "covariate"],
        "node_x": ["bad", "350", "1000", "1", "2"],
        "node_y": ["20", "120", "600", "1", "2"],
        "node_aggregation": ["sum", "max", "median", "last", "last"],
        "node_time_start": ["7:00", "bad", "23:30", "", ""],
        "node_time_end": ["8:15", "18:00", "25:00", "", ""],
        "node_day_lag": ["30", "1", "-1", "0", "0"],
        "node_missing_strategy": ["interpolate", "forward_fill", "bad", "drop", "drop"],
        "node_min_valid": ["0", "", "bad", "", ""],
        "node_max_valid": ["10.5", "", "99", "", ""],
        "node_rolling_days": ["0", "7", "99", "1", "1"],
        "node_value_type": ["binary_threshold", "continuous", "bad", "continuous", "continuous"],
        "node_threshold_operator": ["less_than", "more_than", "bad", "more_than", "more_than"],
        "node_threshold": ["33", "", "bad", "", ""],
        "edge_source": ["sensor.a", "sensor.b", "sensor.b", "sensor.a", "sensor.c"],
        "edge_target": ["sensor.b", "sensor.b", "sensor.missing", "sensor.b", "sensor.a"],
        "edge_label": ["a-b", "self", "missing", "a-b", "c-a"],
    }
    data.update(overrides)
    pairs = []
    for key, values in data.items():
        for value in values:
            pairs.append((key, value))
    return MultiDict(pairs)


def test_parse_dag_form_normalizes_nodes_and_filters_edges():
    dag = parse_dag_form(dag_form())

    assert [node["id"] for node in dag["nodes"]] == ["sensor.a", "sensor.b", "sensor.c"]
    assert dag["nodes"][0] == {
        "id": "sensor.a",
        "label": "sensor.a",
        "description": "A desc",
        "role": "outcome",
        "x": 120,
        "y": 64,
        "aggregation": "sum",
        "time_start": "07:00",
        "time_end": "08:15",
        "day_lag": 21,
        "missing_strategy": "interpolate",
        "min_valid": 0.0,
        "max_valid": 10.5,
        "rolling_days": 1,
        "value_type": "binary_threshold",
        "threshold_operator": "less_than",
        "threshold": 33.0,
    }
    assert dag["nodes"][1]["label"] == "B label"
    assert dag["nodes"][1]["time_start"] == ""
    assert dag["nodes"][1]["time_end"] == "18:00"
    assert dag["nodes"][1]["rolling_days"] == 7
    assert dag["nodes"][2]["role"] == "covariate"
    assert dag["nodes"][2]["aggregation"] == "last"
    assert dag["nodes"][2]["day_lag"] == 0
    assert dag["nodes"][2]["rolling_days"] == 30
    assert dag["edges"] == [
        {"source": "sensor.a", "target": "sensor.b", "label": "a-b"},
        {"source": "sensor.c", "target": "sensor.a", "label": "c-a"},
    ]


@pytest.mark.parametrize(
    "edges, has_cycle",
    [
        ([("a", "b"), ("b", "c")], False),
        ([("a", "b"), ("b", "a")], True),
        ([("a", "b"), ("b", "c"), ("c", "a")], True),
        ([("a", "b"), ("c", "d"), ("d", "c")], True),
        ([("a", "missing"), ("b", "c")], False),
        ([], False),
    ],
)
def test_find_causal_dag_cycle_detects_directed_cycles(edges, has_cycle):
    dag = {
        "nodes": [{"id": node_id} for node_id in ["a", "b", "c", "d"]],
        "edges": [{"source": source, "target": target} for source, target in edges],
    }

    cycle = find_causal_dag_cycle(dag)

    assert bool(cycle) is has_cycle
    if cycle:
        assert cycle[0] == cycle[-1]


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"id": "sensor.a", "x": -20, "y": 1}, {"x": 0, "y": 64}),
        ({"id": "sensor.a", "x": 9999, "y": 9999}, {"x": 920, "y": 520}),
        ({"id": "sensor.a", "role": "bad"}, {"role": "covariate"}),
        ({"id": "sensor.a", "aggregation": "median"}, {"aggregation": "last"}),
        ({"id": "sensor.a", "day_lag": 99}, {"day_lag": 21}),
        ({"id": "sensor.a", "rolling_days": 0}, {"rolling_days": 1}),
        ({"id": "sensor.a", "rolling_days": 99}, {"rolling_days": 30}),
        ({"id": "sensor.a", "threshold": "bad"}, {"threshold": None}),
    ],
)
def test_normalize_causal_dag_clamps_node_fields(row, expected):
    dag = normalize_causal_dag({"nodes": [row], "edges": []})

    for key, value in expected.items():
        assert dag["nodes"][0][key] == value


def test_normalize_causal_dag_filters_invalid_and_duplicate_edges():
    dag = normalize_causal_dag(
        {
            "nodes": [{"id": "sensor.a"}, {"id": "sensor.b"}],
            "edges": [
                {"source": "sensor.a", "target": "sensor.b", "label": "valid"},
                {"source": "sensor.a", "target": "sensor.b", "label": "valid"},
                {"source": "sensor.b", "target": "sensor.b", "label": "self"},
                {"source": "sensor.a", "target": "sensor.missing", "label": "missing"},
            ],
        }
    )

    assert dag["edges"] == [{"source": "sensor.a", "target": "sensor.b", "label": "valid"}]


def test_apply_analysis_settings_from_dag_uses_outcome_exposure_and_covariates():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["causal_dag"] = {
        "nodes": [
            {
                "id": "sensor.outcome",
                "label": "Outcome",
                "role": "outcome",
                "threshold_operator": "more_than",
                "threshold": "12.5",
            },
            {
                "id": "sensor.exposure",
                "label": "Exposure",
                "role": "exposure",
                "aggregation": "mean",
                "threshold_operator": "less_than",
                "threshold": "8",
            },
            {"id": "sensor.covariate", "role": "covariate", "aggregation": "max", "day_lag": "4"},
            {"id": PRIOR_PANAS_ENTITY, "role": "covariate", "aggregation": "last", "day_lag": "1"},
            {"id": "sensor.exposure", "role": "covariate", "aggregation": "sum", "day_lag": "1"},
            {"id": "draft_node", "role": "outcome"},
        ],
        "edges": [],
    }

    apply_analysis_settings_from_dag(config)

    assert config["sleep_sensor"] == "sensor.outcome"
    assert config["outcome_operator"] == "more_than"
    assert config["outcome_threshold"] == 12.5
    assert config["exposure_sensor"] == "sensor.exposure"
    assert config["exposure_aggregation"] == "mean"
    assert config["exposure_operator"] == "less_than"
    assert config["exposure_threshold"] == 8.0
    assert config["predictor_settings"] == [
        {"entity": "sensor.covariate", "aggregation": "max", "day_offset": 4},
        {"entity": PRIOR_PANAS_ENTITY, "aggregation": "last", "day_offset": 1},
    ]


def test_apply_analysis_settings_from_dag_clears_missing_outcome_or_exposure():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["causal_dag"] = {"nodes": [{"id": "draft", "role": "outcome"}], "edges": []}

    apply_analysis_settings_from_dag(config)

    assert config["sleep_sensor"] == ""
    assert config["exposure_sensor"] == ""
    assert config["predictor_settings"] == []


@pytest.mark.parametrize(
    "entities, aggregations, offsets, expected",
    [
        (["sensor.a"], ["sum"], ["1"], [{"entity": "sensor.a", "aggregation": "sum", "day_offset": 1}]),
        (["sensor.a"], [], [], [{"entity": "sensor.a", "aggregation": "last", "day_offset": 0}]),
        ([" sensor.a ", ""], ["mean", "max"], ["0", "1"], [{"entity": "sensor.a", "aggregation": "mean", "day_offset": 0}]),
        (["sensor.a", "sensor.b"], ["median", "max"], ["2", "1"], [{"entity": "sensor.a", "aggregation": "last", "day_offset": 0}, {"entity": "sensor.b", "aggregation": "max", "day_offset": 1}]),
        ([], [], [], []),
    ],
)
def test_form_predictor_settings_aligns_repeated_form_fields(entities, aggregations, offsets, expected):
    form = MultiDict()
    for value in entities:
        form.add("predictor_entities", value)
    for value in aggregations:
        form.add("predictor_aggregations", value)
    for value in offsets:
        form.add("predictor_day_offsets", value)

    assert form_predictor_settings(form) == expected

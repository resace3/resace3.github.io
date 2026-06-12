import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from html import escape as html_escape
from pathlib import Path
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests
from flask import Flask, Response, abort, flash, jsonify, redirect, render_template, request, url_for


DEFAULT_CONFIG = {
    "sleep_sensor": "sensor.nick_r_mood_score",
    "exposure_sensor": "sensor.nick_r_sleep_minutes_asleep",
    "exposure_aggregation": "last",
    "exposure_operator": "more_than",
    "exposure_threshold": 420,
    "outcome_operator": "less_than",
    "outcome_threshold": 33,
    "sleep_start_sensor": "sensor.nick_r_sleep_start_time",
    "exclude_before_sleep_start_hours": 0,
    "exclude_before_sleep_start_minutes": 0,
    "predictor_entities": [
        "sensor.nick_r_steps",
    ],
    "predictor_settings": [
        {
            "entity": "sensor.nick_r_steps",
            "aggregation": "max",
            "day_offset": 1,
        },
    ],
    "cross_validation_entities": [],
    "analysis_window_days": 90,
    "analysis_method": "g_formula",
    "lag_days": 1,
    "max_lag_days": 7,
    "state_forgetting": 0.985,
    "saved_analyses": [],
    "causal_dag": {
        "nodes": [
            {
                "id": "sensor.nick_r_sleep_minutes_asleep",
                "label": "Sleep duration",
                "role": "exposure",
                "x": 80,
                "y": 140,
                "aggregation": "last",
                "day_lag": 1,
                "value_type": "binary_threshold",
                "threshold_operator": "more_than",
                "threshold": 420,
            },
            {
                "id": "sensor.nick_r_steps",
                "label": "Steps",
                "role": "covariate",
                "x": 80,
                "y": 300,
                "aggregation": "max",
                "day_lag": 1,
                "value_type": "binary_threshold",
                "threshold_operator": "more_than",
                "threshold": 8000,
            },
            {
                "id": "sensor.nick_r_mood_score",
                "label": "PANAS Score",
                "role": "outcome",
                "x": 520,
                "y": 210,
                "aggregation": "last",
                "day_lag": 0,
                "value_type": "binary_threshold",
                "threshold_operator": "less_than",
                "threshold": 33,
            },
        ],
        "edges": [
            {"source": "sensor.nick_r_sleep_minutes_asleep", "target": "sensor.nick_r_mood_score", "label": "effect"},
            {"source": "sensor.nick_r_steps", "target": "sensor.nick_r_sleep_minutes_asleep", "label": "activity-sleep"},
            {"source": "sensor.nick_r_steps", "target": "sensor.nick_r_mood_score", "label": "adjust"},
        ],
    },
    "sensor_maps": [
        {
            "sensor": "sensor.nick_r_mood_score",
            "description": "Daily PANAS summed score on a 10 to 50 scale, where lower values indicate lower affect.",
            "role": "outcome",
        },
        {
            "sensor": "sensor.nick_r_awakenings_count",
            "description": "Number of recorded awakenings during a sleep session.",
            "role": "outcome",
        },
        {
            "sensor": "sensor.nick_r_sleep_minutes_awake",
            "description": "Minutes recorded awake during the sleep window.",
            "role": "outcome",
        },
        {
            "sensor": "sensor.nick_r_sleep_time_in_bed",
            "description": "Total minutes recorded in bed.",
            "role": "outcome",
        },
        {
            "sensor": "sensor.nick_r_sleep_minutes_asleep",
            "description": "Total minutes asleep during the prior sleep period.",
            "role": "covariate",
        },
        {
            "sensor": "sensor.nick_r_sleep_minutes_to_fall_asleep",
            "description": "Minutes recorded before falling asleep.",
            "role": "outcome",
        },
        {
            "sensor": "sensor.nick_r_sleep_efficiency",
            "description": "Sleep efficiency percentage from the sleep tracker.",
            "role": "outcome",
        },
        {
            "sensor": "sensor.nick_r_sleep_start_time",
            "description": "Reported sleep start time.",
            "role": "outcome",
        },
        {
            "sensor": "binary_sensor.pantry_door_window",
            "description": (
                "This is my pantry door that I get food out of. On means I opened it, "
                "Off means I closed it"
            ),
            "role": "covariate",
        },
        {
            "sensor": "sensor.nick_r_steps",
            "description": "The number of steps I take a day. It is cumulative for each day.",
            "role": "covariate",
        },
        {
            "sensor": "sensor.nutribullet_plug_current",
            "description": (
                "the number of amps going through my nutribullet. A value over 1 means "
                "I made a smoothie at that time"
            ),
            "role": "covariate",
        },
        {
            "sensor": "binary_sensor.bthome_sensor_ee5c_window",
            "description": (
                "This is my refrigerator door that I get food out of in my room. On means "
                "I opened it, Off means I closed it"
            ),
            "role": "covariate",
        },
    ],
}


def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "sleep-analysis-dev")
    config_path = Path(os.environ.get("SLEEP_ANALYSIS_CONFIG", "/config/sleep_analysis.json"))

    @app.context_processor
    def static_asset_versions():
        def static_version(filename):
            path = Path(app.static_folder) / filename
            try:
                return int(path.stat().st_mtime)
            except OSError:
                return 0

        return {
            "static_version": static_version,
            "analysis_methods": analysis_method_options(),
        }

    @app.after_request
    def prevent_browser_cache(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.template_filter("hours_label")
    def hours_label(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        return f"{number / 60:.2f} hr"

    @app.get("/")
    def index():
        config = load_config(config_path)
        preview = build_analysis_preview(config)
        return render_template(
            "index.html", active_tab="analysis", config=config, preview=preview
        )

    @app.get("/sensor-maps")
    def sensor_maps():
        config = load_config(config_path)
        return render_template(
            "sensor_maps.html", active_tab="sensor_maps", config=config
        )

    @app.get("/saved-analyses")
    def saved_analyses():
        config = load_config(config_path)
        return render_template(
            "saved_analyses.html", active_tab="saved_analyses", config=config
        )

    @app.get("/saved-analyses/<analysis_id>")
    def saved_analysis_report(analysis_id):
        config = load_config(config_path)
        analysis = find_saved_analysis(config, analysis_id)
        if analysis is None:
            abort(404)
        return render_template(
            "saved_analysis_report.html",
            active_tab="saved_analyses",
            analysis=analysis,
            report=analysis_report_snapshot(analysis),
        )

    @app.get("/saved-analyses/<analysis_id>/pdf")
    def download_saved_analysis_pdf(analysis_id):
        config = load_config(config_path)
        analysis = find_saved_analysis(config, analysis_id)
        if analysis is None:
            abort(404)
        report = analysis_report_snapshot(analysis)
        pdf = build_saved_analysis_pdf(report)
        filename = f"causal-analysis-engine-{analysis_id}.pdf"
        return Response(
            pdf,
            mimetype="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/saved-analyses/<analysis_id>/delete")
    def delete_saved_analysis(analysis_id):
        config = load_config(config_path)
        before = len(config.get("saved_analyses", []))
        config["saved_analyses"] = [
            row for row in normalize_saved_analyses(config.get("saved_analyses")) if row.get("id") != analysis_id
        ]
        if len(config["saved_analyses"]) == before:
            abort(404)
        save_config(config_path, config)
        flash("Saved analysis deleted.")
        return redirect(url_for("saved_analyses"))

    @app.get("/dag")
    def causal_dag():
        return redirect(url_for("index"))

    @app.post("/dag/save")
    def save_causal_dag():
        config = load_config(config_path)
        dag = parse_dag_form(request.form)
        cycle = find_causal_dag_cycle(dag)
        if cycle:
            flash(f"DAG not saved because it has a directed cycle: {' -> '.join(cycle)}.")
            return redirect(url_for("index"))
        config["causal_dag"] = dag
        apply_analysis_settings_from_dag(config)
        save_config(config_path, config)
        flash("DAG saved.")
        return redirect(url_for("index"))

    @app.post("/dag/autosave")
    def autosave_causal_dag():
        config = load_config(config_path)
        dag = parse_dag_form(request.form)
        cycle = find_causal_dag_cycle(dag)
        if cycle:
            return jsonify({"ok": False, "message": "DAG has a directed cycle.", "cycle": cycle}), 400
        config["causal_dag"] = dag
        apply_analysis_settings_from_dag(config)
        save_config(config_path, config)
        return jsonify({"ok": True})

    @app.get("/entities/search")
    def search_entities():
        query_text = request.args.get("q", "").strip()
        config = load_config(config_path)
        entities = search_home_assistant_entities(query_text)
        mapped = [
            {
                "entity_id": row["sensor"],
                "label": row["description"] or row["role"],
            }
            for row in config["sensor_maps"]
        ]
        merged = {row["entity_id"]: row for row in mapped + entities}
        if query_text:
            needle = query_text.lower()
            merged = {
                key: row
                for key, row in merged.items()
                if needle in key.lower() or needle in str(row.get("label", "")).lower()
            }
        return jsonify({"ok": True, "entities": list(merged.values())[:60]})

    @app.get("/dag/node-histogram")
    def dag_node_histogram():
        config = load_config(config_path)
        sensor = request.args.get("sensor", "").strip()
        if not sensor:
            return jsonify({"ok": False, "message": "Choose a sensor for this box.", "bins": []})
        if "." not in sensor:
            return jsonify({"ok": False, "message": "Histograms need a Home Assistant entity ID, such as sensor.example.", "bins": []})
        aggregation = normalize_aggregation(request.args.get("aggregation", "last"))
        time_start = request.args.get("time_start", "").strip()
        time_end = request.args.get("time_end", "").strip()
        day_lag = max(0, min(21, parse_int(request.args.get("day_lag"), 0)))
        days = max(14, min(730, parse_int(request.args.get("analysis_window_days"), config["analysis_window_days"])))
        history_config = {
            **config,
            "sleep_sensor": sensor,
            "predictor_entities": [],
            "predictor_settings": [],
            "cross_validation_entities": [],
            "sleep_start_sensor": "",
            "analysis_window_days": days,
        }
        try:
            history = fetch_home_assistant_history(history_config)
        except requests.RequestException:
            return jsonify({"ok": False, "message": "Home Assistant rejected history lookup for this entity.", "bins": []})
        if history.empty:
            return jsonify({"ok": False, "message": "No Home Assistant history was returned for this sensor.", "bins": []})
        sensor_history = history[history["entity_id"] == sensor].dropna(subset=["state"])
        sensor_history = filter_history_time_window(sensor_history, time_start, time_end)
        values = aggregate_daily_values(sensor_history, aggregation)
        if day_lag:
            values = values.shift(day_lag)
        if not values.empty:
            latest_day = values.index.max()
            start = latest_day - pd.Timedelta(days=days - 1)
            values = values[values.index >= start].dropna()
        if values.empty:
            return jsonify({"ok": False, "message": "No usable values matched this sensor and time window.", "bins": []})
        return jsonify(
            build_series_histogram(
                values,
                sensor,
                days,
                extra={
                    "aggregation": aggregation,
                    "time_start": time_start,
                    "time_end": time_end,
                    "day_lag": day_lag,
                },
            )
        )

    @app.get("/outcome-histogram")
    def outcome_histogram():
        config = load_config(config_path)
        sensor = request.args.get("sensor", config["sleep_sensor"]).strip()
        config["analysis_window_days"] = max(
            14,
            min(
                730,
                parse_int(request.args.get("analysis_window_days"), config["analysis_window_days"]),
            ),
        )
        config["sleep_sensor"] = sensor
        histogram = build_outcome_histogram(config)
        return jsonify(histogram)

    @app.get("/covariate-histogram")
    def covariate_histogram():
        config = load_config(config_path)
        sensor = request.args.get("sensor", "").strip()
        config["sleep_sensor"] = request.args.get("sleep_sensor", config["sleep_sensor"]).strip()
        config["outcome_operator"] = normalize_outcome_operator(
            request.args.get("outcome_operator", config["outcome_operator"])
        )
        config["outcome_threshold"] = parse_float(
            request.args.get("outcome_threshold"),
            config["outcome_threshold"],
        )
        config["analysis_window_days"] = max(
            14,
            min(
                730,
                parse_int(request.args.get("analysis_window_days"), config["analysis_window_days"]),
            ),
        )
        config["sleep_start_sensor"] = request.args.get(
            "sleep_start_sensor", config["sleep_start_sensor"]
        ).strip()
        config["exclude_before_sleep_start_minutes"] = max(
            0,
            min(
                1440,
                parse_int(
                    request.args.get("exclude_before_sleep_start_minutes"),
                    config["exclude_before_sleep_start_minutes"],
                ),
            ),
        )
        histogram = build_covariate_histogram(
            config,
            {
                "entity": sensor,
                "aggregation": normalize_aggregation(request.args.get("aggregation", "last")),
                "day_offset": normalize_day_offset(request.args.get("day_offset", "0")),
            },
        )
        return jsonify(histogram)

    @app.post("/sensor-maps/remove")
    def remove_sensor_map():
        config = load_config(config_path)
        sensor = request.form.get("sensor", "").strip()
        config["sensor_maps"] = [
            row for row in config["sensor_maps"] if row["sensor"] != sensor
        ]
        save_config(config_path, config)
        flash("Sensor removed.")
        return redirect(url_for("sensor_maps"))

    @app.post("/sensor-maps/save")
    def save_sensor_maps():
        config = load_config(config_path)
        sensors = request.form.getlist("sensor")
        descriptions = request.form.getlist("description")
        roles = request.form.getlist("role")
        config["sensor_maps"] = [
            {
                "sensor": sensor.strip(),
                "description": description.strip(),
                "role": normalize_sensor_role(role),
            }
            for sensor, description, role in zip(sensors, descriptions, roles)
        ]
        save_config(config_path, config)
        flash("Sensor maps saved.")
        return redirect(url_for("sensor_maps"))

    @app.post("/sensor-maps/add")
    def add_sensor_map():
        config = load_config(config_path)
        sensor = request.form.get("sensor", "").strip()
        description = request.form.get("description", "").strip()
        role = normalize_sensor_role(request.form.get("role", "covariate"))
        if not sensor:
            flash("Sensor entity is required.")
            return redirect(url_for("sensor_maps"))

        config["sensor_maps"] = [
            row for row in config["sensor_maps"] if row["sensor"] != sensor
        ]
        config["sensor_maps"].append(
            {"sensor": sensor, "description": description, "role": role}
        )
        save_config(config_path, config)
        flash("Sensor added.")
        return redirect(url_for("sensor_maps"))

    @app.post("/settings")
    def save_settings():
        config = load_config(config_path)
        config = apply_settings_form(config, request.form)
        save_config(config_path, config)
        flash("Settings saved.")
        return redirect(url_for("index"))

    @app.post("/settings/autosave")
    def autosave_settings():
        config = load_config(config_path)
        config = apply_settings_form(config, request.form)
        save_config(config_path, config)
        return jsonify({"ok": True})

    @app.post("/analyses/save")
    def save_analysis():
        config = load_config(config_path)
        result = run_n_of_1_analysis(config)
        if not result.get("ok"):
            flash(result.get("message", "Analysis could not be saved."))
            return redirect(url_for("index"))

        now = datetime.now(timezone.utc)
        snapshot = build_saved_analysis_snapshot(config, result, now)
        entry = {
            "id": now.strftime("%Y%m%d%H%M%S"),
            "name": request.form.get("name", "").strip() or snapshot["name"],
            "created_at": now.strftime("%Y-%m-%d %H:%M UTC"),
            "created_at_iso": now.isoformat(),
            "immutable": True,
            "snapshot": snapshot,
            "settings": saved_analysis_settings(config),
            "result": result,
            "outcome_label": result.get("outcome_label"),
            "rows": result.get("rows"),
            "cross_val_accuracy": result.get("cross_val_accuracy"),
        }
        config["saved_analyses"] = [entry] + normalize_saved_analyses(config.get("saved_analyses"))
        config["saved_analyses"] = config["saved_analyses"][:25]
        save_config(config_path, config)
        flash("Analysis saved.")
        return redirect(url_for("saved_analyses"))

    @app.post("/run")
    def run_analysis():
        config = load_config(config_path)
        if request.form:
            config = apply_settings_form(config, request.form)
            save_config(config_path, config)
        result = run_n_of_1_analysis(config)
        preview = build_analysis_preview(config)
        return render_template(
            "index.html",
            active_tab="analysis",
            config=config,
            preview=preview,
            result=result,
        )

    @app.post("/stepwise")
    def run_stepwise_selection():
        config = load_config(config_path)
        if request.form:
            config = apply_settings_form(config, request.form)
            save_config(config_path, config)
        result = run_stepwise_aic_selection(config)
        preview = build_analysis_preview(config)
        return render_template(
            "index.html",
            active_tab="analysis",
            config=config,
            preview=preview,
            result=result,
        )

    return app


def load_config(path):
    config = {**DEFAULT_CONFIG}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            config.update(json.load(handle))
    return normalize_config(config)


def save_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(normalize_config(config), handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def read_addon_options():
    options_path = Path("/data/options.json")
    if not options_path.exists():
        return {}
    with options_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_config(config):
    normalized = {**DEFAULT_CONFIG, **config}
    normalized["predictor_entities"] = ensure_list(normalized.get("predictor_entities"))
    normalized["predictor_settings"] = normalize_predictor_settings(
        normalized.get("predictor_settings"), normalized["predictor_entities"], normalized.get("lag_days", 0)
    )
    normalized["predictor_entities"] = [
        row["entity"] for row in normalized["predictor_settings"]
    ]
    normalized["cross_validation_entities"] = ensure_list(
        normalized.get("cross_validation_entities")
    )
    normalized["sensor_maps"] = normalize_sensor_maps(normalized.get("sensor_maps"))
    normalized["saved_analyses"] = normalize_saved_analyses(
        normalized.get("saved_analyses")
    )
    normalized["causal_dag"] = normalize_causal_dag(normalized.get("causal_dag"))
    normalized["outcome_operator"] = normalize_outcome_operator(
        normalized.get("outcome_operator")
    )
    normalized["exposure_sensor"] = str(normalized.get("exposure_sensor", "")).strip()
    normalized["exposure_aggregation"] = normalize_aggregation(
        normalized.get("exposure_aggregation", "sum")
    )
    normalized["exposure_operator"] = normalize_outcome_operator(
        normalized.get("exposure_operator", "more_than")
    )
    normalized["exposure_threshold"] = parse_float(
        normalized.get("exposure_threshold"), 0
    )
    normalized["outcome_threshold"] = parse_float(
        normalized.get("outcome_threshold"), 33
    )
    normalized["sleep_start_sensor"] = str(normalized.get("sleep_start_sensor", "")).strip()
    normalized["exclude_before_sleep_start_hours"] = max(
        0, min(24, parse_int(normalized.get("exclude_before_sleep_start_hours"), 0))
    )
    normalized["exclude_before_sleep_start_minutes"] = max(
        0, min(1440, parse_int(normalized.get("exclude_before_sleep_start_minutes"), 0))
    )
    normalized["analysis_window_days"] = max(
        14, min(730, parse_int(normalized.get("analysis_window_days"), 90))
    )
    normalized["analysis_method"] = normalize_analysis_method(
        normalized.get("analysis_method")
    )
    normalized["lag_days"] = max(0, min(14, parse_int(normalized.get("lag_days"), 1)))
    normalized["max_lag_days"] = max(1, min(21, parse_int(normalized.get("max_lag_days"), 7)))
    normalized["state_forgetting"] = max(
        0.90, min(0.999, parse_float(normalized.get("state_forgetting"), 0.985))
    )
    apply_analysis_settings_from_dag(normalized)
    return normalized


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return split_entities(str(value))


def split_entities(value):
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def form_entities(form, name):
    values = form.getlist(name)
    if not values:
        return split_entities(form.get(name, ""))
    entities = []
    for value in values:
        entities.extend(split_entities(value))
    return entities


def apply_settings_form(config, form):
    config.update(
        {
            "sleep_sensor": form.get("sleep_sensor", "").strip(),
            "exposure_sensor": form.get("exposure_sensor", "").strip(),
            "exposure_aggregation": normalize_aggregation(
                form.get("exposure_aggregation", "sum")
            ),
            "exposure_operator": form.get("exposure_operator", "more_than"),
            "exposure_threshold": parse_float(form.get("exposure_threshold"), 0),
            "outcome_operator": form.get("outcome_operator", "less_than"),
            "outcome_threshold": parse_float(form.get("outcome_threshold"), 33),
            "sleep_start_sensor": form.get("sleep_start_sensor", "").strip(),
            "exclude_before_sleep_start_hours": 0,
            "exclude_before_sleep_start_minutes": parse_int(
                form.get("exclude_before_sleep_start_minutes"), 0
            ),
            "predictor_settings": form_predictor_settings(form),
            "cross_validation_entities": [],
            "analysis_window_days": parse_int(form.get("analysis_window_days"), 90),
            "analysis_method": normalize_analysis_method(form.get("analysis_method")),
            "lag_days": 0,
            "max_lag_days": parse_int(form.get("max_lag_days"), 7),
            "state_forgetting": parse_float(form.get("state_forgetting"), 0.985),
        }
    )
    config["predictor_entities"] = [
        row["entity"] for row in config["predictor_settings"]
    ]
    apply_analysis_settings_from_dag(config)
    return config


def apply_analysis_settings_from_dag(config):
    nodes = config.get("causal_dag", {}).get("nodes", [])
    outcome = next((node for node in nodes if node.get("role") == "outcome" and "." in node.get("id", "")), None)
    exposure = next((node for node in nodes if node.get("role") == "exposure" and "." in node.get("id", "")), None)
    covariates = [
        node
        for node in nodes
        if node.get("role") == "covariate" and "." in node.get("id", "")
    ]

    if outcome:
        config["sleep_sensor"] = outcome["id"]
        config["outcome_operator"] = normalize_threshold_operator(
            outcome.get("threshold_operator", "less_than")
        )
        threshold = outcome.get("threshold")
        config["outcome_threshold"] = 33 if threshold is None else parse_float(threshold, 33)
    else:
        config["sleep_sensor"] = ""

    if exposure:
        config["exposure_sensor"] = exposure["id"]
        config["exposure_aggregation"] = normalize_aggregation(exposure.get("aggregation", "max"))
        config["exposure_operator"] = normalize_threshold_operator(
            exposure.get("threshold_operator", "more_than")
        )
        threshold = exposure.get("threshold")
        config["exposure_threshold"] = 0.5 if threshold is None else parse_float(threshold, 0.5)
    else:
        config["exposure_sensor"] = ""

    predictor_settings = []
    seen = set()
    for node in covariates:
        entity = node["id"]
        if entity in seen or entity in {config.get("sleep_sensor"), config.get("exposure_sensor")}:
            continue
        seen.add(entity)
        predictor_settings.append(
            {
                "entity": entity,
                "aggregation": normalize_aggregation(node.get("aggregation", "last")),
                "day_offset": max(0, min(21, parse_int(node.get("day_lag"), 1))),
            }
        )
    config["predictor_settings"] = predictor_settings
    config["predictor_entities"] = [row["entity"] for row in predictor_settings]
    return config


def form_predictor_settings(form):
    entities = form.getlist("predictor_entities")
    aggregations = form.getlist("predictor_aggregations")
    day_offsets = form.getlist("predictor_day_offsets")
    settings = []
    for index, entity in enumerate(entities):
        entity = entity.strip()
        if not entity:
            continue
        settings.append(
            {
                "entity": entity,
                "aggregation": normalize_aggregation(
                    aggregations[index] if index < len(aggregations) else "last"
                ),
                "day_offset": normalize_day_offset(
                    day_offsets[index] if index < len(day_offsets) else "0"
                ),
            }
        )
    return settings


def parse_dag_form(form):
    node_ids = form.getlist("node_id")
    labels = form.getlist("node_label")
    descriptions = form.getlist("node_description")
    roles = form.getlist("node_role")
    x_values = form.getlist("node_x")
    y_values = form.getlist("node_y")
    aggregations = form.getlist("node_aggregation")
    time_starts = form.getlist("node_time_start")
    time_ends = form.getlist("node_time_end")
    day_lags = form.getlist("node_day_lag")
    missing_strategies = form.getlist("node_missing_strategy")
    min_valid_values = form.getlist("node_min_valid")
    max_valid_values = form.getlist("node_max_valid")
    rolling_days = form.getlist("node_rolling_days")
    value_types = form.getlist("node_value_type")
    threshold_operators = form.getlist("node_threshold_operator")
    thresholds = form.getlist("node_threshold")
    nodes = []
    seen_nodes = set()
    for index, node_id in enumerate(node_ids):
        node_id = node_id.strip()
        if not node_id or node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "label": labels[index].strip() if index < len(labels) and labels[index].strip() else node_id,
                "description": descriptions[index].strip() if index < len(descriptions) else "",
                "role": normalize_dag_role(roles[index] if index < len(roles) else "covariate"),
                "x": parse_float(x_values[index] if index < len(x_values) else 120, 120),
                "y": max(64, parse_float(y_values[index] if index < len(y_values) else 120, 120)),
                "aggregation": normalize_aggregation(
                    aggregations[index] if index < len(aggregations) else "last"
                ),
                "time_start": normalize_time_text(time_starts[index] if index < len(time_starts) else ""),
                "time_end": normalize_time_text(time_ends[index] if index < len(time_ends) else ""),
                "day_lag": max(0, min(21, parse_int(day_lags[index] if index < len(day_lags) else 0, 0))),
                "missing_strategy": normalize_missing_strategy(
                    missing_strategies[index] if index < len(missing_strategies) else "drop"
                ),
                "min_valid": parse_optional_float(min_valid_values[index] if index < len(min_valid_values) else ""),
                "max_valid": parse_optional_float(max_valid_values[index] if index < len(max_valid_values) else ""),
                "rolling_days": max(1, min(30, parse_int(rolling_days[index] if index < len(rolling_days) else 1, 1))),
                "value_type": normalize_node_value_type(
                    value_types[index] if index < len(value_types) else "continuous"
                ),
                "threshold_operator": normalize_threshold_operator(
                    threshold_operators[index] if index < len(threshold_operators) else "more_than"
                ),
                "threshold": parse_optional_float(thresholds[index] if index < len(thresholds) else ""),
            }
        )

    sources = form.getlist("edge_source")
    targets = form.getlist("edge_target")
    edge_labels = form.getlist("edge_label")
    edges = []
    seen_edges = set()
    node_set = {node["id"] for node in nodes}
    for index, source in enumerate(sources):
        source = source.strip()
        target = targets[index].strip() if index < len(targets) else ""
        label = edge_labels[index].strip() if index < len(edge_labels) else ""
        key = (source, target, label)
        if not source or not target or source == target or source not in node_set or target not in node_set or key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append({"source": source, "target": target, "label": label})

    return {"nodes": nodes, "edges": edges}


def find_causal_dag_cycle(dag):
    node_ids = {node["id"] for node in dag.get("nodes", [])}
    adjacency = {node_id: [] for node_id in node_ids}
    for edge in dag.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if source in node_ids and target in node_ids:
            adjacency[source].append(target)

    visiting = set()
    visited = set()
    stack = []

    def visit(node_id):
        if node_id in visiting:
            start = stack.index(node_id)
            return stack[start:] + [node_id]
        if node_id in visited:
            return []
        visiting.add(node_id)
        stack.append(node_id)
        for next_id in adjacency.get(node_id, []):
            cycle = visit(next_id)
            if cycle:
                return cycle
        stack.pop()
        visiting.remove(node_id)
        visited.add(node_id)
        return []

    for node_id in node_ids:
        cycle = visit(node_id)
        if cycle:
            return cycle
    return []


def normalize_predictor_settings(value, predictor_entities, lag_days):
    settings = []
    seen = set()
    if isinstance(value, list):
        for row in value:
            if not isinstance(row, dict):
                continue
            entity = str(row.get("entity", "")).strip()
            if not entity or entity in seen:
                continue
            seen.add(entity)
            settings.append(
                {
                    "entity": entity,
                    "aggregation": normalize_aggregation(row.get("aggregation", "last")),
                    "day_offset": normalize_day_offset(row.get("day_offset", lag_days)),
                }
            )
    for entity in predictor_entities:
        if entity in seen:
            continue
        seen.add(entity)
        settings.append(
            {
                "entity": entity,
                "aggregation": "last",
                "day_offset": normalize_day_offset(lag_days),
            }
        )
    return settings


def normalize_aggregation(value):
    if value in {"sum", "min", "max", "mean"}:
        return value
    return "last"


def normalize_time_text(value):
    text = str(value or "").strip()
    if not text:
        return ""
    parts = text.split(":")
    if len(parts) < 2:
        return ""
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return ""
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return ""
    return f"{hour:02d}:{minute:02d}"


def filter_history_time_window(history, time_start, time_end):
    time_start = normalize_time_text(time_start)
    time_end = normalize_time_text(time_end)
    if history.empty or (not time_start and not time_end):
        return history
    minutes = history["last_changed"].dt.hour * 60 + history["last_changed"].dt.minute
    start_minutes = parse_time_minutes(time_start, 0) if time_start else 0
    end_minutes = parse_time_minutes(time_end, 1439) if time_end else 1439
    if start_minutes <= end_minutes:
        mask = (minutes >= start_minutes) & (minutes <= end_minutes)
    else:
        mask = (minutes >= start_minutes) | (minutes <= end_minutes)
    return history[mask]


def parse_time_minutes(value, fallback):
    normalized = normalize_time_text(value)
    if not normalized:
        return fallback
    hour, minute = normalized.split(":")
    return int(hour) * 60 + int(minute)


def normalize_day_offset(value):
    return 1 if str(value) == "1" else 0


def normalize_sensor_maps(value):
    if not isinstance(value, list):
        return []

    maps = []
    seen = set()
    for row in value:
        if not isinstance(row, dict):
            continue
        sensor = str(row.get("sensor", "")).strip()
        description = str(row.get("description", "")).strip()
        role = normalize_sensor_role(row.get("role", infer_sensor_role(sensor)))
        if not sensor or sensor in seen:
            continue
        seen.add(sensor)
        maps.append({"sensor": sensor, "description": description, "role": role})
    return maps


def normalize_sensor_role(value):
    if value == "outcome":
        return "outcome"
    return "covariate"


def normalize_dag_role(value):
    if value in {"exposure", "outcome", "covariate", "latent"}:
        return value
    return "covariate"


def infer_sensor_role(sensor):
    return "outcome" if "sleep" in sensor else "covariate"


def normalize_saved_analyses(value):
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def find_saved_analysis(config, analysis_id):
    return next(
        (
            row
            for row in normalize_saved_analyses(config.get("saved_analyses"))
            if str(row.get("id")) == str(analysis_id)
        ),
        None,
    )


def normalize_causal_dag(value):
    if not isinstance(value, dict):
        value = {}
    nodes = []
    seen_nodes = set()
    for index, row in enumerate(value.get("nodes", [])):
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("id", "")).strip()
        if not node_id or node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "label": str(row.get("label", node_id)).strip() or node_id,
                "description": str(row.get("description", "")).strip(),
                "role": normalize_dag_role(row.get("role", "covariate")),
                "x": max(0, min(920, parse_float(row.get("x"), 80 + index * 180))),
                "y": max(64, min(520, parse_float(row.get("y"), 120))),
                "aggregation": normalize_aggregation(row.get("aggregation", "last")),
                "time_start": normalize_time_text(row.get("time_start", "")),
                "time_end": normalize_time_text(row.get("time_end", "")),
                "day_lag": max(0, min(21, parse_int(row.get("day_lag"), 0))),
                "missing_strategy": normalize_missing_strategy(row.get("missing_strategy", "drop")),
                "min_valid": parse_optional_float(row.get("min_valid")),
                "max_valid": parse_optional_float(row.get("max_valid")),
                "rolling_days": max(1, min(30, parse_int(row.get("rolling_days"), 1))),
                "value_type": normalize_node_value_type(row.get("value_type", "continuous")),
                "threshold_operator": normalize_threshold_operator(
                    row.get("threshold_operator", "more_than")
                ),
                "threshold": parse_optional_float(row.get("threshold")),
            }
        )
    edges = []
    seen_edges = set()
    node_ids = {node["id"] for node in nodes}
    for row in value.get("edges", []):
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", "")).strip()
        target = str(row.get("target", "")).strip()
        label = str(row.get("label", "")).strip()
        key = (source, target, label)
        if not source or not target or source == target or source not in node_ids or target not in node_ids or key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append({"source": source, "target": target, "label": label})
    return {"nodes": nodes, "edges": edges}


def saved_analysis_settings(config):
    return {
        "sleep_sensor": config["sleep_sensor"],
        "exposure_sensor": config.get("exposure_sensor", ""),
        "exposure_aggregation": config.get("exposure_aggregation", "sum"),
        "exposure_operator": config.get("exposure_operator", "more_than"),
        "exposure_threshold": config.get("exposure_threshold", 0),
        "outcome_operator": config["outcome_operator"],
        "outcome_threshold": config["outcome_threshold"],
        "analysis_window_days": config["analysis_window_days"],
        "analysis_method": config.get("analysis_method", "g_formula"),
        "max_lag_days": config.get("max_lag_days", 7),
        "state_forgetting": config.get("state_forgetting", 0.985),
        "sleep_start_sensor": config.get("sleep_start_sensor", ""),
        "exclude_before_sleep_start_minutes": config.get("exclude_before_sleep_start_minutes", 0),
        "predictor_settings": config["predictor_settings"],
    }


def build_saved_analysis_snapshot(config, result, created_at):
    settings = saved_analysis_settings(config)
    dag = normalize_causal_dag(config.get("causal_dag"))
    method = normalize_analysis_method(result.get("analysis_method", settings.get("analysis_method")))
    return {
        "id": created_at.strftime("%Y%m%d%H%M%S"),
        "name": f"{result['outcome_label']} · {created_at.strftime('%Y-%m-%d %H:%M UTC')}",
        "created_at": created_at.strftime("%Y-%m-%d %H:%M UTC"),
        "created_at_iso": created_at.isoformat(),
        "immutable": True,
        "metadata": {
            "analysis_kind": result.get("analysis_kind", "dynamic_causal"),
            "rows": result.get("rows"),
            "date_start": result.get("date_start"),
            "date_end": result.get("date_end"),
            "outcome_label": result.get("outcome_label"),
            "exposure_label": result.get("exposure_label"),
            "analysis_window_days": settings.get("analysis_window_days"),
            "analysis_method": method,
            "analysis_method_label": analysis_method_label(method),
            "covariate_count": result.get("covariate_count"),
        },
        "settings": settings,
        "model_settings": {
            "analysis_method": method,
            "analysis_method_label": analysis_method_label(method),
            "state_forgetting": settings.get("state_forgetting"),
            "model_terms": result.get("model_terms", []),
            "positivity": result.get("positivity", []),
        },
        "dag": dag,
        "dag_svg": build_dag_svg(dag),
        "variable_settings": dag.get("nodes", []),
        "effect_estimates": result.get("latest_effect", {}),
        "plots": result.get("method_outputs", {}),
        "diagnostics": {
            "validation_errors": result.get("result_validation_errors", []),
            "positivity": result.get("positivity", []),
        },
        "result": result,
    }


def analysis_report_snapshot(analysis):
    snapshot = analysis.get("snapshot")
    if isinstance(snapshot, dict):
        return snapshot
    result = analysis.get("result", {})
    settings = analysis.get("settings", {})
    dag = normalize_causal_dag(settings.get("causal_dag") or {})
    return {
        "id": analysis.get("id"),
        "name": analysis.get("name", "Saved analysis"),
        "created_at": analysis.get("created_at", "Saved analysis"),
        "immutable": True,
        "metadata": {
            "analysis_kind": result.get("analysis_kind", "dynamic_causal"),
            "rows": result.get("rows") or analysis.get("rows"),
            "date_start": result.get("date_start"),
            "date_end": result.get("date_end"),
            "outcome_label": result.get("outcome_label") or settings.get("sleep_sensor"),
            "exposure_label": result.get("exposure_label") or settings.get("exposure_sensor"),
            "analysis_window_days": settings.get("analysis_window_days"),
            "analysis_method": result.get("analysis_method") or settings.get("analysis_method", "g_formula"),
            "analysis_method_label": result.get("analysis_method_label") or analysis_method_label(settings.get("analysis_method")),
            "covariate_count": result.get("covariate_count"),
        },
        "settings": settings,
        "model_settings": {
            "analysis_method": result.get("analysis_method") or settings.get("analysis_method", "g_formula"),
            "analysis_method_label": result.get("analysis_method_label") or analysis_method_label(settings.get("analysis_method")),
            "state_forgetting": settings.get("state_forgetting"),
            "model_terms": result.get("model_terms", []),
            "positivity": result.get("positivity", []),
        },
        "dag": dag,
        "dag_svg": build_dag_svg(dag),
        "variable_settings": dag.get("nodes", []),
        "effect_estimates": result.get("latest_effect", {}),
        "plots": result.get("method_outputs", {}),
        "diagnostics": {
            "validation_errors": result.get("result_validation_errors", []),
            "positivity": result.get("positivity", []),
        },
        "result": result,
    }


def build_dag_svg(dag):
    nodes = normalize_causal_dag(dag).get("nodes", [])
    edges = normalize_causal_dag(dag).get("edges", [])
    if not nodes:
        return '<svg viewBox="0 0 800 260" role="img" aria-label="Causal DAG"></svg>'
    node_width = 180
    node_height = 74
    padding = 48
    min_x = min(float(node.get("x", 0)) for node in nodes)
    min_y = min(float(node.get("y", 0)) for node in nodes)
    max_x = max(float(node.get("x", 0)) + node_width for node in nodes)
    max_y = max(float(node.get("y", 0)) + node_height for node in nodes)
    width = max(640, int(max_x - min_x + padding * 2))
    height = max(260, int(max_y - min_y + padding * 2))

    def shifted(node):
        return (
            float(node.get("x", 0)) - min_x + padding,
            float(node.get("y", 0)) - min_y + padding,
        )

    node_by_id = {node["id"]: node for node in nodes}
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Causal DAG" xmlns="http://www.w3.org/2000/svg">',
        "<defs><marker id=\"report-arrow\" markerWidth=\"10\" markerHeight=\"10\" refX=\"8\" refY=\"3\" orient=\"auto\"><path d=\"M0,0 L0,6 L9,3 z\" fill=\"#2563eb\"/></marker></defs>",
        '<rect width="100%" height="100%" rx="16" fill="#fbfcfe"/>',
    ]
    for edge in edges:
        source = node_by_id.get(edge.get("source"))
        target = node_by_id.get(edge.get("target"))
        if not source or not target:
            continue
        sx, sy = shifted(source)
        tx, ty = shifted(target)
        x1 = sx + node_width
        y1 = sy + node_height / 2
        x2 = tx
        y2 = ty + node_height / 2
        if tx < sx:
            x1 = sx
            x2 = tx + node_width
        mid_x = (x1 + x2) / 2
        parts.append(
            f'<polyline points="{x1:.1f},{y1:.1f} {mid_x:.1f},{y1:.1f} {mid_x:.1f},{y2:.1f} {x2:.1f},{y2:.1f}" fill="none" stroke="#2563eb" stroke-width="2.4" marker-end="url(#report-arrow)"/>'
        )
    for node in nodes:
        x, y = shifted(node)
        role = html_escape(str(node.get("role", "variable")))
        label = html_escape(str(node.get("label") or humanize_report_label(node.get("id"))))
        time = "t" if node.get("role") == "outcome" else "t-1"
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{node_width}" height="{node_height}" rx="14" fill="#ffffff" stroke="#d0d5dd"/>',
                f'<text x="{x + 16:.1f}" y="{y + 30:.1f}" fill="#101828" font-family="Inter, Arial, sans-serif" font-size="17" font-weight="700">{label}</text>',
                f'<text x="{x + 16:.1f}" y="{y + 55:.1f}" fill="#667085" font-family="Inter, Arial, sans-serif" font-size="12">({role})</text>',
                f'<text x="{x + node_width - 32:.1f}" y="{y + 30:.1f}" fill="#98a2b3" font-family="Inter, Arial, sans-serif" font-size="13">{time}</text>',
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def humanize_report_label(value):
    text = str(value or "Variable").split(".")[-1]
    return text.replace("nick_r_", "").replace("_", " ").title()


class SimplePdf:
    def __init__(self):
        self.width = 612
        self.height = 792
        self.pages = []
        self.commands = []
        self.y = 742
        self.add_page()

    def add_page(self):
        if self.commands:
            self.pages.append("\n".join(self.commands))
        self.commands = []
        self.y = 742

    def ensure_space(self, height):
        if self.y - height < 48:
            self.add_page()

    def text(self, x, y, text, size=10, bold=False, color=(16, 24, 40)):
        safe = pdf_escape(text)
        font = "/F2" if bold else "/F1"
        r, g, b = [value / 255 for value in color]
        self.commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg BT {font} {size} Tf {x:.1f} {y:.1f} Td ({safe}) Tj ET")

    def line(self, x1, y1, x2, y2, color=(208, 213, 221), width=1):
        r, g, b = [value / 255 for value in color]
        self.commands.append(f"{r:.3f} {g:.3f} {b:.3f} RG {width:.2f} w {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S")

    def rect(self, x, y, width, height, fill=None, stroke=(229, 231, 235), line_width=1):
        if fill:
            r, g, b = [value / 255 for value in fill]
            self.commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x:.1f} {y:.1f} {width:.1f} {height:.1f} re f")
        if stroke:
            r, g, b = [value / 255 for value in stroke]
            self.commands.append(f"{r:.3f} {g:.3f} {b:.3f} RG {line_width:.2f} w {x:.1f} {y:.1f} {width:.1f} {height:.1f} re S")

    def paragraph(self, x, text, width=512, size=10, bold=False, color=(71, 84, 103), leading=None):
        leading = leading or size + 4
        lines = wrap_pdf_text(text, max(20, int(width / (size * 0.5))))
        self.ensure_space(len(lines) * leading + 4)
        for line in lines:
            self.text(x, self.y, line, size=size, bold=bold, color=color)
            self.y -= leading
        self.y -= 4

    def heading(self, text):
        self.ensure_space(34)
        self.text(50, self.y, text, size=15, bold=True, color=(16, 24, 40))
        self.y -= 20
        self.line(50, self.y, 562, self.y)
        self.y -= 14

    def key_values(self, rows):
        self.ensure_space(len(rows) * 18 + 10)
        for label, value in rows:
            self.text(58, self.y, str(label), size=9, bold=True, color=(102, 112, 133))
            self.text(210, self.y, str(value if value is not None else "-"), size=9, color=(16, 24, 40))
            self.y -= 18
        self.y -= 6

    def output(self):
        if self.commands:
            self.pages.append("\n".join(self.commands))
            self.commands = []
        objects = []
        objects.append("<< /Type /Catalog /Pages 2 0 R >>")
        kids = " ".join(f"{3 + index * 2} 0 R" for index in range(len(self.pages)))
        objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(self.pages)} >>")
        for index, content in enumerate(self.pages):
            page_object_id = 3 + index * 2
            content_object_id = page_object_id + 1
            objects.append(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
                f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> "
                f"/F2 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> >> >> "
                f"/Contents {content_object_id} 0 R >>"
            )
            stream = content.encode("latin-1", "replace")
            objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode('latin-1')}\nendstream")
        pdf = ["%PDF-1.4"]
        offsets = [0]
        for number, obj in enumerate(objects, start=1):
            offsets.append(sum(len(part.encode("latin-1")) + 1 for part in pdf))
            pdf.append(f"{number} 0 obj\n{obj}\nendobj")
        xref = sum(len(part.encode("latin-1")) + 1 for part in pdf)
        pdf.append(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f ")
        pdf.extend(f"{offset:010d} 00000 n " for offset in offsets[1:])
        pdf.append(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF")
        return "\n".join(pdf).encode("latin-1", "replace")


def pdf_escape(value):
    return (
        str(value or "")
        .encode("latin-1", "replace")
        .decode("latin-1")
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def wrap_pdf_text(text, max_chars):
    words = str(text or "").replace("\n", " ").split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def build_saved_analysis_pdf(report):
    pdf = SimplePdf()
    metadata = report.get("metadata", {})
    settings = report.get("settings", {})
    effect = report.get("effect_estimates", {})
    pdf.text(50, pdf.y, "Causal Analysis Engine report", size=20, bold=True)
    pdf.y -= 26
    pdf.paragraph(50, report.get("name", "Saved analysis"), size=11, color=(71, 84, 103))
    pdf.key_values(
        [
            ("Created", report.get("created_at")),
            ("Analysis window", f"{metadata.get('analysis_window_days') or settings.get('analysis_window_days')} days"),
            ("Method", metadata.get("analysis_method_label")),
            ("State forgetting", report.get("model_settings", {}).get("state_forgetting")),
            ("Outcome (Dependent)", metadata.get("outcome_label")),
            ("Exposure (Independent)", metadata.get("exposure_label")),
            ("Rows", metadata.get("rows")),
            ("Date range", f"{metadata.get('date_start') or '-'} to {metadata.get('date_end') or '-'}"),
        ]
    )
    pdf.heading("Causal DAG")
    draw_pdf_dag(pdf, report.get("dag", {}))
    pdf.heading("Variable settings")
    for node in report.get("variable_settings", []):
        pdf.paragraph(
            58,
            f"{node.get('label') or humanize_report_label(node.get('id'))} ({node.get('role')}): "
            f"sensor {node.get('id')}; aggregate {node.get('aggregation')}; lag {node.get('day_lag', 0)} day(s); "
            f"type {node.get('value_type', 'continuous')}.",
            size=9,
        )
    pdf.heading("Main effect estimate")
    pdf.key_values(
        [
            ("Latest date", effect.get("date")),
            ("Estimate", effect.get("contemporaneous")),
            ("95% CI", f"{effect.get('lower')} to {effect.get('upper')}"),
            ("Lag 1", effect.get("lag1")),
            ("Cumulative", effect.get("cumulative")),
        ]
    )
    pdf.heading("Plots and diagnostics")
    write_pdf_outputs(pdf, report.get("plots", {}))
    positivity = report.get("diagnostics", {}).get("positivity", [])
    if positivity:
        pdf.heading("Positivity")
        for row in positivity[:8]:
            pdf.paragraph(
                58,
                f"{row.get('duration')} day(s): {row.get('observed')} of {row.get('possible')} histories observed ({row.get('percent')}%).",
                size=9,
            )
    return pdf.output()


def draw_pdf_dag(pdf, dag):
    nodes = normalize_causal_dag(dag).get("nodes", [])
    edges = normalize_causal_dag(dag).get("edges", [])
    if not nodes:
        pdf.paragraph(58, "No DAG was saved with this analysis.", size=9)
        return
    pdf.ensure_space(180)
    box_x, box_y, box_w, box_h = 58, pdf.y - 170, 496, 160
    pdf.rect(box_x, box_y, box_w, box_h, fill=(251, 252, 254), stroke=(229, 231, 235))
    min_x = min(float(node.get("x", 0)) for node in nodes)
    min_y = min(float(node.get("y", 0)) for node in nodes)
    max_x = max(float(node.get("x", 0)) for node in nodes)
    max_y = max(float(node.get("y", 0)) for node in nodes)
    scale_x = (box_w - 120) / max(1, max_x - min_x)
    scale_y = (box_h - 70) / max(1, max_y - min_y)
    scale = min(scale_x, scale_y, 1.2)
    node_w, node_h = 94, 34

    def point(node):
        return (
            box_x + 30 + (float(node.get("x", 0)) - min_x) * scale,
            box_y + box_h - 50 - (float(node.get("y", 0)) - min_y) * scale,
        )

    node_by_id = {node["id"]: node for node in nodes}
    for edge in edges:
        source = node_by_id.get(edge.get("source"))
        target = node_by_id.get(edge.get("target"))
        if not source or not target:
            continue
        sx, sy = point(source)
        tx, ty = point(target)
        pdf.line(sx + node_w, sy + node_h / 2, tx, ty + node_h / 2, color=(37, 99, 235), width=1.2)
    for node in nodes:
        x, y = point(node)
        pdf.rect(x, y, node_w, node_h, fill=(255, 255, 255), stroke=(208, 213, 221))
        pdf.text(x + 6, y + 20, (node.get("label") or humanize_report_label(node.get("id")))[:18], size=8, bold=True)
        pdf.text(x + 6, y + 8, f"({node.get('role')})", size=7, color=(102, 112, 133))
    pdf.y = box_y - 20


def write_pdf_outputs(pdf, method_outputs):
    if not isinstance(method_outputs, dict):
        pdf.paragraph(58, "No method-specific outputs were stored.", size=9)
        return
    sections = [
        ("Main results", method_outputs.get("main", [])),
        ("Diagnostics", method_outputs.get("diagnostics", [])),
        ("Advanced results", method_outputs.get("advanced", [])),
    ]
    wrote = False
    for section_label, outputs in sections:
        if not outputs:
            continue
        wrote = True
        pdf.paragraph(58, section_label, size=10, bold=True, color=(16, 24, 40))
        for output in outputs:
            write_pdf_output_summary(pdf, output)
    if not wrote:
        pdf.paragraph(58, "No additional plots or diagnostics were stored for this method.", size=9)


def write_pdf_output_summary(pdf, output):
    title = output.get("title", output.get("id", "Output"))
    kind = output.get("kind")
    pdf.paragraph(70, title, size=9, bold=True, color=(16, 24, 40))
    if output.get("subtitle"):
        pdf.paragraph(70, output["subtitle"], size=8)
    if kind == "line_chart":
        chart = output.get("chart", {})
        pdf.paragraph(70, f"Line plot with {len(chart.get('points', []))} time points and {len(chart.get('series', []))} series.", size=8)
    elif kind == "histogram":
        bins = output.get("bins", [])
        total = sum(int(row.get("count", 0)) for row in bins)
        pdf.paragraph(70, f"Histogram with {len(bins)} bins and {total} observations.", size=8)
    elif kind == "love_plot":
        rows = output.get("rows", [])
        for row in rows[:8]:
            pdf.paragraph(70, f"{row.get('label')}: before {row.get('before')}, after {row.get('after')}.", size=8)
    elif kind == "ess_summary":
        rows = output.get("rows", [])
        pdf.paragraph(70, "; ".join(f"{row.get('label')}: {row.get('value')}" for row in rows), size=8)
    elif kind == "counterfactual_table":
        for row in output.get("rows", []):
            pdf.paragraph(70, f"{row.get('component')}: {row.get('mean')} - {row.get('detail')}", size=8)


def normalize_outcome_operator(value):
    if value == "more_than":
        return "more_than"
    return "less_than"


def normalize_node_value_type(value):
    if value in {"continuous", "binary_threshold"}:
        return value
    return "continuous"


def normalize_threshold_operator(value):
    if value == "less_than":
        return "less_than"
    return "more_than"


def normalize_missing_strategy(value):
    if value in {"drop", "forward_fill", "interpolate"}:
        return value
    return "drop"


def analysis_method_options():
    return {
        "dag_auto": {
            "label": "Choose from DAG",
            "description": "Use the DAG roles and arrows to choose the estimator workflow.",
        },
        "g_formula": {
            "label": "G-formula / outcome model",
            "description": "Estimate counterfactual outcomes with an outcome-regression model.",
        },
        "ipw": {
            "label": "Inverse probability weighting (IPW)",
            "description": "Weight observed histories by estimated exposure probabilities.",
        },
        "doubly_robust": {
            "label": "Doubly robust / AIPW",
            "description": "Combine outcome modeling with exposure-propensity weighting.",
        },
    }


def normalize_analysis_method(value):
    if value in analysis_method_options():
        return value
    return "g_formula"


def analysis_method_label(value):
    return analysis_method_options()[normalize_analysis_method(value)]["label"]


def parse_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def parse_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def parse_optional_float(value):
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def histogram_bin_count(requested_days):
    return max(5, int(np.ceil(requested_days / 5)))


def build_distribution_statistics(values):
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {}

    percentile_points = [5, 10, 25, 50, 75, 90, 95]
    statistics = {
        "mean": round(float(clean.mean()), 3),
        "median": round(float(clean.median()), 3),
        "std_dev": round(float(clean.std(ddof=1)), 3) if len(clean) > 1 else 0,
        "minimum": round(float(clean.min()), 3),
        "maximum": round(float(clean.max()), 3),
        "percentiles": [
            {
                "label": f"P{point}",
                "value": round(float(np.percentile(clean.to_numpy(dtype=float), point)), 3),
            }
            for point in percentile_points
        ],
    }

    if set(clean.unique()).issubset({0, 1}):
        on_count = int((clean == 1).sum())
        off_count = int((clean == 0).sum())
        statistics.update(
            {
                "on_count": on_count,
                "off_count": off_count,
                "on_percent": round((on_count / len(clean)) * 100, 1),
            }
        )

    return statistics


def build_analysis_preview(config):
    predictors = config["predictor_entities"]
    validators = config["cross_validation_entities"]
    return {
        "ready": bool(config["sleep_sensor"] and config.get("exposure_sensor")),
        "predictor_count": len(predictors),
        "validator_count": len(validators),
        "feature_count": len(set(predictors + validators)),
    }


def build_outcome_histogram(config):
    if not config["sleep_sensor"]:
        return {"ok": False, "message": "Choose an outcome sensor.", "bins": []}

    history_config = {
        **config,
        "predictor_entities": [],
        "cross_validation_entities": [],
        "lag_days": 0,
    }
    history = fetch_home_assistant_history(history_config)
    if history.empty:
        return {
            "ok": False,
            "message": "No numeric Home Assistant history was returned for this sensor.",
            "bins": [],
        }

    values = daily_sensor_values(history, config["sleep_sensor"], config["analysis_window_days"])
    if values.empty:
        return {
            "ok": False,
            "message": "No usable daily numeric values were found for this sensor.",
            "bins": [],
        }

    return build_series_histogram(
        values,
        config["sleep_sensor"],
        int(config["analysis_window_days"]),
    )


def build_covariate_histogram(config, predictor):
    if not predictor["entity"]:
        return {"ok": False, "message": "Choose a covariate sensor.", "bins": []}

    history_config = {
        **config,
        "predictor_entities": [predictor["entity"]],
        "predictor_settings": [predictor],
        "cross_validation_entities": [],
    }
    history = fetch_home_assistant_history(history_config)
    if history.empty:
        return {
            "ok": False,
            "message": "No numeric Home Assistant history was returned for this covariate.",
            "bins": [],
        }

    history = exclude_covariate_window(history, history_config)
    entity_history = history[history["entity_id"] == predictor["entity"]].dropna(subset=["state"])
    values = aggregate_daily_values(entity_history, predictor["aggregation"])
    if predictor["day_offset"]:
        values = values.shift(predictor["day_offset"])
    if not values.empty:
        latest_day = values.index.max()
        start = latest_day - pd.Timedelta(days=config["analysis_window_days"] - 1)
        values = values[values.index >= start].dropna()

    if values.empty:
        return {
            "ok": False,
            "message": "No usable daily values were found for this covariate.",
            "bins": [],
        }

    outcome_values = daily_sensor_values(
        history,
        config["sleep_sensor"],
        int(config["analysis_window_days"]),
    )
    outcome = build_binary_outcome(outcome_values, config) if not outcome_values.empty else pd.Series(dtype=bool)

    return build_series_histogram(
        values,
        predictor["entity"],
        int(config["analysis_window_days"]),
        outcome=outcome,
        extra={
            "aggregation": predictor["aggregation"],
            "day_offset": predictor["day_offset"],
            "outcome_sensor": config["sleep_sensor"],
            "outcome_label": build_outcome_label(config),
        },
    )


def build_series_histogram(values, sensor, requested_days, outcome=None, extra=None):
    association = build_histogram_association(values, outcome)
    unique_values = sorted({round(float(value), 6) for value in values.to_list()})
    is_binary = len(unique_values) <= 2 and set(unique_values).issubset({0.0, 1.0})
    if is_binary:
        return build_binary_level_histogram(values, sensor, requested_days, outcome, association, extra)

    bin_count = histogram_bin_count(requested_days)
    counts, edges = np.histogram(values.to_numpy(dtype=float), bins=bin_count)
    aligned_outcome = None
    exposed_values = []
    if outcome is not None and not outcome.empty:
        aligned_outcome = outcome.reindex(values.index)
        exposed_values = [
            round(float(value), 3)
            for value in values[aligned_outcome == True].dropna().to_list()
        ]
    bins = []
    for index, count in enumerate(counts):
        if aligned_outcome is not None:
            lower = edges[index]
            upper = edges[index + 1]
            if index == len(counts) - 1:
                in_bin = values.between(lower, upper, inclusive="both")
            else:
                in_bin = (values >= lower) & (values < upper)
            bin_outcomes = aligned_outcome[in_bin].dropna()
            exposed_count = int(bin_outcomes.sum()) if not bin_outcomes.empty else 0
            outcome_days = int(len(bin_outcomes))
            exposed_percent = round((exposed_count / outcome_days) * 100, 1) if outcome_days else 0
        else:
            exposed_count = 0
            outcome_days = 0
            exposed_percent = 0
        bins.append(
            {
                "lower": round(float(edges[index]), 3),
                "upper": round(float(edges[index + 1]), 3),
                "count": int(count),
                "outcome_days": outcome_days,
                "exposed_count": exposed_count,
                "exposed_percent": exposed_percent,
            }
        )

    return {
        "ok": True,
        "sensor": sensor,
        "rows": int(len(values)),
        "requested_days": requested_days,
        "missing_days": max(0, requested_days - int(len(values))),
        "minimum": round(float(values.min()), 3),
        "maximum": round(float(values.max()), 3),
        "max_count": int(max(counts)) if len(counts) else 0,
        "statistics": build_distribution_statistics(values),
        "association": association,
        "values": [round(float(value), 3) for value in values.to_list()],
        "exposed_values": exposed_values,
        "bins": bins,
        **(extra or {}),
    }


def build_binary_level_histogram(values, sensor, requested_days, outcome=None, association=None, extra=None):
    aligned_outcome = None
    exposed_values = []
    if outcome is not None and not outcome.empty:
        aligned_outcome = outcome.reindex(values.index)
        exposed_values = [
            round(float(value), 3)
            for value in values[aligned_outcome == True].dropna().to_list()
        ]

    bins = []
    for value, label in [(0.0, "Off / closed / 0"), (1.0, "On / open / 1")]:
        in_level = values == value
        count = int(in_level.sum())
        if aligned_outcome is not None:
            level_outcomes = aligned_outcome[in_level].dropna()
            exposed_count = int(level_outcomes.sum()) if not level_outcomes.empty else 0
            outcome_days = int(len(level_outcomes))
            exposed_percent = round((exposed_count / outcome_days) * 100, 1) if outcome_days else 0
        else:
            exposed_count = 0
            outcome_days = 0
            exposed_percent = 0
        bins.append(
            {
                "lower": value,
                "upper": value,
                "label": label,
                "count": count,
                "outcome_days": outcome_days,
                "exposed_count": exposed_count,
                "exposed_percent": exposed_percent,
            }
        )

    return {
        "ok": True,
        "kind": "binary",
        "sensor": sensor,
        "rows": int(len(values)),
        "requested_days": requested_days,
        "missing_days": max(0, requested_days - int(len(values))),
        "minimum": round(float(values.min()), 3),
        "maximum": round(float(values.max()), 3),
        "max_count": max([bin_row["count"] for bin_row in bins] + [0]),
        "statistics": build_distribution_statistics(values),
        "association": association or build_histogram_association(values, outcome),
        "values": [round(float(value), 3) for value in values.to_list()],
        "exposed_values": exposed_values,
        "bins": bins,
        **(extra or {}),
    }


def build_histogram_association(values, outcome):
    if outcome is None or outcome.empty:
        return {"correlation": None, "available_days": 0}
    paired = pd.concat([values.rename("value"), outcome.rename("outcome")], axis=1).dropna()
    if len(paired) < 2 or paired["value"].nunique() < 2 or paired["outcome"].nunique() < 2:
        return {"correlation": None, "available_days": int(len(paired))}
    correlation = paired["value"].corr(paired["outcome"].astype(float))
    if pd.isna(correlation):
        correlation = None
    return {
        "correlation": None if correlation is None else round(float(correlation), 3),
        "available_days": int(len(paired)),
    }


def run_n_of_1_analysis(config):
    causal_result = run_dynamic_causal_analysis(config)
    if causal_result.get("ok"):
        return causal_result
    return causal_result


def run_dynamic_causal_analysis(config):
    if not config["sleep_sensor"]:
        return {"ok": False, "message": "Choose an outcome sensor before running the causal analysis."}
    if not config.get("exposure_sensor"):
        return {"ok": False, "message": "Choose a time-varying exposure sensor before running the causal analysis."}

    history = fetch_home_assistant_history(config)
    if history.empty:
        return {
            "ok": False,
            "message": "No Home Assistant history was returned. Check entity IDs and long-term history.",
        }

    dataset = prepare_causal_dataset(history, config)
    if dataset.empty or "outcome" not in dataset.columns:
        return {
            "ok": False,
            "message": "No usable history was found for the selected sleep outcome sensor.",
        }

    if "exposure" not in dataset.columns:
        return {
            "ok": False,
            "message": "No usable history was found for the selected exposure sensor.",
        }

    if len(dataset) < max(21, config["max_lag_days"] + 10):
        return {
            "ok": False,
            "message": "More usable daily observations are needed for dynamic causal analysis.",
        }

    if dataset["exposure"].nunique() < 2:
        adjusted = apply_balanced_exposure_threshold(dataset, config)
        if adjusted:
            dataset = prepare_causal_dataset(history, config)
        if dataset.empty or "exposure" not in dataset.columns or dataset["exposure"].nunique() < 2:
            return {
                "ok": False,
                "message": (
                    "The exposure threshold creates only one exposure class. The recent "
                    "history does not contain enough exposure variation for this sensor."
                ),
            }

    effect_model = fit_dynamic_effect_model(dataset, config)
    if not effect_model["effects"]:
        return {"ok": False, "message": "The dynamic model could not estimate exposure effects."}

    requested_method = normalize_analysis_method(config.get("analysis_method"))
    resolved_method = resolve_result_method(requested_method, dataset)
    method_outputs = build_method_result_outputs(dataset, effect_model, config, resolved_method)

    return {
        "ok": True,
        "analysis_kind": "dynamic_causal",
        "rows": len(dataset),
        "date_start": dataset.index.min().strftime("%Y-%m-%d"),
        "date_end": dataset.index.max().strftime("%Y-%m-%d"),
        "outcome_label": config["sleep_sensor"],
        "exposure_label": build_exposure_label(config),
        "analysis_method": resolved_method,
        "analysis_method_requested": requested_method,
        "analysis_method_label": analysis_method_label(resolved_method),
        "covariate_count": len(config["predictor_entities"]),
        "latest_effect": effect_model["latest_effect"],
        "effect_chart": build_effect_chart(effect_model["effects"], "effect"),
        "forest_plot": effect_model["forest_plot"],
        "method_outputs": method_outputs,
        "result_validation_errors": method_outputs["validation_errors"],
        "impulse_response": effect_model["impulse_response"],
        "step_response": effect_model["step_response"],
        "general_strategies": effect_model["general_strategies"],
        "positivity": build_positivity_diagnostics(dataset["exposure"], config["max_lag_days"]),
        "model_terms": effect_model["terms"],
        "prediction_note": (
            "This estimates paper-style dynamic N-of-1 causal contrasts from observational "
            "Home Assistant time series. Interpretation still depends on the paper's key "
            "assumptions: consistency, positivity, no unmeasured time-varying confounding, "
            "and a correctly specified relevant history."
        ),
    }


METHOD_RESULT_SPECS = {
    "ipw": {
        "primary": ["effect_estimate"],
        "main": [],
        "diagnostics": ["love_plot", "weight_histogram", "effective_sample_size"],
        "advanced": ["effect_distribution", "influence_diagnostics"],
        "expected": ["effect_estimate", "love_plot", "weight_histogram", "effective_sample_size"],
    },
    "g_formula": {
        "primary": ["effect_estimate"],
        "main": ["predicted_outcome_curves", "observed_predicted_plot", "counterfactual_summary"],
        "diagnostics": [],
        "advanced": ["effect_distribution"],
        "expected": [
            "effect_estimate",
            "predicted_outcome_curves",
            "observed_predicted_plot",
            "counterfactual_summary",
        ],
    },
    "doubly_robust": {
        "primary": ["effect_estimate"],
        "main": ["predicted_outcome_curves"],
        "diagnostics": ["love_plot", "weight_histogram", "effective_sample_size"],
        "advanced": ["effect_distribution", "influence_diagnostics"],
        "expected": [
            "effect_estimate",
            "love_plot",
            "weight_histogram",
            "predicted_outcome_curves",
            "effective_sample_size",
        ],
    },
}


def resolve_result_method(method, dataset):
    method = normalize_analysis_method(method)
    if method == "dag_auto":
        return "doubly_robust" if causal_covariate_columns(dataset) else "g_formula"
    return method


def result_method_spec(method):
    return METHOD_RESULT_SPECS.get(resolve_result_method(method, pd.DataFrame()), METHOD_RESULT_SPECS["g_formula"])


def build_method_result_outputs(dataset, effect_model, config, method):
    method = resolve_result_method(method, dataset)
    propensity = build_propensity_diagnostics(dataset)
    counterfactual = build_counterfactual_predictions(dataset, effect_model)
    all_outputs = {
        "effect_estimate": build_effect_estimate_output(effect_model, config, method),
        "love_plot": build_love_plot_output(dataset, propensity),
        "weight_histogram": build_weight_histogram_output(propensity),
        "effective_sample_size": build_effective_sample_size_output(propensity, len(dataset)),
        "predicted_outcome_curves": build_predicted_outcome_curves_output(counterfactual),
        "observed_predicted_plot": build_observed_predicted_output(counterfactual),
        "counterfactual_summary": build_counterfactual_summary_output(counterfactual, effect_model),
        "effect_distribution": build_effect_distribution_output(effect_model),
        "influence_diagnostics": build_influence_diagnostics_output(dataset, propensity, counterfactual),
    }
    spec = METHOD_RESULT_SPECS[method]
    validation_errors = validate_method_outputs(method, spec["expected"], all_outputs)

    def collect(section):
        return [
            all_outputs[output_id]
            for output_id in spec[section]
            if is_result_output_non_empty(all_outputs.get(output_id))
        ]

    return {
        "method": method,
        "method_label": analysis_method_label(method),
        "primary": collect("primary"),
        "main": collect("main"),
        "diagnostics": collect("diagnostics"),
        "advanced": collect("advanced"),
        "validation_errors": validation_errors,
    }


def build_effect_estimate_output(effect_model, config, method):
    latest = effect_model.get("latest_effect", {})
    return {
        "id": "effect_estimate",
        "kind": "effect_estimate",
        "title": "Treatment effect estimate",
        "subtitle": f"{build_exposure_label(config)} on {config['sleep_sensor']}",
        "method_label": analysis_method_label(method),
        "estimate": latest.get("contemporaneous"),
        "lower": latest.get("lower"),
        "upper": latest.get("upper"),
        "date": latest.get("date"),
        "interpretation": build_treatment_effect_interpretation(latest, config, method),
    }


def build_propensity_diagnostics(dataset):
    exposure = dataset["exposure"].astype(int)
    covariates = causal_covariate_columns(dataset)
    probabilities = estimate_exposure_probabilities(dataset, covariates)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.03, 0.97)
    exposure_values = exposure.to_numpy(dtype=float)
    weights = exposure_values / probabilities + (1 - exposure_values) / (1 - probabilities)
    weights = np.clip(weights, 0, np.nanpercentile(weights, 98) if len(weights) else 0)
    return {
        "covariates": covariates,
        "probabilities": pd.Series(probabilities, index=dataset.index),
        "weights": pd.Series(weights, index=dataset.index),
    }


def estimate_exposure_probabilities(dataset, covariates):
    exposure = dataset["exposure"].astype(int)
    base_rate = float(exposure.mean())
    fallback = np.full(len(dataset), np.clip(base_rate, 0.03, 0.97), dtype=float)
    if exposure.nunique() < 2 or not covariates or len(dataset) < 12:
        return fallback
    try:
        features = dataset[covariates].astype(float)
        model = fit_logistic_regression(features, exposure, iterations=900, learning_rate=0.05, l2=0.01)
        probabilities = predict_logistic_regression(features, model)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return fallback
    if not np.all(np.isfinite(probabilities)):
        return fallback
    return probabilities


def causal_covariate_columns(dataset):
    return [
        column
        for column in dataset.columns
        if column not in {"outcome", "exposure", "exposure_raw"}
    ]


def build_love_plot_output(dataset, propensity):
    rows = []
    exposure = dataset["exposure"].astype(int)
    weights = propensity["weights"]
    for column in propensity["covariates"]:
        values = dataset[column].astype(float)
        before = standardized_mean_difference(values, exposure)
        after = standardized_mean_difference(values, exposure, weights)
        rows.append(
            {
                "label": humanize_result_label(column),
                "before": round(before, 3),
                "after": round(after, 3),
                "before_width": round(min(abs(before), 1.0) * 100, 1),
                "after_width": round(min(abs(after), 1.0) * 100, 1),
            }
        )
    return {
        "id": "love_plot",
        "kind": "love_plot",
        "title": "Covariate balance Love Plot",
        "subtitle": "Absolute standardized mean differences before and after weighting.",
        "rows": rows,
    }


def standardized_mean_difference(values, exposure, weights=None):
    exposed_mask = exposure == 1
    control_mask = exposure == 0
    if exposed_mask.sum() == 0 or control_mask.sum() == 0:
        return 0.0

    exposed_values = values[exposed_mask].astype(float)
    control_values = values[control_mask].astype(float)
    if weights is None:
        exposed_mean = float(exposed_values.mean())
        control_mean = float(control_values.mean())
        exposed_var = float(exposed_values.var(ddof=0))
        control_var = float(control_values.var(ddof=0))
    else:
        exposed_weights = weights[exposed_mask].astype(float)
        control_weights = weights[control_mask].astype(float)
        exposed_mean = weighted_mean(exposed_values, exposed_weights)
        control_mean = weighted_mean(control_values, control_weights)
        exposed_var = weighted_variance(exposed_values, exposed_weights, exposed_mean)
        control_var = weighted_variance(control_values, control_weights, control_mean)

    denominator = np.sqrt(max((exposed_var + control_var) / 2, 0))
    if denominator == 0:
        return 0.0 if exposed_mean == control_mean else 1.0
    return abs((exposed_mean - control_mean) / denominator)


def weighted_mean(values, weights):
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / weight_sum)


def weighted_variance(values, weights, mean):
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return float(np.var(values))
    return float(np.sum(weights * (values - mean) ** 2) / weight_sum)


def build_weight_histogram_output(propensity):
    weights = propensity["weights"].dropna().astype(float)
    return {
        "id": "weight_histogram",
        "kind": "histogram",
        "title": "Weight distribution histogram",
        "subtitle": "Inverse probability weights used to rebalance observed exposure histories.",
        "x_label": "Weight",
        "bins": build_numeric_histogram_bins(weights),
    }


def build_numeric_histogram_bins(values, bins=10):
    clean = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.empty:
        return []
    minimum = float(clean.min())
    maximum = float(clean.max())
    if minimum == maximum:
        return [
            {
                "label": round(minimum, 3),
                "count": int(len(clean)),
                "height": 100,
            }
        ]
    counts, edges = np.histogram(clean.to_numpy(), bins=min(bins, max(4, len(clean) // 4)))
    max_count = int(max(counts)) or 1
    rows = []
    for index, count in enumerate(counts):
        rows.append(
            {
                "label": f"{edges[index]:.2f}-{edges[index + 1]:.2f}",
                "count": int(count),
                "height": round((int(count) / max_count) * 100, 1),
            }
        )
    return rows


def build_effective_sample_size_output(propensity, row_count):
    weights = propensity["weights"].dropna().astype(float).to_numpy()
    weight_sum = float(np.sum(weights)) if len(weights) else 0.0
    denominator = float(np.sum(weights**2)) if len(weights) else 0.0
    ess = (weight_sum**2 / denominator) if denominator > 0 else 0.0
    return {
        "id": "effective_sample_size",
        "kind": "ess_summary",
        "title": "Effective sample size summary",
        "subtitle": "How much information remains after weighting.",
        "rows": [
            {"label": "Observed days", "value": int(row_count)},
            {"label": "Effective sample size", "value": round(float(ess), 1)},
            {"label": "Mean weight", "value": round(float(np.mean(weights)), 3) if len(weights) else None},
            {"label": "Max weight", "value": round(float(np.max(weights)), 3) if len(weights) else None},
        ],
    }


def build_counterfactual_predictions(dataset, effect_model):
    latest_effect = float(effect_model.get("latest_effect", {}).get("contemporaneous", 0.0) or 0.0)
    effect_by_date = {
        pd.to_datetime(row["date"]): float(row.get("effect", latest_effect))
        for row in effect_model.get("effects", [])
    }
    observed = dataset["outcome"].astype(float)
    rolling_baseline = observed.rolling(7, min_periods=1).mean().shift(1).fillna(observed.expanding().mean())
    points = []
    for date, row in dataset.iterrows():
        effect = effect_by_date.get(pd.to_datetime(date), latest_effect)
        baseline = float(rolling_baseline.loc[date])
        no_exposure = baseline
        exposure = baseline + effect
        observed_value = float(row["outcome"])
        predicted = no_exposure + float(row["exposure"]) * effect
        points.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "no_exposure": no_exposure,
                "exposure": exposure,
                "observed": observed_value,
                "predicted": predicted,
            }
        )
    return points


def build_predicted_outcome_curves_output(points):
    return {
        "id": "predicted_outcome_curves",
        "kind": "line_chart",
        "title": "Predicted outcome curves under each intervention",
        "subtitle": "Estimated outcome path if exposure were set to no exposure versus exposure.",
        "chart": build_multi_series_chart(
            points,
            [
                {"key": "no_exposure", "label": "No exposure", "class": "series-control"},
                {"key": "exposure", "label": "Exposure (Independent)", "class": "series-treatment"},
            ],
        ),
    }


def build_observed_predicted_output(points):
    return {
        "id": "observed_predicted_plot",
        "kind": "line_chart",
        "title": "Observed vs predicted outcome plot",
        "subtitle": "Observed outcomes compared with fitted predictions from the outcome model.",
        "chart": build_multi_series_chart(
            points,
            [
                {"key": "observed", "label": "Observed", "class": "series-observed"},
                {"key": "predicted", "label": "Predicted", "class": "series-predicted"},
            ],
        ),
    }


def build_counterfactual_summary_output(points, effect_model):
    if not points:
        rows = []
    else:
        no_exposure = np.asarray([point["no_exposure"] for point in points], dtype=float)
        exposure = np.asarray([point["exposure"] for point in points], dtype=float)
        rows = [
            {"component": "No exposure", "mean": round(float(np.mean(no_exposure)), 3), "detail": "Mean predicted outcome under A=0"},
            {"component": "Exposure (Independent)", "mean": round(float(np.mean(exposure)), 3), "detail": "Mean predicted outcome under A=1"},
            {
                "component": "Contrast",
                "mean": round(float(np.mean(exposure - no_exposure)), 3),
                "detail": (
                    f"Latest 95% CI {effect_model['latest_effect'].get('lower')} to "
                    f"{effect_model['latest_effect'].get('upper')}"
                ),
            },
        ]
    return {
        "id": "counterfactual_summary",
        "kind": "counterfactual_table",
        "title": "Counterfactual outcome summary table",
        "subtitle": "Mean modeled outcomes under each intervention strategy.",
        "rows": rows,
    }


def build_effect_distribution_output(effect_model):
    effects = [row.get("effect") for row in effect_model.get("effects", [])]
    return {
        "id": "effect_distribution",
        "kind": "histogram",
        "title": "Effect estimate distribution",
        "subtitle": "Distribution of recursive effect estimates over the analysis window.",
        "x_label": "Outcome (Dependent) unit effect",
        "bins": build_numeric_histogram_bins(effects),
    }


def build_influence_diagnostics_output(dataset, propensity, points):
    if not points:
        rows = []
    else:
        residuals = np.asarray(
            [float(point["observed"]) - float(point["predicted"]) for point in points],
            dtype=float,
        )
        weights = propensity["weights"].to_numpy(dtype=float)
        influence = np.abs(residuals * weights)
        dates = [point["date"] for point in points]
        order = np.argsort(influence)[-min(5, len(influence)) :][::-1]
        rows = [
            {
                "label": dates[index],
                "value": round(float(influence[index]), 3),
                "width": round(float(influence[index] / max(np.max(influence), 1e-9)) * 100, 1),
            }
            for index in order
        ]
    return {
        "id": "influence_diagnostics",
        "kind": "influence_diagnostics",
        "title": "Influence diagnostics",
        "subtitle": "Largest weighted residual contributions.",
        "rows": rows,
    }


def build_multi_series_chart(points, series):
    if not points:
        return {"points": [], "series": [], "x_ticks": [], "y_ticks": [], "minimum": 0, "maximum": 0}
    values = []
    for point in points:
        for row in series:
            value = point.get(row["key"])
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
    if not values:
        return {"points": [], "series": [], "x_ticks": [], "y_ticks": [], "minimum": 0, "maximum": 0}
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum or 1.0
    last_index = max(len(points) - 1, 1)
    chart_points = []
    for index, point in enumerate(points):
        chart_point = {"date": point["date"], "date_short": point["date"][5:], "x": round((index / last_index) * 100, 3)}
        for row in series:
            value = float(point[row["key"]])
            chart_point[row["key"]] = round(value, 3)
            chart_point[f"{row['key']}_y"] = round(100 - ((value - minimum) / span) * 100, 3)
        chart_points.append(chart_point)
    chart_series = []
    for row in series:
        key = row["key"]
        chart_series.append(
            {
                **row,
                "polyline": " ".join(
                    f"{point['x'] * 10},{point[f'{key}_y'] * 10}" for point in chart_points
                ),
            }
        )
    return {
        "points": chart_points,
        "series": chart_series,
        "x_ticks": [
            {"x": chart_points[index]["x"], "label": chart_points[index]["date_short"]}
            for index in sorted({0, len(chart_points) // 2, len(chart_points) - 1})
        ],
        "y_ticks": [
            {"y": 0, "label": round(maximum, 2)},
            {"y": 50, "label": round((minimum + maximum) / 2, 2)},
            {"y": 100, "label": round(minimum, 2)},
        ],
        "minimum": round(minimum, 2),
        "maximum": round(maximum, 2),
    }


def validate_method_outputs(method, expected_ids, all_outputs):
    errors = []
    labels = {
        "effect_estimate": "Treatment effect estimate with confidence interval",
        "love_plot": "Covariate balance Love Plot",
        "weight_histogram": "Weight distribution histogram",
        "effective_sample_size": "Effective sample size summary",
        "predicted_outcome_curves": "Predicted outcome curves",
        "observed_predicted_plot": "Observed vs predicted outcome plot",
        "counterfactual_summary": "Counterfactual outcome summary table",
    }
    for output_id in expected_ids:
        output = all_outputs.get(output_id)
        if is_result_output_non_empty(output):
            continue
        reason = result_output_failure_reason(output_id, output)
        errors.append(
            {
                "output_id": output_id,
                "label": labels.get(output_id, output_id.replace("_", " ").title()),
                "message": f"{labels.get(output_id, output_id)} failed for {analysis_method_label(method)}: {reason}",
            }
        )
    return errors


def result_output_failure_reason(output_id, output):
    if output is None:
        return "the output object was not created."
    if output_id == "effect_estimate":
        return "the estimate or confidence interval was missing."
    if output_id == "love_plot":
        return "no covariates were available to compare before and after weighting."
    if output_id == "weight_histogram":
        return "no finite inverse probability weights were generated."
    if output_id == "effective_sample_size":
        return "the weighting model produced an empty or zero effective sample size."
    if output_id in {"predicted_outcome_curves", "observed_predicted_plot"}:
        return "the outcome model did not produce plottable predicted values."
    if output_id == "counterfactual_summary":
        return "the counterfactual summary table had no rows."
    return "the output was empty."


def is_result_output_non_empty(output):
    if not output:
        return False
    kind = output.get("kind")
    if kind == "effect_estimate":
        return all(output.get(key) is not None for key in ["estimate", "lower", "upper"])
    if kind == "love_plot":
        return bool(output.get("rows"))
    if kind == "histogram":
        return bool(output.get("bins")) and sum(int(row.get("count", 0)) for row in output["bins"]) > 0
    if kind == "ess_summary":
        rows = output.get("rows", [])
        ess_row = next((row for row in rows if row.get("label") == "Effective sample size"), None)
        return bool(ess_row and ess_row.get("value") and float(ess_row["value"]) > 0)
    if kind == "line_chart":
        chart = output.get("chart", {})
        return bool(chart.get("points")) and bool(chart.get("series"))
    if kind == "counterfactual_table":
        return bool(output.get("rows"))
    if kind == "influence_diagnostics":
        return bool(output.get("rows"))
    return True


def humanize_result_label(value):
    text = str(value or "").split(".")[-1]
    for prefix in ["nick_r_", "sensor_", "binary_sensor_"]:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.replace("_", " ").strip().title() or str(value)


def build_prediction_chart(points):
    if not points:
        return {"points": [], "polyline": "", "x_ticks": [], "y_ticks": []}

    last_index = max(len(points) - 1, 1)
    chart_points = []
    for index, point in enumerate(points):
        x_percent = (index / last_index) * 100
        y_percent = 100 - (point["probability"] * 100)
        chart_points.append(
            {
                **point,
                "x": round(x_percent, 3),
                "y": round(y_percent, 3),
                "date_short": point["date"][5:],
            }
        )

    tick_indexes = sorted({0, len(points) // 2, len(points) - 1})
    return {
        "points": chart_points,
        "polyline": " ".join(f"{point['x'] * 10},{point['y'] * 10}" for point in chart_points),
        "x_ticks": [
            {
                "x": chart_points[index]["x"],
                "label": chart_points[index]["date_short"],
            }
            for index in tick_indexes
        ],
        "y_ticks": [
            {"y": 0, "label": "100%"},
            {"y": 25, "label": "75%"},
            {"y": 50, "label": "50%"},
            {"y": 75, "label": "25%"},
            {"y": 100, "label": "0%"},
        ],
    }


def build_roc_curve(actual, probabilities):
    actual = np.asarray(actual, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    positives = int(np.sum(actual == 1))
    negatives = int(np.sum(actual == 0))
    if positives == 0 or negatives == 0:
        return {"auc": None, "points": [], "polyline": "", "x_ticks": [], "y_ticks": []}

    thresholds = sorted(set(probabilities.tolist()), reverse=True)
    points = [{"fpr": 0.0, "tpr": 0.0, "x": 0.0, "y": 1000.0}]
    for threshold in thresholds:
        predicted = probabilities >= threshold
        true_positive = int(np.sum((predicted == 1) & (actual == 1)))
        false_positive = int(np.sum((predicted == 1) & (actual == 0)))
        tpr = true_positive / positives
        fpr = false_positive / negatives
        points.append(
            {
                "fpr": round(fpr, 3),
                "tpr": round(tpr, 3),
                "x": round(fpr * 1000, 3),
                "y": round((1 - tpr) * 1000, 3),
            }
        )
    points.append({"fpr": 1.0, "tpr": 1.0, "x": 1000.0, "y": 0.0})
    points = sorted({(point["fpr"], point["tpr"]): point for point in points}.values(), key=lambda row: row["fpr"])
    auc = float(np.trapz([point["tpr"] for point in points], [point["fpr"] for point in points]))
    return {
        "auc": round(auc, 3),
        "points": points,
        "polyline": " ".join(f"{point['x']},{point['y']}" for point in points),
        "x_ticks": [
            {"x": 0, "label": "0%"},
            {"x": 50, "label": "50%"},
            {"x": 100, "label": "100%"},
        ],
        "y_ticks": [
            {"y": 0, "label": "100%"},
            {"y": 50, "label": "50%"},
            {"y": 100, "label": "0%"},
        ],
    }


def build_logistic_equation_terms(feature_columns, model):
    terms = []
    for index, column in enumerate(feature_columns):
        terms.append(
            {
                "entity": column,
                "coefficient": round(float(model["coefficients"][index + 1]), 4),
                "mean": round(float(model["means"][index]), 4),
                "std": round(float(model["stds"][index]), 4),
            }
        )
    return terms


def build_logistic_equation_text(terms, intercept):
    pieces = [f"logit(p) = {float(intercept):.4f}"]
    for term in terms:
        sign = "+" if term["coefficient"] >= 0 else "-"
        pieces.append(
            f" {sign} {abs(term['coefficient']):.4f} * z({term['entity']})"
        )
    pieces.append("; p = 1 / (1 + exp(-logit(p)))")
    return "".join(pieces)


def build_logistic_equation_latex(terms, intercept):
    pieces = [f"\\operatorname{{logit}}(p) = {float(intercept):.4f}"]
    for term in terms:
        sign = "+" if term["coefficient"] >= 0 else "-"
        pieces.append(
            f" {sign} {abs(term['coefficient']):.4f}\\,z(\\text{{{latex_escape_text(term['entity'])}}})"
        )
    return {
        "logit": "".join(pieces),
        "probability": "p = \\frac{1}{1 + e^{-\\operatorname{logit}(p)}}",
    }


def latex_escape_text(value):
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def build_covariate_distributions(dataset, feature_columns):
    distributions = []
    for column in feature_columns:
        values = dataset[column].dropna()
        if values.empty:
            continue
        minimum = float(values.min())
        maximum = float(values.max())
        span = maximum - minimum or 1.0
        distributions.append(
            {
                "entity": column,
                "minimum": round(minimum, 3),
                "maximum": round(maximum, 3),
                "mean": round(float(values.mean()), 3),
                "points": [
                    {
                        "date": index.strftime("%Y-%m-%d"),
                        "value": round(float(value), 3),
                        "height_percent": round(((float(value) - minimum) / span) * 100, 1),
                    }
                    for index, value in values.items()
                ],
            }
        )
    return distributions


def build_binary_outcome(values, config):
    threshold = config["outcome_threshold"]
    if config["outcome_operator"] == "more_than":
        return values > threshold
    return values < threshold


def build_outcome_label(config):
    direction = "more than" if config["outcome_operator"] == "more_than" else "less than"
    threshold = config["outcome_threshold"]
    threshold_text = int(threshold) if float(threshold).is_integer() else threshold
    return f"{config['sleep_sensor']} {direction} {threshold_text}"


def fetch_home_assistant_history(config):
    if os.environ.get("PERSONAL_WEBSITE_USE_HOME_ASSISTANT") != "1":
        return build_demo_history(config)

    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        return build_demo_history(config)

    entities = list(
        dict.fromkeys(
            [config["sleep_sensor"]]
            + ([config["exposure_sensor"]] if config.get("exposure_sensor") else [])
            + config["predictor_entities"]
            + config["cross_validation_entities"]
            + ([config["sleep_start_sensor"]] if config.get("sleep_start_sensor") else [])
        )
    )
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=config["analysis_window_days"] + 7)
    query = urlencode(
        {
            "end_time": end.isoformat(),
            "filter_entity_id": ",".join(entities),
            "minimal_response": "",
            "no_attributes": "",
            "significant_changes_only": "0",
        }
    )
    url = (
        "http://supervisor/core/api/history/period/"
        f"{quote(start.isoformat())}?{query}"
    )
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    series = []
    for entity_history in response.json():
        if not entity_history:
            continue
        entity_id = entity_history[0].get("entity_id")
        for state in entity_history:
            numeric_state = to_float(state.get("state"))
            if numeric_state is None and entity_id != config.get("sleep_start_sensor"):
                continue
            series.append(
                {
                    "entity_id": entity_id,
                    "last_changed": pd.to_datetime(state["last_changed"], utc=True),
                    "raw_state": state.get("state"),
                    "state": numeric_state,
                }
            )
    return pd.DataFrame(series)


def build_demo_history(config):
    end_day = pd.Timestamp.now(tz="UTC").floor("D")
    days = pd.date_range(
        end=end_day,
        periods=max(120, int(config.get("analysis_window_days", 90)) + 14),
        freq="D",
        tz="UTC",
    )
    rows = []
    for index, day in enumerate(days):
        seasonal = np.sin(index / 5.0)
        weekly = np.cos(index / 11.0)
        steps = 7600 + 1350 * seasonal + 520 * weekly + (index % 6) * 95
        sleep_minutes = 410 + 32 * np.sin((index - 2) / 6.0) + 18 * weekly + max(0, steps - 8200) / 95
        panas_score = 35 + (sleep_minutes - 420) / 12 + (steps - 7600) / 650 + 3.2 * np.sin((index + 3) / 4.0)
        panas_score = max(10, min(50, round(float(panas_score))))
        day_values = {
            "sensor.nick_r_mood_score": panas_score,
            "sensor.nick_r_sleep_minutes_asleep": round(float(sleep_minutes), 1),
            "sensor.nick_r_steps": round(float(steps), 0),
        }
        for entity_id, value in day_values.items():
            rows.append(
                {
                    "entity_id": entity_id,
                    "last_changed": day + pd.Timedelta(hours=20),
                    "raw_state": str(value),
                    "state": float(value),
                }
            )

    return pd.DataFrame(rows)


def search_home_assistant_entities(query_text):
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        return []
    try:
        response = requests.get(
            "http://supervisor/core/api/states",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException:
        return []

    needle = query_text.lower()
    results = []
    for row in response.json():
        entity_id = str(row.get("entity_id", ""))
        attributes = row.get("attributes") or {}
        friendly_name = str(attributes.get("friendly_name", ""))
        if needle and needle not in entity_id.lower() and needle not in friendly_name.lower():
            continue
        results.append({"entity_id": entity_id, "label": friendly_name})
    return sorted(results, key=lambda item: item["entity_id"])[:60]


def daily_sensor_values(history, sensor, analysis_window_days):
    if history.empty:
        return pd.Series(dtype=float)
    sensor_history = history[history["entity_id"] == sensor].copy()
    daily = last_daily_values(sensor_history)
    if daily.empty:
        return pd.Series(dtype=float)
    latest_day = daily.index.max()
    start = latest_day - pd.Timedelta(days=analysis_window_days - 1)
    return daily[daily.index >= start].dropna()


def prepare_dataset(history, config):
    history = exclude_covariate_window(history, config)
    series_by_entity = {}
    sleep_history = history[history["entity_id"] == config["sleep_sensor"]].dropna(subset=["state"])
    if not sleep_history.empty:
        series_by_entity[config["sleep_sensor"]] = last_daily_values(sleep_history)

    for setting in config["predictor_settings"]:
        entity = setting["entity"]
        entity_history = history[history["entity_id"] == entity].dropna(subset=["state"])
        if entity_history.empty:
            continue
        series = aggregate_daily_values(entity_history, setting["aggregation"])
        if setting["day_offset"]:
            series = series.shift(setting["day_offset"])
        series_by_entity[entity] = series

    if not series_by_entity:
        return pd.DataFrame()

    daily = pd.concat(series_by_entity, axis=1).sort_index()
    latest_day = daily.index.max()
    start = latest_day - pd.Timedelta(days=config["analysis_window_days"] - 1)
    daily = daily[daily.index >= start]

    sleep_sensor = config["sleep_sensor"]
    features = config["predictor_entities"] + config["cross_validation_entities"]
    keep_columns = [column for column in [sleep_sensor] + features if column in daily.columns]
    daily = daily[keep_columns]

    if sleep_sensor not in daily.columns:
        return pd.DataFrame()

    if config["lag_days"]:
        lagged_features = daily[[column for column in features if column in daily.columns]].shift(
            config["lag_days"]
        )
        daily = pd.concat([daily[[sleep_sensor]], lagged_features], axis=1)

    return daily.dropna()


def prepare_causal_dataset(history, config):
    history = exclude_covariate_window(history, config)
    series_by_name = {}

    outcome_history = history[history["entity_id"] == config["sleep_sensor"]].dropna(subset=["state"])
    if not outcome_history.empty:
        series_by_name["outcome"] = last_daily_values(outcome_history)

    exposure_history = history[history["entity_id"] == config["exposure_sensor"]].dropna(subset=["state"])
    if not exposure_history.empty:
        raw_exposure = aggregate_daily_values(
            exposure_history, config.get("exposure_aggregation", "sum")
        )
        series_by_name["exposure_raw"] = raw_exposure
        series_by_name["exposure"] = build_binary_exposure(raw_exposure, config).astype(float)

    for setting in config["predictor_settings"]:
        entity = setting["entity"]
        if entity == config["exposure_sensor"]:
            continue
        entity_history = history[history["entity_id"] == entity].dropna(subset=["state"])
        if entity_history.empty:
            continue
        series = aggregate_daily_values(entity_history, setting["aggregation"])
        if setting["day_offset"]:
            series = series.shift(setting["day_offset"])
        series_by_name[entity] = series

    if not series_by_name:
        return pd.DataFrame()

    daily = pd.concat(series_by_name, axis=1).sort_index()
    latest_day = daily.index.max()
    start = latest_day - pd.Timedelta(days=config["analysis_window_days"] - 1)
    daily = daily[daily.index >= start].dropna(subset=["outcome", "exposure"])
    return daily.dropna()


def build_binary_exposure(values, config):
    threshold = config.get("exposure_threshold", 0)
    if config.get("exposure_operator") == "less_than":
        return values < threshold
    return values > threshold


def apply_balanced_exposure_threshold(dataset, config):
    raw = dataset.get("exposure_raw")
    if raw is None:
        return False
    threshold = balanced_binary_threshold(raw)
    if threshold is None:
        return False
    config["exposure_operator"] = "more_than"
    config["exposure_threshold"] = threshold
    return True


def balanced_binary_threshold(values):
    clean = values.dropna().astype(float)
    unique_values = sorted(clean.unique().tolist())
    if len(unique_values) < 2:
        return None

    best_threshold = None
    best_score = None
    for lower, upper in zip(unique_values, unique_values[1:]):
        threshold = (lower + upper) / 2
        exposed = clean > threshold
        exposed_count = int(exposed.sum())
        not_exposed_count = int((~exposed).sum())
        if not exposed_count or not not_exposed_count:
            continue
        score = abs(exposed_count - not_exposed_count)
        if best_score is None or score < best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def build_exposure_label(config):
    direction = "less than" if config.get("exposure_operator") == "less_than" else "more than"
    threshold = config.get("exposure_threshold", 0)
    threshold_text = int(threshold) if float(threshold).is_integer() else threshold
    aggregation = config.get("exposure_aggregation", "sum")
    return f"{aggregation}({config.get('exposure_sensor')}) {direction} {threshold_text}"


def format_interpretation_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    absolute = abs(number)
    if absolute >= 100:
        return f"{absolute:,.0f}"
    if absolute >= 10:
        return f"{absolute:,.1f}"
    return f"{absolute:,.2f}"


def format_threshold_value(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.1f}"


def dag_variable_label(config, sensor):
    for node in config.get("causal_dag", {}).get("nodes", []):
        if node.get("id") == sensor and node.get("label"):
            return str(node["label"])
    return humanize_result_label(sensor)


def exposure_state_text(config, comparison):
    exposure = dag_variable_label(config, config.get("exposure_sensor"))
    threshold = format_threshold_value(config.get("exposure_threshold", 0))
    operator = config.get("exposure_operator", "more_than")
    aggregation = config.get("exposure_aggregation", "last")
    aggregation_prefix = {
        "last": "",
        "sum": "daily sum of ",
        "mean": "daily mean of ",
        "min": "daily minimum of ",
        "max": "daily maximum of ",
    }.get(aggregation, f"{aggregation} of ")
    if operator == "less_than":
        exposed_relation, reference_relation = "below", "at or above"
    else:
        exposed_relation, reference_relation = "above", "at or below"
    relation = exposed_relation if comparison == "exposed" else reference_relation
    return f"{aggregation_prefix}{exposure} is {relation} {threshold}"


def outcome_unit_text(config):
    outcome = str(config.get("sleep_sensor", "")).lower()
    if "minute" in outcome:
        return "minutes"
    if "efficiency" in outcome or "percent" in outcome:
        return "percentage points"
    return "outcome units"


def signed_effect_phrase(value, unit):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    amount = format_interpretation_number(number)
    direction = "higher" if number > 0 else "lower"
    return f"about {amount} {unit} {direction}"


def confidence_interval_phrase(lower, upper, unit):
    try:
        lower_number = float(lower)
        upper_number = float(upper)
    except (TypeError, ValueError):
        return ""
    if lower_number <= 0 <= upper_number:
        return f"{signed_effect_phrase(lower_number, unit)} to {signed_effect_phrase(upper_number, unit)}"
    direction_value = lower_number if abs(lower_number) >= abs(upper_number) else upper_number
    amounts = sorted(
        [format_interpretation_number(lower_number), format_interpretation_number(upper_number)],
        key=lambda text: float(text.replace(",", "")),
    )
    direction = "higher" if direction_value > 0 else "lower"
    return f"{amounts[0]} to {amounts[1]} {unit} {direction}"


def build_treatment_effect_interpretation(latest, config, method):
    estimate = latest.get("contemporaneous")
    effect_phrase = signed_effect_phrase(estimate, outcome_unit_text(config))
    if not effect_phrase:
        return ""
    method_label = analysis_method_label(method)
    outcome = dag_variable_label(config, config.get("sleep_sensor"))
    exposed_state = exposure_state_text(config, "exposed")
    reference_state = exposure_state_text(config, "reference")
    ci_phrase = confidence_interval_phrase(
        latest.get("lower"),
        latest.get("upper"),
        outcome_unit_text(config),
    )
    sentence = (
        f"Interpretation: Under the DAG-defined {method_label}, days when {exposed_state} "
        f"are estimated to have {outcome} that is {effect_phrase} than otherwise similar days when "
        f"{reference_state}."
    )
    if ci_phrase:
        sentence += f" The 95% confidence interval is {ci_phrase}."
    return sentence


def fit_dynamic_effect_model(dataset, config):
    max_lag = int(config["max_lag_days"])
    terms = ["intercept", "outcome_lag1"] + [f"exposure_lag{lag}" for lag in range(max_lag + 1)]
    covariates = [
        column
        for column in dataset.columns
        if column not in {"outcome", "exposure", "exposure_raw"}
    ]
    terms.extend([f"{column}_lag1" for column in covariates])

    rows = []
    for date, row in dataset.iterrows():
        position = dataset.index.get_loc(date)
        if position < max_lag + 1:
            continue
        features = [1.0, float(dataset["outcome"].iloc[position - 1])]
        features.extend(float(dataset["exposure"].iloc[position - lag]) for lag in range(max_lag + 1))
        features.extend(float(dataset[column].iloc[position - 1]) for column in covariates)
        rows.append((date, float(row["outcome"]), np.asarray(features, dtype=float)))

    if len(rows) < max(12, len(terms) + 4):
        return {"effects": [], "latest_effect": {}, "terms": terms}

    coefficients = np.zeros(len(terms), dtype=float)
    covariance = np.eye(len(terms), dtype=float) * 100.0
    forgetting = float(config["state_forgetting"])
    effects = []

    for date, observed, features in rows:
        prediction = float(features @ coefficients)
        denominator = forgetting + float(features @ covariance @ features.T)
        gain = (covariance @ features) / denominator
        coefficients = coefficients + gain * (observed - prediction)
        covariance = (covariance - np.outer(gain, features) @ covariance) / forgetting
        exposure_coefficients = {
            lag: float(coefficients[terms.index(f"exposure_lag{lag}")])
            for lag in range(max_lag + 1)
        }
        standard_errors = {
            lag: float(np.sqrt(max(covariance[terms.index(f"exposure_lag{lag}"), terms.index(f"exposure_lag{lag}")], 0)))
            for lag in range(max_lag + 1)
        }
        effects.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "effect": round(exposure_coefficients[0], 4),
                "lag1_effect": round(exposure_coefficients.get(1, 0.0), 4),
                "step1_effect": round(sum(exposure_coefficients.get(lag, 0.0) for lag in range(min(1, max_lag) + 1)), 4),
                "lower": round(exposure_coefficients[0] - 1.96 * standard_errors[0], 4),
                "upper": round(exposure_coefficients[0] + 1.96 * standard_errors[0], 4),
                "lags": {
                    str(lag): {
                        "effect": round(exposure_coefficients[lag], 4),
                        "lower": round(exposure_coefficients[lag] - 1.96 * standard_errors[lag], 4),
                        "upper": round(exposure_coefficients[lag] + 1.96 * standard_errors[lag], 4),
                    }
                    for lag in range(max_lag + 1)
                },
            }
        )

    latest = effects[-1]
    latest_lags = {int(lag): values for lag, values in latest["lags"].items()}
    impulse = []
    running_total = 0.0
    for lag in range(max_lag + 1):
        value = latest_lags[lag]["effect"]
        running_total += value
        impulse.append(
            {
                "lag": lag,
                "effect": value,
                "lower": latest_lags[lag]["lower"],
                "upper": latest_lags[lag]["upper"],
            }
        )
    step = []
    running_total = 0.0
    for lag in range(max_lag + 1):
        running_total += latest_lags[lag]["effect"]
        step.append({"lag": lag, "effect": round(running_total, 4)})

    strategies = build_general_strategies(latest_lags, max_lag)
    forest_rows = build_forest_rows(latest_lags)
    return {
        "effects": effects,
        "latest_effect": {
            "date": latest["date"],
            "contemporaneous": latest["effect"],
            "lower": latest["lower"],
            "upper": latest["upper"],
            "lag1": latest["lag1_effect"],
            "step1": latest["step1_effect"],
            "cumulative": round(sum(row["effect"] for row in impulse), 4),
        },
        "impulse_response": build_lag_chart(impulse),
        "step_response": build_lag_chart(step),
        "forest_plot": build_forest_plot(forest_rows),
        "general_strategies": strategies,
        "terms": terms,
    }


def build_general_strategies(latest_lags, max_lag):
    strategy_length = min(7, max_lag + 1)
    patterns = [
        ("Early", [1 if index < min(3, strategy_length) else 0 for index in range(strategy_length)]),
        ("Alternating", [1 if index % 2 == 0 and sum(1 for j in range(index + 1) if j % 2 == 0) <= 3 else 0 for index in range(strategy_length)]),
        ("Spaced", [1 if index in {0, 3, 6} and index < strategy_length else 0 for index in range(strategy_length)]),
    ]
    rows = []
    for name, pattern in patterns:
        effect = sum(pattern[lag] * latest_lags[lag]["effect"] for lag in range(strategy_length))
        rows.append(
            {
                "name": name,
                "pattern": "".join(str(value) for value in pattern),
                "effect": round(float(effect), 4),
            }
        )
    return rows


def build_forest_rows(latest_lags):
    rows = []
    for lag, values in sorted(latest_lags.items()):
        rows.append(
            {
                "label": "Same day" if lag == 0 else f"Lag {lag} day",
                "effect": values["effect"],
                "lower": values["lower"],
                "upper": values["upper"],
            }
        )
    return rows


def build_forest_plot(rows):
    if not rows:
        return {"rows": [], "zero_x": 50}
    values = []
    for row in rows:
        values.extend([float(row["lower"]), float(row["upper"]), 0.0])
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum or 1.0
    chart_rows = []
    for row in rows:
        chart_rows.append(
            {
                **row,
                "x": round(((float(row["effect"]) - minimum) / span) * 100, 3),
                "lower_x": round(((float(row["lower"]) - minimum) / span) * 100, 3),
                "upper_x": round(((float(row["upper"]) - minimum) / span) * 100, 3),
            }
        )
    return {
        "rows": chart_rows,
        "zero_x": round(((0.0 - minimum) / span) * 100, 3),
        "minimum": round(minimum, 2),
        "maximum": round(maximum, 2),
    }


def build_positivity_diagnostics(exposure, max_lag):
    values = [int(value) for value in exposure.dropna().astype(int).tolist()]
    rows = []
    for length in range(1, max_lag + 2):
        observed = {
            "".join(str(item) for item in values[index : index + length])
            for index in range(0, max(0, len(values) - length + 1))
        }
        possible = 2**length
        percent = (len(observed) / possible) * 100 if possible else 0
        rows.append(
            {
                "duration": length,
                "observed": len(observed),
                "possible": possible,
                "percent": round(percent, 1),
                "width_percent": round(percent, 1),
            }
        )
    return rows


def build_effect_chart(points, value_key):
    if not points:
        return {"points": [], "polyline": "", "zero_y": 50, "x_ticks": [], "y_ticks": []}
    values = [float(point[value_key]) for point in points]
    lower_values = [float(point.get("lower", point[value_key])) for point in points]
    upper_values = [float(point.get("upper", point[value_key])) for point in points]
    minimum = min(values + lower_values + [0.0])
    maximum = max(values + upper_values + [0.0])
    span = maximum - minimum or 1.0
    last_index = max(len(points) - 1, 1)
    chart_points = []
    for index, point in enumerate(points):
        lower = float(point.get("lower", point[value_key]))
        upper = float(point.get("upper", point[value_key]))
        chart_points.append(
            {
                **point,
                "x": round((index / last_index) * 100, 3),
                "y": round(100 - ((float(point[value_key]) - minimum) / span) * 100, 3),
                "lower_y": round(100 - ((lower - minimum) / span) * 100, 3),
                "upper_y": round(100 - ((upper - minimum) / span) * 100, 3),
            }
        )
    return {
        "points": chart_points,
        "polyline": " ".join(f"{point['x'] * 10},{point['y'] * 10}" for point in chart_points),
        "zero_y": round(100 - ((0 - minimum) / span) * 100, 3),
        "x_ticks": [
            {"x": chart_points[index]["x"], "label": chart_points[index]["date"][5:]}
            for index in sorted({0, len(chart_points) // 2, len(chart_points) - 1})
        ],
        "y_ticks": [
            {"y": 0, "label": round(maximum, 2)},
            {"y": round(100 - ((0 - minimum) / span) * 100, 3), "label": 0},
            {"y": 100, "label": round(minimum, 2)},
        ],
    }


def build_lag_chart(points):
    if not points:
        return {"points": [], "polyline": "", "zero_y": 50, "x_ticks": [], "y_ticks": []}
    values = [float(point["effect"]) for point in points]
    minimum = min(values + [0.0])
    maximum = max(values + [0.0])
    span = maximum - minimum or 1.0
    last_index = max(len(points) - 1, 1)
    chart_points = []
    for index, point in enumerate(points):
        chart_points.append(
            {
                **point,
                "x": round((index / last_index) * 100, 3),
                "y": round(100 - ((float(point["effect"]) - minimum) / span) * 100, 3),
            }
        )
    return {
        "points": chart_points,
        "polyline": " ".join(f"{point['x'] * 10},{point['y'] * 10}" for point in chart_points),
        "zero_y": round(100 - ((0 - minimum) / span) * 100, 3),
        "x_ticks": [{"x": point["x"], "label": point["lag"]} for point in chart_points],
        "y_ticks": [
            {"y": 0, "label": round(maximum, 2)},
            {"y": round(100 - ((0 - minimum) / span) * 100, 3), "label": 0},
            {"y": 100, "label": round(minimum, 2)},
        ],
    }


def exclude_covariate_window(history, config):
    sleep_start_sensor = config.get("sleep_start_sensor")
    excluded_minutes = (
        config.get("exclude_before_sleep_start_hours", 0) * 60
        + config.get("exclude_before_sleep_start_minutes", 0)
    )
    if history.empty or not sleep_start_sensor or excluded_minutes <= 0:
        return history

    covariates = set(
        config["predictor_entities"]
        + config["cross_validation_entities"]
        + ([config["exposure_sensor"]] if config.get("exposure_sensor") else [])
    )
    if not covariates:
        return history

    starts = history[history["entity_id"] == sleep_start_sensor]
    intervals = []
    for _, row in starts.iterrows():
        start_at = parse_sleep_start_time(row.get("raw_state"), row["last_changed"])
        if start_at is None:
            continue
        intervals.append((start_at - pd.Timedelta(minutes=excluded_minutes), start_at))

    if not intervals:
        return history

    keep = pd.Series(True, index=history.index)
    covariate_rows = history["entity_id"].isin(covariates)
    for window_start, window_end in intervals:
        in_window = history["last_changed"].between(window_start, window_end, inclusive="both")
        keep &= ~(covariate_rows & in_window)
    return history[keep]


def parse_sleep_start_time(value, fallback_datetime):
    if value is None:
        return None
    text = str(value).strip()
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    if not pd.isna(parsed):
        return parsed

    parts = text.split(":")
    if len(parts) < 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return None
    base = fallback_datetime.floor("D")
    start_at = base + pd.Timedelta(hours=hour, minutes=minute)
    if start_at > fallback_datetime + pd.Timedelta(hours=6):
        start_at -= pd.Timedelta(days=1)
    return start_at


def last_daily_values(history):
    if history.empty:
        return pd.Series(dtype=float)
    return (
        history.assign(day=history["last_changed"].dt.date)
        .sort_values("last_changed")
        .groupby("day")["state"]
        .last()
        .pipe(lambda series: series.set_axis(pd.to_datetime(series.index)))
    )


def aggregate_daily_values(history, aggregation):
    if history.empty:
        return pd.Series(dtype=float)
    grouped = (
        history.assign(day=history["last_changed"].dt.date)
        .sort_values("last_changed")
        .groupby("day")["state"]
    )
    if aggregation == "sum":
        series = grouped.sum()
    elif aggregation == "min":
        series = grouped.min()
    elif aggregation == "max":
        series = grouped.max()
    elif aggregation == "mean":
        series = grouped.mean()
    else:
        series = grouped.last()
    return series.set_axis(pd.to_datetime(series.index))


def to_float(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"on", "open"}:
            return 1.0
        if normalized in {"off", "closed"}:
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fit_logistic_regression(x, y, iterations=1200, learning_rate=0.08, l2=0.001):
    values = x.to_numpy(dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1.0
    scaled = (values - means) / stds
    matrix = np.column_stack([np.ones(len(scaled)), scaled])
    target = y.to_numpy(dtype=float)
    coefficients = np.zeros(matrix.shape[1], dtype=float)

    for _ in range(iterations):
        probabilities = sigmoid(matrix @ coefficients)
        gradient = matrix.T @ (probabilities - target) / len(target)
        gradient[1:] += l2 * coefficients[1:]
        coefficients -= learning_rate * gradient

    return {"coefficients": coefficients, "means": means, "stds": stds}


def predict_logistic_regression(x, model):
    values = x.to_numpy(dtype=float)
    scaled = (values - model["means"]) / model["stds"]
    matrix = np.column_stack([np.ones(len(scaled)), scaled])
    return sigmoid(matrix @ model["coefficients"])


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -30, 30)))


def time_series_binary_scores(x, y):
    split_count = min(5, max(2, len(x) // 7))
    fold_size = len(x) // (split_count + 1)
    scores = []
    for fold in range(1, split_count + 1):
        train_end = fold * fold_size
        test_end = min(len(x), train_end + fold_size)
        if test_end <= train_end or y.iloc[:train_end].nunique() < 2:
            continue
        model = fit_logistic_regression(x.iloc[:train_end], y.iloc[:train_end])
        probabilities = predict_logistic_regression(x.iloc[train_end:test_end], model)
        actual = y.iloc[train_end:test_end].to_numpy(dtype=int)
        predictions = probabilities >= 0.5
        scores.append(
            {
                "accuracy": float(np.mean(predictions == actual)),
                "brier": float(np.mean((probabilities - actual) ** 2)),
            }
        )
    if scores:
        return scores
    rate = float(np.mean(y.to_numpy(dtype=int)))
    return [{"accuracy": max(rate, 1 - rate), "brier": rate * (1 - rate)}]


def single_feature_binary_importance(x, y):
    importances = {}
    target = y.to_numpy(dtype=int)
    rate = float(np.mean(target))
    baseline = np.mean((np.full(len(target), rate) - target) ** 2)
    for column in x.columns:
        model = fit_logistic_regression(x[[column]], y)
        probabilities = predict_logistic_regression(x[[column]], model)
        score = np.mean((probabilities - target) ** 2)
        importances[column] = max(0.0, baseline - score)
    return importances

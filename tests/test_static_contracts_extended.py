import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "page_name",
    ["index.html", "new-projects.html", "causal-dag.html", "agentic-prompt.html"],
)
def test_static_pages_keep_personal_site_navigation(page_name):
    html = (ROOT / page_name).read_text(encoding="utf-8")

    assert "Nick Rezaee" in html
    assert "New Projects" in html
    assert "Experience" in html
    assert "Skills" in html
    assert "Contact" in html


@pytest.mark.parametrize(
    "forbidden",
    [
        "Unable to load histogram.",
        "Sensor not found",
        "Home Assistant",
        "Saving DAG",
        "Choose from DAG",
        "Advanced results",
        "Effect estimate distribution",
        "effect distribution",
        "direction is uncertain",
        "Independent Variable",
        "Dependent Variable",
        "Mood Score",
    ],
)
def test_static_causal_dag_snapshot_excludes_removed_or_confusing_copy(forbidden):
    html = (ROOT / "causal-dag.html").read_text(encoding="utf-8")

    assert forbidden not in html


def test_static_causal_dag_snapshot_default_graph_is_panas_sleep_steps():
    html = (ROOT / "causal-dag.html").read_text(encoding="utf-8")
    labels = re.findall(r'<strong class="dag-node-title" data-node-title>(.*?)</strong>', html)
    edges = set(
        re.findall(
            r'name="edge_source" value="([^"]+)" data-edge-source>\s+<input name="edge_target" value="([^"]+)"',
            html,
        )
    )

    assert labels == ["PANAS Score (t-1)", "Sleep", "Steps", "PANAS Score"]
    assert edges == {
        ("derived.nick_r_mood_score_lag1", "sensor.nick_r_sleep_minutes_asleep"),
        ("derived.nick_r_mood_score_lag1", "sensor.nick_r_steps"),
        ("sensor.nick_r_sleep_minutes_asleep", "sensor.nick_r_mood_score"),
        ("sensor.nick_r_steps", "sensor.nick_r_mood_score"),
    }


def test_static_causal_dag_analysis_method_has_single_clear_recommended_selection():
    html = (ROOT / "causal-dag.html").read_text(encoding="utf-8")
    select_match = re.search(r'<select name="analysis_method" data-analysis-method>(.*?)</select>', html, flags=re.S)

    assert select_match is not None
    select_html = select_match.group(1)
    assert 'value="g_formula" selected' in select_html
    assert select_html.count("selected") == 1
    assert "Recommended: G-formula / outcome model" in select_html
    assert "Default" not in select_html
    assert "dag_auto" not in select_html


def test_app_javascript_uses_run_analysis_copy_instead_of_saving_dag_copy():
    script = (ROOT / "sleep_causal_app" / "static" / "app.js").read_text(encoding="utf-8")

    assert "Running analysis" in script
    assert "Saving DAG" not in script


def test_app_javascript_finishes_guide_after_guided_analysis_run():
    script = (ROOT / "sleep_causal_app" / "static" / "app.js").read_text(encoding="utf-8")

    assert "onboardingResumeAfterAnalysis = true" in script
    assert "if (onboardingResumeAfterAnalysis)" in script
    assert "finishOnboarding();" in script
    assert 'nextLabel: "Run Analysis"' in script


def test_onboarding_styles_use_spotlight_dot_without_card_triangle():
    css = (ROOT / "sleep_causal_app" / "static" / "styles.css").read_text(encoding="utf-8")

    assert ".onboarding-spotlight::after" in css
    assert re.search(r"\.onboarding-card::(?:before|after)", css) is None
    assert "onboardingClick" in css


def test_architecture_diagram_uses_capitalized_meds_label():
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    assert ">Meds<" in html
    assert "VA EHR" not in html
    assert ">meds<" not in html


def test_agentic_prompt_development_pill_is_large_red_and_links_to_github():
    html = (ROOT / "agentic-prompt.html").read_text(encoding="utf-8")
    css = (ROOT / "agentic-prompt.css").read_text(encoding="utf-8")

    assert 'href="https://github.com/resace3/Agentic_Prompt_App"' in html
    assert "Still in development: view on GitHub" in html
    assert ".dev-note" in css
    assert "font-size: 1.8rem" in css
    assert "font-weight: 900" in css
    assert "rgba(225, 29, 72" in css


def test_static_causal_dag_advanced_controls_stay_under_closed_advanced_toggle():
    html = (ROOT / "causal-dag.html").read_text(encoding="utf-8")

    assert '<details class="sidebar-advanced-settings">' in html
    assert "<summary>Advanced</summary>" in html
    assert "<summary>Advanced</summary>" in html.split("Daily aggregate", 1)[0]
    assert "Daily aggregate" in html
    assert "From time" in html
    assert "To time" in html

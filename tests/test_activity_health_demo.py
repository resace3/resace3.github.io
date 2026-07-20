import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "activity-health-demo.html").read_text(encoding="utf-8")
SCRIPT = (ROOT / "activity-health-demo.js").read_text(encoding="utf-8")


def test_demo_exposes_three_keyboard_navigable_views():
    assert HTML.count('role="tab"') == 3
    assert HTML.count('role="tabpanel"') == 3
    assert 'data-tab="overview"' in HTML
    assert 'data-tab="rhythm"' in HTML
    assert 'data-tab="configuration"' in HTML
    assert 'event.key === "ArrowRight"' in SCRIPT
    assert 'event.key === "ArrowLeft"' in SCRIPT
    assert 'event.key === "Home"' in SCRIPT
    assert 'event.key === "End"' in SCRIPT


def test_demo_uses_only_fictional_profile_and_virtual_activity_copy():
    required_copy = [
        "Synthetic people and virtual step data only",
        "fictional activity profile",
        "Virtual activity data",
        "Fictional health history",
        "All names, entities, and values on this page are fictional",
    ]
    for phrase in required_copy:
        assert phrase in HTML

    assert "sensor.nick_r_steps" not in HTML
    assert "Connected to Home Assistant" not in HTML
    assert "stored locally in Home Assistant" not in HTML


def test_demo_configuration_drives_activity_and_risk_calculations():
    for field in [
        "step_profile",
        "timezone",
        "age",
        "sex",
        "bmi",
        "self_health",
        "smoking",
        "alcohol",
        "hypertension",
        "diabetes",
        "heart_attack",
        "stroke",
        "cancer",
    ]:
        assert f'name="{field}"' in HTML
        assert f"profile.{field}" in SCRIPT

    assert "calculateRisk(weeklyTotal / 7)" in SCRIPT
    assert "activeWindow(activity.hourly)" in SCRIPT
    assert "periodOpportunity(activity.hourly)" in SCRIPT
    assert "window.__activityHealthDemoState" in SCRIPT


def test_demo_has_complete_deterministic_chart_inputs():
    assert 'const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]' in SCRIPT
    assert len(re.findall(r"^\s{6}daily: \[", SCRIPT, flags=re.M)) == 3
    assert len(re.findall(r"^\s{6}prior: \[", SCRIPT, flags=re.M)) == 3
    assert len(re.findall(r"^\s{6}hourly: \[", SCRIPT, flags=re.M)) == 3
    assert "weekNoise" in SCRIPT
    assert "Math.random" not in SCRIPT


def test_removed_day_comparison_is_not_reintroduced():
    assert "Day comparison" not in HTML
    assert "No earlier baseline is available" not in HTML
    assert "Unavailable" not in HTML
    assert "comparison-morning" not in SCRIPT

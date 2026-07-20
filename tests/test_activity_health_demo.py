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
    assert len(re.findall(r"^\s{6}daily: \[", SCRIPT, flags=re.M)) == 1
    assert len(re.findall(r"^\s{6}prior: \[", SCRIPT, flags=re.M)) == 1
    assert len(re.findall(r"^\s{6}hourly: \[", SCRIPT, flags=re.M)) == 1
    assert "weekNoise" in SCRIPT
    assert "Math.random" not in SCRIPT


def test_demo_offers_one_step_pattern_and_maps_low_risk_to_left_side_of_gauge():
    assert HTML.count('<option value="steady">Steady daytime walker</option>') == 1
    assert "Weekend explorer" not in HTML + SCRIPT
    assert "Evening mover" not in HTML + SCRIPT
    assert "const gaugeSweep = Math.min(100, risk * 5);" in SCRIPT
    assert "rotate(${gaugeSweep * 1.8}deg)" in SCRIPT
    assert "-90 +" not in SCRIPT


def test_removed_day_comparison_is_not_reintroduced():
    assert "Day comparison" not in HTML
    assert "No earlier baseline is available" not in HTML
    assert "Unavailable" not in HTML
    assert "comparison-morning" not in SCRIPT


def test_activity_rhythm_includes_a_real_fourier_series_fit():
    assert "Fourier series fit" in HTML
    assert "3-harmonic fit" in HTML
    assert 'data-fourier-chart' in HTML
    assert 'data-fourier-line' in HTML
    assert 'data-fourier-points' in HTML
    assert 'data-fourier-score' in HTML
    assert 'data-fourier-peak' in HTML
    assert "solveLinearSystem" in SCRIPT
    assert "fitFourierSeries(hourly, 3)" in SCRIPT
    assert "2 * Math.PI * harmonic * hour / 24" in SCRIPT
    assert "rSquared" in SCRIPT

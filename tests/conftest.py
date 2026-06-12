import os

import pytest

from sleep_causal_app import create_app


@pytest.fixture()
def isolated_app(tmp_path, monkeypatch):
    monkeypatch.setenv("SLEEP_ANALYSIS_CONFIG", str(tmp_path / "sleep_analysis.json"))
    monkeypatch.delenv("PERSONAL_WEBSITE_USE_HOME_ASSISTANT", raising=False)
    monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)

    app = create_app()
    app.config.update(TESTING=True)
    return app


@pytest.fixture()
def isolated_client(isolated_app):
    return isolated_app.test_client()


@pytest.fixture()
def config_path():
    return os.environ["SLEEP_ANALYSIS_CONFIG"]

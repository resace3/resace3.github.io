from __future__ import annotations

import os
from pathlib import Path

from sleep_causal_app import create_app


CONFIG_PATH = Path("/tmp/personal_website_sleep_causal_config.json")

os.environ.setdefault("SLEEP_ANALYSIS_CONFIG", str(CONFIG_PATH))

app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5051, debug=False, use_reloader=False)

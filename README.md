# Nick Rezaee Personal Website

Personal portfolio site for Nick Rezaee, with project pages for open source health apps and embedded demos for:

- Causal DAG Builder / N-of-1 Causal Analysis Engine
- Agentic Prompt App concept page

The repository is published from `main` through GitHub Pages at `resace3.github.io`.

## Project Structure

- `index.html`, `new-projects.html`, `agentic-prompt.html`: public site pages
- `style.css`, `agentic-prompt.css`, `script.js`: site styling and page transitions
- `sleep_causal_app/`: copied Flask app used for the Causal DAG Builder demo
- `run_sleep_causal_copy.py`: starts the copied Flask app on `127.0.0.1:5051`
- `dev_proxy.py`: serves the personal website on `0.0.0.0:8000` and proxies `/causal-dag.html` to the Flask app
- `.github/workflows/ci.yml`: GitHub Actions workflow for static, Python, and browser checks

## Local Development

Install Python and Node dependencies:

```bash
python3 -m pip install -r requirements.txt
npm ci
npx playwright install chromium
```

Run the copied Causal DAG app:

```bash
python3 run_sleep_causal_copy.py
```

In a second terminal, run the personal website proxy:

```bash
python3 dev_proxy.py
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/new-projects.html`
- `http://127.0.0.1:8000/agentic-prompt.html`
- `http://127.0.0.1:8000/causal-dag.html`

To expose the local site through ngrok:

```bash
ngrok http 8000
```

## Verification

Run Python compile checks:

```bash
python3 -m compileall -q dev_proxy.py run_sleep_causal_copy.py sleep_causal_app tests
```

Run Python tests:

```bash
python3 -m pytest
```

Run browser smoke tests:

```bash
npx playwright test
```

On this Home Assistant Alpine/ARM environment, Playwright's downloaded browser may not launch. Use the system Chromium override:

```bash
PLAYWRIGHT_CHROMIUM_EXECUTABLE=/usr/bin/chromium npx playwright test
```

## GitHub Actions

The CI workflow runs on:

- every push
- every pull request targeting `main`

It checks:

- Python dependency install
- Python compile validity
- Flask route and demo-data smoke tests
- static page asset references
- Playwright browser smoke tests for the home page, project pages, Agentic Prompt page, and proxied Causal DAG app

To make sure `main` only updates after checks pass, configure GitHub branch protection for `main` with required pull requests and required status checks.


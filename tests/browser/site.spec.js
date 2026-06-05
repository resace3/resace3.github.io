const { expect, test } = require("@playwright/test");

const pageErrors = new WeakMap();

async function dismissOnboarding(page) {
  await page.evaluate(() => {
    if (typeof window.finishOnboarding === "function") {
      window.finishOnboarding();
      return;
    }
    const overlay = document.querySelector("[data-onboarding-overlay]");
    if (overlay) overlay.hidden = true;
    document.body.classList.remove("is-onboarding-active");
  }).catch(() => {});
  const skip = page.locator("[data-onboarding-skip]");
  if (await skip.isVisible().catch(() => false)) {
    await skip.click();
  }
}

async function expandCausalAnalysis(page) {
  const toggle = page.locator("[data-analysis-tools-toggle]");
  await expect(toggle).toBeAttached({ timeout: 10000 });
  await page.evaluate(() => {
    const expand =
      window.setAnalysisToolsExpanded ||
      window.setStaticAnalysisToolsExpanded ||
      ((expanded) => {
        const section = document.querySelector("[data-analysis-section]");
        const body = document.querySelector("[data-analysis-tools-body]");
        const toggle = document.querySelector("[data-analysis-tools-toggle]");
        if (!section || !body || !toggle) return;
        toggle.setAttribute("aria-expanded", String(expanded));
        body.hidden = false;
        section.classList.toggle("is-analysis-collapsed", !expanded);
      });
    expand(true);
  });
  await expect(page.locator("[data-analysis-form]")).toBeVisible({ timeout: 10000 });
}

test.beforeEach(async ({ page }) => {
  const errors = [];
  pageErrors.set(page, errors);

  await page.addInitScript(() => {
    window.personalWebsiteDisableOnboarding = true;
  });

  await page.route("https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/javascript",
      body: "window.MathJax = window.MathJax || {}; window.MathJax.typesetPromise = () => Promise.resolve();",
    })
  );

  page.on("console", (message) => {
    const text = message.text();
    const isBrowserFaviconNoise =
      message.type() === "error" &&
      text === "Failed to load resource: the server responded with a status of 404 (File not found)";

    if (isBrowserFaviconNoise) {
      return;
    }

    if (message.type() === "error") {
      errors.push(text);
    }
  });

  page.on("pageerror", (error) => {
    errors.push(error.message);
  });
});

test.afterEach(async ({ page }) => {
  expect(pageErrors.get(page)).toEqual([]);
});

test("home page renders the personal site", async ({ page }) => {
  await page.goto("/");

  await expect(page).toHaveTitle(/Nick Rezaee/);
  await expect(page.getByRole("link", { name: "Nick Rezaee home" })).toBeVisible();
  await expect(page.getByText("Biomedical Data")).toBeVisible();
  await expect(page.getByRole("link", { name: "New Projects" }).first()).toBeVisible();
});

test("new projects page links to both app concepts", async ({ page }) => {
  await page.goto("/new-projects.html");

  await expect(page.getByRole("heading", { name: "Open Source Health Apps" })).toBeVisible();
  await expect(page.getByRole("link", { name: /Causal DAG Builder/ })).toBeVisible();
  await expect(page.getByRole("link", { name: /Agentic Prompt App/ })).toBeVisible();
});

test("agentic prompt page renders the static prompt animation", async ({ page }) => {
  await page.goto("/agentic-prompt.html");

  await expect(page.getByRole("heading", { name: "Agentic Prompt App" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Still in development: view on GitHub" })).toBeVisible();
  await expect(page.locator(".typed-line")).toHaveAttribute(
    "aria-label",
    "Create health plots for sleep duration, steps, and mood over the last 90 days."
  );
  await expect(page.getByRole("heading", { name: "Sleep Health Plots" })).toBeVisible();
  await expect(page.getByText("Generated Notes")).toBeVisible();
});

test("causal DAG app is served through the personal site proxy", async ({ page }) => {
  await page.goto("/causal-dag.html");

  await expect(page.getByRole("link", { name: "Nick Rezaee home" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "N of 1 Causal Analysis Engine" })).toBeVisible();
  await expect(page.locator(".workflow-step-label", { hasText: "Step 1" })).toBeVisible();
  await expect(page.getByRole("button", { name: /Guide me/i })).toBeVisible();
  await expect(page.getByRole("button", { name: /Run analysis/i })).toBeVisible();
});

test("static causal DAG snapshot runs analysis without a backend", async ({ page }) => {
  test.setTimeout(90000);
  await page.goto("/__static/causal-dag.html", { waitUntil: "domcontentloaded" });
  await dismissOnboarding(page);
  await expandCausalAnalysis(page);

  await page.evaluate(() => {
    const originalFetch = window.fetch.bind(window);
    window.__staticFetchCalls = [];
    window.fetch = (input, init) => {
      const requestUrl =
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.href
            : input?.url || String(input || "");
      window.__staticFetchCalls.push({ url: requestUrl, method: init?.method || "GET" });
      return originalFetch(input, init);
    };
  });

  const setupButtons = await page.locator(".workbench-controls button:visible").evaluateAll((buttons) =>
    buttons.map((button) => button.innerText.trim() || button.getAttribute("aria-label") || button.title)
  );
  expect(setupButtons.at(-1)).toBe("Run Analysis");

  const startingUrl = page.url();
  await dismissOnboarding(page);
  await page.locator("[data-run-analysis-button]").evaluate((button) => button.click());
  await page.waitForTimeout(150);
  await expect(page.locator("[data-run-analysis-button]")).not.toContainText(/Saving DAG/i);
  const staticResults = page.locator("[data-analysis-results]");
  await expect(staticResults).toContainText("Sleep duration and low mood", { timeout: 15000 });
  const staticResultText = await staticResults.innerText();
  expect(staticResultText).toContain("95% CI -0.63 to -0.18");
  expect(staticResultText).toContain("Predicted probability of low mood");
  expect(staticResultText).not.toContain("Run the model from the setup panel");
  expect(page.url()).toBe(startingUrl);

  const fetchPaths = await page.evaluate(() =>
    window.__staticFetchCalls.map((call) => new URL(call.url, window.location.href).pathname)
  );
  expect(fetchPaths).toContain("/dag/autosave");
  expect(fetchPaths).toContain("/run");
});

test("causal DAG proxy run analysis renders output in place", async ({ page }) => {
  test.setTimeout(240000);
  await page.goto("/causal-dag.html", { waitUntil: "domcontentloaded" });
  await dismissOnboarding(page);
  await expandCausalAnalysis(page);

  const startingUrl = page.url();
  await dismissOnboarding(page);
  await page.locator("[data-run-analysis-button]").scrollIntoViewIfNeeded();
  await page.locator("[data-run-analysis-button]").evaluate((button) => button.click());

  const results = page.locator("[data-analysis-results]");
  await expect(results).toContainText("Treatment effect estimate", { timeout: 120000 });
  const resultText = await results.innerText();
  expect(resultText).toContain("95% CI");
  expect(resultText).toContain("Observed vs predicted outcome plot");
  expect(resultText).not.toContain("Run the model from the setup panel");
  expect(page.url()).toBe(startingUrl);
});

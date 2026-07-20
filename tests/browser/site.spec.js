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
    const disableMotion = () => {
      const style = document.createElement("style");
      style.dataset.testMotionOverride = "true";
      style.textContent = `
        *, *::before, *::after {
          animation-duration: 1ms !important;
          animation-delay: 0s !important;
          transition-duration: 1ms !important;
          transition-delay: 0s !important;
          scroll-behavior: auto !important;
        }
      `;
      document.head?.appendChild(style);
    };
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", disableMotion, { once: true });
    } else {
      disableMotion();
    }
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

test("new projects page links to all interactive app concepts", async ({ page }) => {
  await page.goto("/new-projects.html");

  await expect(page.getByRole("heading", { name: "Open Source Health Apps" })).toBeVisible();
  await expect(page.getByRole("link", { name: /Causal DAG Builder/ })).toBeVisible();
  await expect(page.getByRole("link", { name: /Agentic Prompt App/ })).toBeVisible();
  await expect(page.getByRole("link", { name: /Activity Health Insights/ })).toBeVisible();
});

test("activity health demo recalculates every view from fictional configuration", async ({ page }) => {
  await page.goto("/activity-health-demo.html");

  await expect(page.getByText("Synthetic people and virtual step data only")).toBeVisible();
  await expect(page.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("[data-weekly-chart] .weekly-bar")).toHaveCount(7);

  const initial = await page.evaluate(() => window.__activityHealthDemoState());
  expect(initial.profile.step_profile).toBe("steady");
  expect(initial.daily).toHaveLength(7);
  expect(initial.hourly).toHaveLength(24);
  expect(initial.risk).toBeGreaterThan(0);

  await page.getByRole("button", { name: "About this model estimate" }).click();
  await expect(page.getByRole("dialog").getByRole("heading", { name: "What is this estimate?" })).toBeVisible();
  await page.getByRole("button", { name: "Close model explanation" }).click();

  await page.getByRole("tab", { name: "Configuration" }).click();
  await page.locator('[name="step_profile"]').selectOption("evening");
  await page.locator('[name="timezone"]').selectOption("Europe/London");
  await page.locator('[name="age"]').fill("68");
  await page.locator('[name="bmi"]').fill("32");
  await page.locator('[name="sex"]').selectOption("male");
  await page.locator('[name="smoking"]').selectOption("current");
  await page.locator('[name="hypertension"]').selectOption("yes");
  await page.locator('[name="diabetes"]').selectOption("yes");
  await page.locator('[name="self_health"]').selectOption("fair");
  await page.getByRole("button", { name: "Save and recalculate demo" }).click();
  await expect(page.getByText("Demo recalculated.")).toBeVisible();

  const configured = await page.evaluate(() => window.__activityHealthDemoState());
  expect(configured.risk).toBeGreaterThan(initial.risk);
  expect(configured.weeklyTotal).not.toBe(initial.weeklyTotal);
  expect(configured.profile.timezone).toBe("Europe/London");

  await page.getByRole("tab", { name: "Overview" }).click();
  await expect(page.locator("[data-risk-value]")).toHaveText(`${configured.risk.toFixed(1)}%`);
  await expect(page.locator("[data-weekly-steps]")).toHaveText(configured.weeklyTotal.toLocaleString());

  await page.getByRole("button", { name: "Generate another virtual week" }).click();
  const regenerated = await page.evaluate(() => window.__activityHealthDemoState());
  expect(regenerated.weekVariant).toBe(1);
  expect(regenerated.weeklyTotal).not.toBe(configured.weeklyTotal);

  await page.getByRole("tab", { name: "Activity Rhythm" }).click();
  await expect(page.locator("[data-hourly-chart] .hourly-bar")).toHaveCount(24);
  await expect(page.locator("[data-hourly-copy]")).toContainText("Europe/London");
  await expect(page.locator("[data-active-window]")).not.toHaveText("Loading…");
  await page.getByRole("button", { name: "About average hourly steps" }).click();
  await expect(page.getByRole("dialog").getByRole("heading", { name: "How is the rhythm built?" })).toBeVisible();
  await page.getByRole("button", { name: "Close activity explanation" }).click();

  await page.getByRole("tab", { name: "Activity Rhythm" }).press("End");
  await expect(page.getByRole("tab", { name: "Configuration" })).toHaveAttribute("aria-selected", "true");
  await page.getByRole("button", { name: "Reset fictional profile" }).click();
  await expect(page.getByText("Fictional defaults restored.")).toBeVisible();
  const reset = await page.evaluate(() => window.__activityHealthDemoState());
  expect(reset.profile.age).toBe(42);
  expect(reset.profile.step_profile).toBe("steady");
  expect(reset.weekVariant).toBe(0);
  expect(reset.risk).toBeCloseTo(initial.risk, 8);
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
  await page.goto("/causal-dag.html", { waitUntil: "domcontentloaded" });

  await expect(page.getByRole("link", { name: "Nick Rezaee home" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "N of 1 Causal Analysis Engine" })).toBeVisible();
  await expect(page.locator(".workflow-step-label", { hasText: "Step 1" })).toBeVisible();
  await expect(page.getByRole("button", { name: /Guide me/i })).toBeVisible();
  await expect(page.getByRole("button", { name: /Run analysis/i })).toBeVisible();
  await expect(page.locator(".brand .guide-tab")).toHaveCount(1);
  await expect(page.locator(".brand .brand-mark")).toHaveCount(0);
  await expect(page.locator(".app-topbar .tabs")).toHaveCount(0);
  await expect(page.locator(".tabs .guide-tab")).toHaveCount(0);
  await expect(page.locator(".app-topbar").getByRole("link", { name: "Analysis" })).toHaveCount(0);
  await expect(page.locator('[data-sidebar-field="node_role"] option[value="exposure"]')).toHaveText(
    "Exposure (Independent)"
  );
  await expect(page.locator('[data-sidebar-field="node_role"] option[value="outcome"]')).toHaveText(
    "Outcome (Dependent)"
  );
  await expect(page.locator(".dag-node.exposure [data-node-role-shadow]")).toContainText(
    "Exposure (Independent)"
  );
  await expect(page.locator(".dag-node.exposure [data-node-role-shadow]")).not.toHaveText(/^\(/);
  await expect(page.locator("[data-dag-node] [data-node-title]")).toHaveText([
    "PANAS Score (t-1)",
    "Sleep",
    "Steps",
    "PANAS Score",
  ]);
  await expect(page.locator(".dag-node.exposure [data-node-title]")).toHaveText("Sleep");
  await expect(page.locator(".dag-node.outcome [data-node-role-shadow]")).toContainText(
    "Outcome (Dependent)"
  );
  await expect(page.locator(".dag-node.outcome [data-node-role-shadow]")).not.toHaveText(/^\(/);
  await expect(page.locator(".dag-node.outcome [data-node-title]")).toHaveText("PANAS Score");
  await expect(page.locator(".dag-node.outcome [data-node-threshold]")).toHaveValue(/^33(?:\.0)?$/);
  await expect(page.getByText("Sensor not found")).toHaveCount(0);
  const sidebarState = await page.evaluate(() => {
    const advanced = document.querySelector(".sidebar-advanced-settings");
    return {
      label: document.documentElement.textContent.includes("Simulated variable"),
      advancedExists: Boolean(advanced),
      advancedOpen: advanced?.open ?? null,
      advancedText: advanced?.textContent || "",
    };
  });
  expect(sidebarState.label).toBe(true);
  expect(sidebarState.advancedExists).toBe(true);
  expect(sidebarState.advancedOpen).toBe(false);
  expect(sidebarState.advancedText).toContain("Daily aggregate");
  expect(sidebarState.advancedText).toContain("From time");
  expect(sidebarState.advancedText).toContain("To time");
  const guideBox = await page.locator(".brand .guide-tab").boundingBox();
  const titleBox = await page.locator(".brand strong").boundingBox();
  expect(guideBox.x + guideBox.width).toBeLessThanOrEqual(titleBox.x);
  const guideStyle = await page.locator(".guide-tab").evaluate((button) => {
    const style = window.getComputedStyle(button);
    return {
      animationName: style.animationName,
      backgroundImage: style.backgroundImage,
      boxShadow: style.boxShadow,
      color: style.color,
    };
  });
  expect(guideStyle.animationName).toContain("guideTabBounce");
  expect(guideStyle.backgroundImage).toContain("linear-gradient");
  expect(guideStyle.boxShadow).not.toBe("none");
  expect(guideStyle.color).toBe("rgb(255, 255, 255)");
});

test("static causal DAG snapshot runs analysis without a backend", async ({ page }) => {
  test.setTimeout(180000);
  await page.goto("/__static/causal-dag.html", { waitUntil: "domcontentloaded" });
  await dismissOnboarding(page);
  await expandCausalAnalysis(page);

  await expect(page.locator("[data-analysis-method] option:checked")).toHaveText(
    "Recommended: G-formula / outcome model"
  );
  await expect(page.locator("[data-analysis-method] option", { hasText: "Choose from DAG" })).toHaveCount(0);
  await expect(page.getByText("Already selected for this demo. Change only if you want a specific causal estimator.")).toBeVisible();
  await expect(page.locator("[data-equation-explain-toggle]")).toHaveText("Explain method");
  await expect(page.locator("[data-method-equation]")).not.toContainText("\\[");
  await expect(page.locator("[data-equation-box]")).toHaveCount(0);

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
  const runButtonText = await page.locator("[data-run-analysis-button]").evaluate((button) => button.textContent || "");
  expect(runButtonText).not.toMatch(/Saving DAG/i);
  const staticResults = page.locator("[data-analysis-results]");
  const staticSummary = staticResults.locator(".clinician-summary-card");
  await expect(staticSummary).toBeVisible({ timeout: 30000 });
  await expect(staticSummary).toContainText("Clinician summary", { timeout: 30000 });
  await expect(staticSummary).toContainText("Sleep may contribute to current PANAS Score changes", { timeout: 30000 });
  await expect(staticSummary).toContainText("Prior PANAS Score may affect Sleep", { timeout: 30000 });
  await expect(staticSummary).toContainText("Prior PANAS Score may affect Steps", { timeout: 30000 });
  await expect(staticResults).not.toContainText("Advanced results");
  await expect(staticResults).not.toContainText("Effect estimate distribution");
  const staticTechnicalDetails = staticResults.locator(".technical-results-details");
  expect(await staticTechnicalDetails.evaluate((details) => details.open)).toBe(false);
  await expect(
    staticResults.getByRole("heading", { name: "Predicted PANAS Score" })
  ).toBeHidden();
  await staticTechnicalDetails.locator(":scope > summary").click();
  expect(await staticTechnicalDetails.evaluate((details) => details.open)).toBe(true);
  await expect(staticResults).toContainText("95% CI -0.54 to 1.99");
  await expect(staticResults).toContainText("Predicted PANAS Score", { timeout: 15000 });
  await expect(staticResults).not.toContainText("Run the model from the setup panel");
  expect(page.url()).toBe(startingUrl);

  const fetchPaths = await page.evaluate(() =>
    window.__staticFetchCalls.map((call) => new URL(call.url, window.location.href).pathname)
  );
  expect(fetchPaths).toContain("/dag/autosave");
  expect(fetchPaths).toContain("/run");
});

test("static causal DAG guide closes after guided analysis run", async ({ page }) => {
  test.setTimeout(120000);
  await page.goto("/__static/causal-dag.html", { waitUntil: "domcontentloaded" });
  const jumpedToRunStep = await page.evaluate(() => window.__showOnboardingStepForTest?.("Run the analysis") || false);
  expect(jumpedToRunStep).toBe(true);
  await expect(page.locator("[data-onboarding-title]")).toHaveText(/Run the analysis/i);

  await expect(page.locator("[data-onboarding-count]")).toContainText("Guide 14 of 14");
  await expect(page.locator("[data-onboarding-skip]")).toHaveText('Skip "Guide Me"');
  const guideButtonGap = await page.locator(".onboarding-actions").evaluate((actions) =>
    Number.parseFloat(window.getComputedStyle(actions).columnGap)
  );
  expect(guideButtonGap).toBeGreaterThanOrEqual(32);
  await page.evaluate(() => {
    window.__staticFetchCalls = [];
    const originalFetch = window.fetch.bind(window);
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

  await page.locator("[data-onboarding-next]").evaluate((button) => button.click());

  const results = page.locator("[data-analysis-results]");
  await expect(results).toContainText("Clinician summary", { timeout: 15000 });
  await expect(results).toContainText("Sleep may contribute to current PANAS Score changes");
  await expect(results).toContainText("Prior PANAS Score may affect Sleep");
  const resultText = await results.innerText();
  expect(resultText).toContain("Show plots and technical details");
  expect(resultText).toContain("95% CI -0.54 to 1.99");
  await expect(page.locator("[data-onboarding-overlay]")).toBeHidden({ timeout: 10000 });

  const fetchPaths = await page.evaluate(() =>
    window.__staticFetchCalls.map((call) => new URL(call.url, window.location.href).pathname)
  );
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
  await expect(results).toContainText("Clinician summary", { timeout: 120000 });
  await expect(results).toContainText("Sleep may contribute to current PANAS Score changes");
  await expect(results).toContainText("Prior PANAS Score may affect Sleep");
  await expect(results).not.toContainText("Advanced results");
  await expect(results).not.toContainText("Effect estimate distribution");
  const technicalDetails = results.locator(".technical-results-details");
  expect(await technicalDetails.evaluate((details) => details.open)).toBe(false);
  await technicalDetails.locator(":scope > summary").click();
  await expect(results).toContainText("Treatment effect estimate");
  await expect(results.getByRole("heading", { name: "Observed vs predicted outcome plot" })).toBeVisible();
  const resultText = await results.innerText();
  expect(resultText).toContain("95% CI");
  expect(resultText).toContain("Observed vs predicted outcome plot");
  expect(resultText).not.toContain("Run the model from the setup panel");
  expect(page.url()).toBe(startingUrl);
});

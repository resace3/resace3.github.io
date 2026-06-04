const { expect, test } = require("@playwright/test");

const pageErrors = new WeakMap();

test.beforeEach(async ({ page }) => {
  const errors = [];
  pageErrors.set(page, errors);

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

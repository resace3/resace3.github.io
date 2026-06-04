const { defineConfig } = require("@playwright/test");

const isCI = Boolean(process.env.CI);
const chromiumExecutable = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;

module.exports = defineConfig({
  testDir: "./tests/browser",
  timeout: 30000,
  expect: {
    timeout: 5000,
  },
  use: {
    baseURL: "http://127.0.0.1:8000",
    launchOptions: chromiumExecutable ? { executablePath: chromiumExecutable } : {},
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: "python3 run_sleep_causal_copy.py",
      url: "http://127.0.0.1:5051",
      reuseExistingServer: !isCI,
      timeout: 30000,
      env: {
        SLEEP_ANALYSIS_CONFIG: "/tmp/personal_website_sleep_causal_ci_config.json",
      },
    },
    {
      command: "python3 dev_proxy.py",
      url: "http://127.0.0.1:8000",
      reuseExistingServer: !isCI,
      timeout: 30000,
    },
  ],
});

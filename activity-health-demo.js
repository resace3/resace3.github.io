(() => {
  const root = document.querySelector("[data-health-demo]");
  if (!root) return;

  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const stepProfiles = {
    steady: {
      label: "Steady daytime walker",
      daily: [12100, 14100, 12800, 15100, 13700, 11200, 12400],
      prior: [11600, 13200, 12400, 13900, 12900, 10900, 11600],
      hourly: [20, 12, 8, 8, 18, 42, 120, 360, 820, 1180, 1510, 1420, 1190, 1030, 790, 680, 590, 510, 430, 330, 240, 160, 80, 32],
    },
    weekend: {
      label: "Weekend explorer",
      daily: [8200, 8800, 9100, 8700, 9500, 16200, 18400],
      prior: [8500, 8100, 9000, 8600, 9200, 14100, 15700],
      hourly: [18, 10, 8, 8, 12, 28, 62, 140, 320, 610, 980, 1320, 1510, 1380, 1160, 920, 700, 520, 400, 300, 210, 130, 72, 30],
    },
    evening: {
      label: "Evening mover",
      daily: [9300, 10100, 9700, 10600, 11300, 8400, 7900],
      prior: [9900, 10300, 10100, 10800, 11000, 9000, 8500],
      hourly: [12, 8, 6, 6, 10, 20, 48, 100, 190, 260, 340, 410, 460, 520, 610, 720, 960, 1260, 1510, 1420, 1180, 810, 360, 90],
    },
  };
  const weekNoise = [
    [0, 0, 0, 0, 0, 0, 0],
    [180, -320, 560, -140, 430, 820, -210],
    [-420, 380, -180, 760, -360, 540, 920],
    [610, 220, -470, 140, 780, -630, 350],
  ];

  let weekVariant = 0;
  let profile = readProfile();
  let lastRendered = null;

  function byData(name) {
    return root.querySelector(`[data-${name}]`);
  }

  function readProfile() {
    const form = root.querySelector("[data-profile-form]");
    if (!form) {
      return { step_profile: "steady", timezone: "America/New_York", age: 42, sex: "female", bmi: 24.6, race: "Fictional multiracial profile", self_health: "very_good", smoking: "never", alcohol: "no", hypertension: "no", diabetes: "no", heart_attack: "no", stroke: "no", cancer: "no" };
    }
    const data = new FormData(form);
    return {
      step_profile: String(data.get("step_profile") || "steady"),
      timezone: String(data.get("timezone") || "America/New_York"),
      age: Number(data.get("age")),
      sex: String(data.get("sex") || "female"),
      bmi: Number(data.get("bmi")),
      race: String(data.get("race") || "Fictional multiracial profile"),
      self_health: String(data.get("self_health") || "very_good"),
      smoking: String(data.get("smoking") || "never"),
      alcohol: String(data.get("alcohol") || "no"),
      hypertension: String(data.get("hypertension") || "no"),
      diabetes: String(data.get("diabetes") || "no"),
      heart_attack: String(data.get("heart_attack") || "no"),
      stroke: String(data.get("stroke") || "no"),
      cancer: String(data.get("cancer") || "no"),
    };
  }

  function currentActivity() {
    const source = stepProfiles[profile.step_profile] || stepProfiles.steady;
    const noise = weekNoise[weekVariant % weekNoise.length];
    const daily = source.daily.map((value, index) => Math.max(800, value + noise[index]));
    const hourlyFactor = 1 + (noise.reduce((sum, value) => sum + value, 0) / 7) / 20000;
    const hourly = source.hourly.map((value, index) => Math.max(0, Math.round(value * hourlyFactor * (1 + ((index + weekVariant) % 5 - 2) * 0.012))));
    return { source, daily, prior: source.prior, hourly };
  }

  function total(values) {
    return values.reduce((sum, value) => sum + value, 0);
  }

  function calculateRisk(averageDailySteps) {
    const selfHealth = { excellent: -0.15, very_good: 0, good: 0.2, fair: 0.55, poor: 1.1 }[profile.self_health] || 0;
    const smoking = { never: 0, former: 0.35, current: 1.15 }[profile.smoking] || 0;
    const risk = 0.9
      + Math.max(0, profile.age - 35) * 0.05
      + Math.max(0, profile.bmi - 23) * 0.06
      + (profile.sex === "male" ? 0.15 : 0)
      + smoking
      + (profile.alcohol === "yes" ? 0.18 : 0)
      + (profile.hypertension === "yes" ? 0.65 : 0)
      + (profile.diabetes === "yes" ? 0.9 : 0)
      + (profile.heart_attack === "yes" ? 1.4 : 0)
      + (profile.stroke === "yes" ? 1.15 : 0)
      + (profile.cancer === "yes" ? 0.8 : 0)
      + selfHealth
      - Math.min(0.75, averageDailySteps / 10000 * 0.45);
    return Math.min(18, Math.max(0.4, risk));
  }

  function formatHour(hour) {
    const normalized = ((hour % 24) + 24) % 24;
    if (normalized === 0) return "12 AM";
    if (normalized === 12) return "12 PM";
    return `${normalized > 12 ? normalized - 12 : normalized} ${normalized >= 12 ? "PM" : "AM"}`;
  }

  function activeWindow(hourly) {
    let bestStart = 0;
    let bestTotal = -1;
    for (let start = 0; start < 24; start += 1) {
      let windowTotal = 0;
      for (let offset = 0; offset < 4; offset += 1) windowTotal += hourly[(start + offset) % 24];
      if (windowTotal > bestTotal) {
        bestTotal = windowTotal;
        bestStart = start;
      }
    }
    return { start: bestStart, end: (bestStart + 4) % 24, total: bestTotal };
  }

  function renderWeekly(daily) {
    const chart = byData("weekly-chart");
    const max = Math.max(...daily);
    chart.replaceChildren(...daily.map((value, index) => {
      const column = document.createElement("div");
      column.className = "weekly-bar-column";
      column.setAttribute("aria-label", `${days[index]} ${value.toLocaleString()} steps`);
      const bar = document.createElement("span");
      bar.className = "weekly-bar";
      bar.style.height = `${Math.max(4, value / max * 100)}%`;
      bar.title = `${value.toLocaleString()} steps`;
      const label = document.createElement("span");
      label.textContent = days[index];
      column.append(bar, label);
      return column;
    }));
  }

  function renderHourly(hourly) {
    const chart = byData("hourly-chart");
    const max = Math.max(...hourly);
    chart.replaceChildren(...hourly.map((value, hour) => {
      const bar = document.createElement("span");
      bar.className = "hourly-bar";
      bar.style.height = `${Math.max(1, value / max * 100)}%`;
      bar.title = `${formatHour(hour)}: ${value.toLocaleString()} average steps`;
      bar.setAttribute("aria-label", bar.title);
      return bar;
    }));
  }

  function pointsFor(values, width, height, padding = 4) {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(1, max - min);
    return values.map((value, index) => ({
      x: padding + index / Math.max(1, values.length - 1) * (width - padding * 2),
      y: padding + (1 - (value - min) / span) * (height - padding * 2),
    }));
  }

  function pathFromPoints(points) {
    return points.map((point, index) => `${index ? "L" : "M"}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");
  }

  function renderSparkline(daily) {
    const points = pointsFor(daily, 620, 105, 7);
    byData("spark-line").setAttribute("d", pathFromPoints(points));
    byData("spark-area").setAttribute("d", `${pathFromPoints(points)} L613,105 L7,105 Z`);
    const pointGroup = byData("spark-points");
    pointGroup.replaceChildren(...points.map((point) => {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", point.x.toFixed(1));
      circle.setAttribute("cy", point.y.toFixed(1));
      circle.setAttribute("r", "3.7");
      return circle;
    }));
  }

  function renderSurvival(risk) {
    const points = Array.from({ length: 11 }, (_, year) => ({ x: year * 100, y: (1 - (1 - risk / 100 * Math.pow(year / 10, 1.35))) * 240 }));
    const line = pathFromPoints(points);
    byData("survival-line").setAttribute("d", line);
    byData("survival-area").setAttribute("d", `${line} L1000,240 L0,240 Z`);
  }

  function periodOpportunity(hourly) {
    const periods = [
      { key: "morning", start: 6, end: 12 },
      { key: "afternoon", start: 12, end: 18 },
      { key: "evening", start: 18, end: 24 },
    ];
    return periods.map((period) => ({ ...period, value: total(hourly.slice(period.start, period.end)) })).sort((a, b) => a.value - b.value)[0].key;
  }

  function render() {
    const activity = currentActivity();
    const weeklyTotal = total(activity.daily);
    const priorTotal = total(activity.prior);
    const weeklyChange = (weeklyTotal - priorTotal) / priorTotal * 100;
    const risk = calculateRisk(weeklyTotal / 7);
    const window = activeWindow(activity.hourly);
    const opportunity = periodOpportunity(activity.hourly);

    byData("weekly-steps").textContent = weeklyTotal.toLocaleString();
    const changeElement = byData("weekly-change");
    changeElement.textContent = `${weeklyChange >= 0 ? "↑" : "↓"} ${Math.abs(weeklyChange).toFixed(1)}% vs prior synthetic week`;
    changeElement.style.color = weeklyChange >= 0 ? "var(--health-green)" : "var(--health-red)";
    byData("risk-value").textContent = `${risk.toFixed(1)}%`;
    const riskContext = risk < 2 ? "Lower simulated range" : risk < 5 ? "Middle simulated range" : "Higher simulated range";
    byData("risk-context").textContent = riskContext;
    byData("model-range").textContent = `${Math.max(0.2, risk * 0.78).toFixed(1)}%–${Math.min(25, risk * 1.34).toFixed(1)}%`;
    byData("gauge-value").style.strokeDasharray = `${Math.min(100, risk * 5)} 100`;
    byData("gauge-needle").style.transform = `rotate(${-90 + Math.min(100, risk * 5) * 1.8}deg)`;
    byData("risk-gauge").setAttribute("aria-label", `Synthetic 10-year model estimate ${risk.toFixed(1)} percent, ${riskContext.toLowerCase()}`);
    byData("active-window").textContent = `${formatHour(window.start)} – ${formatHour(window.end)}`;
    byData("hourly-copy").textContent = `Virtual seven-day average · ${profile.timezone}`;
    byData("strongest-pattern").textContent = `This profile moves most during ${formatHour(window.start)} – ${formatHour(window.end)}.`;
    const opportunityCopy = {
      morning: "A short virtual morning walk would make this pattern more balanced.",
      afternoon: "A short virtual afternoon walk would make this pattern more balanced.",
      evening: "A short virtual evening walk would make this pattern more balanced.",
    }[opportunity];
    byData("opportunity").textContent = opportunityCopy;
    byData("rhythm-note").textContent = `${activity.source.label} profile shown in ${profile.timezone}. Generate a new week to add reproducible variation.`;
    byData("change-summary").textContent = weeklyChange >= 0
      ? `The virtual week is ${weeklyChange.toFixed(1)}% more active than its synthetic baseline.`
      : `The virtual week is ${Math.abs(weeklyChange).toFixed(1)}% less active than its synthetic baseline.`;

    renderWeekly(activity.daily);
    renderHourly(activity.hourly);
    renderSparkline(activity.daily);
    renderSurvival(risk);

    lastRendered = { profile: { ...profile }, weekVariant, weeklyTotal, priorTotal, weeklyChange, risk, activeWindow: { ...window }, opportunity, daily: [...activity.daily], hourly: [...activity.hourly] };
    window.__activityHealthDemoState = () => JSON.parse(JSON.stringify(lastRendered));
  }

  const tabs = Array.from(root.querySelectorAll("[role='tab']"));
  const panels = Array.from(root.querySelectorAll("[role='tabpanel']"));

  function activateTab(name, options = {}) {
    const target = tabs.find((tab) => tab.dataset.tab === name) || tabs[0];
    tabs.forEach((tab) => {
      const active = tab === target;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    panels.forEach((panel) => { panel.hidden = panel.dataset.panel !== target.dataset.tab; });
    if (options.focus) target.focus();
    if (options.updateHash !== false) history.replaceState(null, "", `#${target.dataset.tab}`);
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateTab(tab.dataset.tab));
    tab.addEventListener("keydown", (event) => {
      let nextIndex = null;
      if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") nextIndex = 0;
      if (event.key === "End") nextIndex = tabs.length - 1;
      if (nextIndex === null) return;
      event.preventDefault();
      activateTab(tabs[nextIndex].dataset.tab, { focus: true });
    });
  });

  root.querySelector("[data-profile-form]").addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    if (!form.reportValidity()) return;
    profile = readProfile();
    render();
    const status = byData("profile-status");
    status.textContent = "Demo recalculated.";
    window.setTimeout(() => { if (status.textContent === "Demo recalculated.") status.textContent = ""; }, 3500);
  });

  root.querySelector("[data-reset-demo]").addEventListener("click", () => {
    root.querySelector("[data-profile-form]").reset();
    weekVariant = 0;
    profile = readProfile();
    render();
    byData("profile-status").textContent = "Fictional defaults restored.";
  });

  root.querySelector("[data-generate-week]").addEventListener("click", () => {
    weekVariant = (weekVariant + 1) % weekNoise.length;
    render();
    const button = byData("generate-week");
    button.textContent = `Virtual week ${weekVariant + 1} generated`;
    window.setTimeout(() => { button.textContent = "Generate another virtual week"; }, 1800);
  });

  root.querySelectorAll("[data-open-dialog]").forEach((button) => {
    button.addEventListener("click", () => document.getElementById(button.dataset.openDialog)?.showModal());
  });
  document.querySelectorAll("[data-close-dialog]").forEach((button) => {
    button.addEventListener("click", () => button.closest("dialog")?.close());
  });
  document.querySelectorAll(".health-dialog").forEach((dialog) => {
    dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
  });

  const initialTab = location.hash.replace("#", "");
  activateTab(["overview", "rhythm", "configuration"].includes(initialTab) ? initialTab : "overview", { updateHash: false });
  render();
})();

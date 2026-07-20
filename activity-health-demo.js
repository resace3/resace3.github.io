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

  function solveLinearSystem(matrix, values) {
    const size = values.length;
    const rows = matrix.map((row, index) => [...row, values[index]]);

    for (let column = 0; column < size; column += 1) {
      let pivotRow = column;
      for (let row = column + 1; row < size; row += 1) {
        if (Math.abs(rows[row][column]) > Math.abs(rows[pivotRow][column])) pivotRow = row;
      }
      [rows[column], rows[pivotRow]] = [rows[pivotRow], rows[column]];

      const pivot = rows[column][column];
      if (Math.abs(pivot) < 1e-10) return Array(size).fill(0);
      for (let index = column; index <= size; index += 1) rows[column][index] /= pivot;

      for (let row = 0; row < size; row += 1) {
        if (row === column) continue;
        const factor = rows[row][column];
        for (let index = column; index <= size; index += 1) rows[row][index] -= factor * rows[column][index];
      }
    }

    return rows.map((row) => row[size]);
  }

  function fitFourierSeries(values, harmonics = 3) {
    const basisAt = (hour) => {
      const basis = [1];
      for (let harmonic = 1; harmonic <= harmonics; harmonic += 1) {
        const angle = 2 * Math.PI * harmonic * hour / 24;
        basis.push(Math.cos(angle), Math.sin(angle));
      }
      return basis;
    };

    const design = values.map((_value, hour) => basisAt(hour));
    const terms = design[0].length;
    const normalMatrix = Array.from({ length: terms }, () => Array(terms).fill(0));
    const normalValues = Array(terms).fill(0);
    design.forEach((row, hour) => {
      row.forEach((left, leftIndex) => {
        normalValues[leftIndex] += left * values[hour];
        row.forEach((right, rightIndex) => { normalMatrix[leftIndex][rightIndex] += left * right; });
      });
    });

    const coefficients = solveLinearSystem(normalMatrix, normalValues);
    const predict = (hour) => basisAt(hour).reduce((sum, value, index) => sum + value * coefficients[index], 0);
    const fitted = values.map((_value, hour) => predict(hour));
    const mean = total(values) / values.length;
    const residualSum = values.reduce((sum, value, index) => sum + (value - fitted[index]) ** 2, 0);
    const totalSum = values.reduce((sum, value) => sum + (value - mean) ** 2, 0);
    const rSquared = totalSum ? 1 - residualSum / totalSum : 1;
    const samples = Array.from({ length: 193 }, (_value, index) => {
      const hour = index * 24 / 192;
      return { hour, value: predict(hour) };
    });
    return { harmonics, coefficients, fitted, rSquared, samples };
  }

  function renderFourier(hourly) {
    const fit = fitFourierSeries(hourly, 3);
    const chart = byData("fourier-chart");
    const plot = { left: 24, right: 988, top: 24, bottom: 228 };
    const maximum = Math.max(1, ...hourly, ...fit.samples.map((sample) => sample.value)) * 1.08;
    const x = (hour) => plot.left + hour / 24 * (plot.right - plot.left);
    const y = (value) => plot.bottom - Math.max(0, value) / maximum * (plot.bottom - plot.top);
    const line = fit.samples.map((sample, index) => `${index ? "L" : "M"}${x(sample.hour).toFixed(1)},${y(sample.value).toFixed(1)}`).join(" ");

    byData("fourier-line").setAttribute("d", line);
    byData("fourier-area").setAttribute("d", `${line} L${plot.right},${plot.bottom} L${plot.left},${plot.bottom} Z`);
    byData("fourier-points").replaceChildren(...hourly.map((value, hour) => {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", x(hour).toFixed(1));
      circle.setAttribute("cy", y(value).toFixed(1));
      circle.setAttribute("r", "5");
      circle.setAttribute("aria-label", `${formatHour(hour)} observed ${value.toLocaleString()} steps`);
      return circle;
    }));

    const peak = fit.samples.reduce((best, sample) => sample.value > best.value ? sample : best, fit.samples[0]);
    byData("fourier-score").textContent = `3 harmonics · R² ${Math.max(0, fit.rSquared).toFixed(2)}`;
    byData("fourier-peak").textContent = `Around ${formatHour(Math.round(peak.hour) % 24)}`;
    chart.setAttribute("aria-label", `Three-harmonic Fourier series fit with R squared ${Math.max(0, fit.rSquared).toFixed(2)}, peaking around ${formatHour(Math.round(peak.hour) % 24)}`);
    return { harmonics: fit.harmonics, coefficients: [...fit.coefficients], rSquared: fit.rSquared, peakHour: peak.hour };
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
    const activityWindow = activeWindow(activity.hourly);
    const opportunity = periodOpportunity(activity.hourly);

    byData("weekly-steps").textContent = weeklyTotal.toLocaleString();
    const changeElement = byData("weekly-change");
    changeElement.textContent = `${weeklyChange >= 0 ? "↑" : "↓"} ${Math.abs(weeklyChange).toFixed(1)}% vs prior synthetic week`;
    changeElement.style.color = weeklyChange >= 0 ? "var(--health-green)" : "var(--health-red)";
    byData("risk-value").textContent = `${risk.toFixed(1)}%`;
    const riskContext = risk < 2 ? "Lower simulated range" : risk < 5 ? "Middle simulated range" : "Higher simulated range";
    byData("risk-context").textContent = riskContext;
    byData("model-range").textContent = `${Math.max(0.2, risk * 0.78).toFixed(1)}%–${Math.min(25, risk * 1.34).toFixed(1)}%`;
    const gaugeSweep = Math.min(100, risk * 5);
    byData("gauge-value").style.strokeDasharray = `${gaugeSweep} 100`;
    byData("gauge-needle").style.transform = `rotate(${gaugeSweep * 1.8}deg)`;
    byData("risk-gauge").setAttribute("aria-label", `Synthetic 10-year model estimate ${risk.toFixed(1)} percent, ${riskContext.toLowerCase()}`);
    byData("active-window").textContent = `${formatHour(activityWindow.start)} – ${formatHour(activityWindow.end)}`;
    byData("hourly-copy").textContent = `Virtual seven-day average · ${profile.timezone}`;
    byData("strongest-pattern").textContent = `This profile moves most during ${formatHour(activityWindow.start)} – ${formatHour(activityWindow.end)}.`;
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
    const fourier = renderFourier(activity.hourly);
    renderSparkline(activity.daily);
    renderSurvival(risk);

    lastRendered = { profile: { ...profile }, weekVariant, weeklyTotal, priorTotal, weeklyChange, risk, activeWindow: { ...activityWindow }, opportunity, fourier, daily: [...activity.daily], hourly: [...activity.hourly] };
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

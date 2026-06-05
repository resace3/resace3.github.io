function createEntityRow(table) {
  const fieldName = table.dataset.fieldName;
  const fragment = document.createDocumentFragment();
  const row = document.createElement("tr");
  const distributionRow = document.createElement("tr");
  row.innerHTML = `
    <td><input name="${fieldName}" placeholder="Start typing a sensor name" list="covariate-sensor-options" autocomplete="off" data-entity-search-input></td>
    <td>
      <select name="predictor_aggregations">
        <option value="last">Last</option>
        <option value="sum">Sum</option>
        <option value="mean">Mean</option>
        <option value="min">Min</option>
        <option value="max">Max</option>
      </select>
    </td>
    <td>
      <select name="predictor_day_offsets">
        <option value="0">Same day</option>
        <option value="1">Day before</option>
      </select>
    </td>
    <td><button class="button small-button" type="button" data-covariate-toggle>Show</button></td>
    <td><button class="button small-button" type="button" data-remove-row>Remove</button></td>
  `;
  distributionRow.className = "covariate-distribution-row";
  distributionRow.hidden = true;
  distributionRow.innerHTML = '<td colspan="5"><div class="mini-histogram" data-covariate-distribution></div></td>';
  fragment.appendChild(row);
  fragment.appendChild(distributionRow);
  return fragment;
}

function dagNodeTemplate(index) {
  const offset = 70 + index * 26;
  const nodeId = `variable_${index}`;
  return `
    <article class="dag-node covariate" style="left: ${offset}px; top: ${offset}px" data-dag-node data-node-id="${nodeId}">
      <input type="hidden" name="node_x" value="${offset}" data-node-x>
      <input type="hidden" name="node_y" value="${offset}" data-node-y>
      <strong class="dag-node-title" data-node-title>Variable ${index}</strong>
      <span class="dag-time-shadow" data-node-time-shadow>t-1</span>
      <span class="dag-role-shadow" data-node-role-shadow>(covariate)</span>
      <input type="hidden" name="node_label" value="Variable ${index}" data-node-label-input>
      <input type="hidden" name="node_description" data-node-description>
      <input type="hidden" name="node_id" value="${nodeId}" data-node-id-input>
      <input type="hidden" name="node_role" value="covariate" data-node-role>
      <input type="hidden" name="node_missing_strategy" value="drop" data-node-missing-strategy>
      <input type="hidden" name="node_min_valid" data-node-min-valid>
      <input type="hidden" name="node_max_valid" data-node-max-valid>
      <input type="hidden" name="node_time_start" data-node-time-start>
      <input type="hidden" name="node_time_end" data-node-time-end>
      <input type="hidden" name="node_rolling_days" value="1" data-node-rolling-days>
      <input type="hidden" name="node_aggregation" value="last" data-node-aggregation>
      <input type="hidden" name="node_day_lag" value="1" data-node-day-lag>
      <input type="hidden" name="node_value_type" value="continuous" data-node-value-type>
      <input type="hidden" name="node_threshold_operator" value="more_than" data-node-threshold-operator>
      <input type="hidden" name="node_threshold" data-node-threshold>
      <div class="dag-node-histogram" data-dag-node-histogram hidden></div>
      <button
        class="dag-connect-handle top"
        type="button"
        title="Drag to another box to create an arrow"
        aria-label="Create arrow from this box"
        data-dag-connect-handle
        data-port="top"></button>
      <button
        class="dag-connect-handle right"
        type="button"
        title="Drag to another box to create an arrow"
        aria-label="Create arrow from this box"
        data-dag-connect-handle
        data-port="right"></button>
      <button
        class="dag-connect-handle bottom"
        type="button"
        title="Drag to another box to create an arrow"
        aria-label="Create arrow from this box"
        data-dag-connect-handle
        data-port="bottom"></button>
      <button
        class="dag-connect-handle left"
        type="button"
        title="Drag to another box to create an arrow"
        aria-label="Create arrow from this box"
        data-dag-connect-handle
        data-port="left"></button>
    </article>
  `;
}

function dagZoom(canvas) {
  return Number(canvas?.dataset.zoom) || 1;
}

function setDagZoom(canvas, zoom) {
  if (!canvas) return;
  const clamped = Math.min(1, Math.max(0.35, zoom));
  canvas.dataset.zoom = String(clamped);
  canvas.style.transform = `scale(${clamped})`;
  canvas.style.marginRight = `${canvas.offsetWidth * (clamped - 1)}px`;
  canvas.style.marginBottom = `${canvas.offsetHeight * (clamped - 1)}px`;
  updateDagEdges();
}

function dagEdgeTemplate() {
  return `
    <div class="dag-edge-row" data-dag-edge-row>
      <input name="edge_source" data-edge-source>
      <input name="edge_target" data-edge-target>
      <input name="edge_label" data-edge-label>
    </div>
  `;
}

let autosaveTimer;
let dagAutosaveTimer;
let entitySearchTimer;
let mathJaxLoadPromise;
const DAG_NODE_MARGIN = 64;
const MATHJAX_SRC = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js";
const DAG_NODE_FIELD_SELECTORS = {
  node_label: "[data-node-label-input]",
  node_description: "[data-node-description]",
  node_id: "[data-node-id-input]",
  node_role: "[data-node-role]",
  node_aggregation: "[data-node-aggregation]",
  node_time_start: "[data-node-time-start]",
  node_time_end: "[data-node-time-end]",
  node_day_lag: "[data-node-day-lag]",
  node_rolling_days: "[data-node-rolling-days]",
  node_value_type: "[data-node-value-type]",
  node_threshold_operator: "[data-node-threshold-operator]",
  node_threshold: "[data-node-threshold]",
  node_missing_strategy: "[data-node-missing-strategy]",
  node_min_valid: "[data-node-min-valid]",
  node_max_valid: "[data-node-max-valid]",
};
const ANALYSIS_PROGRESS_STEPS = [
  "Fetching Home Assistant history.",
  "Building daily exposure, outcome, and covariate variables.",
  "Estimating causal effects from the selected method.",
  "Checking positivity and observed exposure histories.",
  "Building result plots and diagnostic artifacts.",
  "Building method diagnostics and comparison views.",
  "Validating method outputs and effect summaries.",
  "Formatting the analysis report.",
  "Rendering the final analysis results.",
];
let analysisProgressTimer;

function analysisFormData() {
  const form = document.querySelector("[data-analysis-form]");
  if (!form?.dataset.autosaveUrl) return null;
  return { form, url: form.dataset.autosaveUrl, data: new FormData(form) };
}

function autosaveSettingsNow() {
  const payload = analysisFormData();
  if (!payload) return;
  fetch(payload.url, {
    method: "POST",
    body: payload.data,
    keepalive: true,
  }).catch(() => {});
}

function scheduleAutosaveSettings() {
  window.clearTimeout(autosaveTimer);
  autosaveTimer = window.setTimeout(autosaveSettingsNow, 650);
}

function saveSettingsBeforeUnload() {
  const payload = analysisFormData();
  if (!payload || !navigator.sendBeacon) return;
  const encoded = new URLSearchParams(payload.data);
  const body = new Blob([encoded.toString()], {
    type: "application/x-www-form-urlencoded;charset=UTF-8",
  });
  navigator.sendBeacon(payload.url, body);
}

function dagFormPayload() {
  const form = document.querySelector("[data-dag-form]");
  if (!form?.dataset.dagAutosaveUrl) return null;
  return { form, url: form.dataset.dagAutosaveUrl, data: new FormData(form) };
}

function autosaveDagNow(useBeacon = false) {
  const payload = dagFormPayload();
  if (!payload) return Promise.resolve(true);
  window.clearTimeout(dagAutosaveTimer);
  if (findDagCycle().length) {
    updateDagCycleWarning();
    return Promise.resolve(false);
  }
  if (useBeacon && navigator.sendBeacon) {
    const encoded = new URLSearchParams(payload.data);
    const body = new Blob([encoded.toString()], {
      type: "application/x-www-form-urlencoded;charset=UTF-8",
    });
    navigator.sendBeacon(payload.url, body);
    return Promise.resolve(true);
  }
  return fetch(payload.url, {
    method: "POST",
    body: payload.data,
    keepalive: true,
  })
    .then((response) => response.ok)
    .catch(() => false);
}

function scheduleDagAutosave() {
  window.clearTimeout(dagAutosaveTimer);
  dagAutosaveTimer = window.setTimeout(() => autosaveDagNow(false), 450);
}

function saveDagBeforeUnload() {
  autosaveDagNow(true);
}

function allEntityDatalists() {
  return Array.from(document.querySelectorAll("#all-sensor-options, #dag-sensor-options, #covariate-sensor-options"));
}

function knownSensorIds() {
  const values = new Set();
  allEntityDatalists().forEach((list) => {
    list.querySelectorAll("option").forEach((option) => {
      if (option.value) values.add(option.value.trim());
    });
  });
  return values;
}

function validateSidebarSensor() {
  const input = sidebarField("node_id");
  const error = document.querySelector("[data-sidebar-sensor-error]");
  if (!input || !error) return true;
  const value = input.value.trim();
  const isSensorLike = value.includes(".");
  const isKnown = knownSensorIds().has(value);
  const invalid = Boolean(value && isSensorLike && !isKnown);
  error.hidden = !invalid;
  input.classList.toggle("has-error", invalid);
  input.setAttribute("aria-invalid", String(invalid));
  return !invalid;
}

async function refreshEntitySuggestions(query) {
  const urlBase = window.entitySearchUrl || document.querySelector("[data-entity-search-url]")?.dataset.entitySearchUrl;
  if (!urlBase) return;
  const url = new URL(urlBase, window.location.origin);
  url.searchParams.set("q", query || "");
  try {
    const response = await fetch(url);
    const data = await response.json();
    if (!data.ok) return;
    const options = data.entities.map((entity) => {
      const label = entity.label ? String(entity.label).replaceAll('"', "&quot;") : "";
      return `<option value="${entity.entity_id}">${label}</option>`;
    }).join("");
    allEntityDatalists().forEach((list) => {
      if (list) list.innerHTML = options;
    });
    validateSidebarSensor();
  } catch {
    // Keep existing local sensor map options if Home Assistant lookup is unavailable.
  }
}

function scheduleEntitySuggestionRefresh(query) {
  window.clearTimeout(entitySearchTimer);
  entitySearchTimer = window.setTimeout(() => refreshEntitySuggestions(query), 250);
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  return Number.isInteger(number) ? String(number) : number.toFixed(1);
}

function formatHours(minutes) {
  const number = Number(minutes);
  if (!Number.isFinite(number)) return "";
  return `${(number / 60).toFixed(2)} hr`;
}

function formatMinutesWithHours(minutes) {
  return `${formatNumber(minutes)} min (${formatHours(minutes)})`;
}

function gaussianDensity(values, minimum, maximum, sampleCount = 72) {
  const numericValues = values.map(Number).filter(Number.isFinite);
  if (!numericValues.length || !Number.isFinite(minimum) || !Number.isFinite(maximum)) {
    return [];
  }
  const span = Math.max(maximum - minimum, 1);
  const mean = numericValues.reduce((sum, value) => sum + value, 0) / numericValues.length;
  const variance = numericValues.reduce((sum, value) => sum + (value - mean) ** 2, 0) / numericValues.length;
  const std = Math.sqrt(variance) || span / 6;
  const bandwidth = Math.max(span / 20, 1.06 * std * numericValues.length ** -0.2);
  const points = [];
  for (let index = 0; index < sampleCount; index += 1) {
    const ratio = sampleCount === 1 ? 0 : index / (sampleCount - 1);
    const xValue = minimum + ratio * span;
    const density = numericValues.reduce((sum, value) => {
      const z = (xValue - value) / bandwidth;
      return sum + Math.exp(-0.5 * z * z);
    }, 0) / (numericValues.length * bandwidth);
    points.push({ xValue, x: ratio * 100, density });
  }
  const maxDensity = Math.max(...points.map((point) => point.density), Number.EPSILON);
  return points.map((point) => ({
    ...point,
    y: 100 - (point.density / maxDensity) * 88,
  }));
}

function fittedLinePath(points) {
  if (!points.length) return "";
  if (points.length === 1) return `M ${points[0].x.toFixed(3)} ${points[0].y.toFixed(3)}`;
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(3)} ${point.y.toFixed(3)}`)
    .join(" ");
}

function densitySvg(values, minimum, maximum, className, label) {
  const points = gaussianDensity(values, minimum, maximum);
  if (points.length < 2) return "";
  const linePath = fittedLinePath(points);
  const areaPath = `${linePath} L 100 100 L 0 100 Z`;
  return `
    <svg class="${className}" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="${label}">
      <path class="density-area" d="${areaPath}"></path>
      <path class="density-line" d="${linePath}"></path>
    </svg>
  `;
}

function updateThresholdHours() {
  const thresholdInput = document.querySelector("[data-threshold-input]");
  const thresholdHours = document.querySelector("[data-threshold-hours]");
  if (!thresholdInput || !thresholdHours) return;
  thresholdHours.textContent = thresholdInput.value
    ? `~ ${formatHours(thresholdInput.value)}`
    : "";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function humanizeVariableName(value) {
  const text = String(value || "").trim();
  if (!text) return "Variable";
  const raw = text.includes(".") ? text.split(".").pop() : text;
  return raw
    .replace(/^nick_r_/, "")
    .replace(/^sensor_/, "")
    .replace(/^binary_sensor_/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function equationContext() {
  const nodes = Array.from(document.querySelectorAll("[data-dag-node]"));
  const byRole = (role) => nodes.filter((node) => nodeFieldValue(node, "node_role") === role);
  const nodeLabel = (node) => humanReadableNodeLabel(node);
  const exposureNode = byRole("exposure")[0];
  const outcomeNode = byRole("outcome")[0];
  const covariateNodes = byRole("covariate");

  return {
    exposure: exposureNode ? nodeLabel(exposureNode) : "No exposure selected",
    outcome: outcomeNode ? nodeLabel(outcomeNode) : "No outcome selected",
    covariates: covariateNodes.map(nodeLabel),
  };
}

function listSentence(items, fallback = "no covariates selected") {
  const clean = items.map((item) => String(item).trim()).filter(Boolean);
  if (!clean.length) return fallback;
  if (clean.length === 1) return clean[0];
  return `${clean.slice(0, -1).join(", ")} and ${clean[clean.length - 1]}`;
}

function methodEquation(method) {
  const equations = {
    dag_auto: {
      title: "DAG-selected estimator",
      equation: "Estimator = selected from DAG roles, arrows, and variable timing",
      latex: String.raw`\widehat{\psi} = \operatorname{Estimator}\!\left(\mathcal{G}, A_t, Y_t, H_t\right)`,
      detail: "Use the graph to identify exposure, outcome, and adjustment variables before estimation.",
    },
    g_formula: {
      title: "G-formula / outcome model",
      equation: "E[Y_t(a)] = E{ m(a, H_t) }",
      latex: String.raw`\mathbb{E}\!\left[Y_t(a)\right] = \mathbb{E}_{H_t}\!\left[m(a, H_t)\right], \quad m(a,H_t)=\mathbb{E}\!\left(Y_t \mid A_t=a,H_t\right)`,
      detail: "Model the outcome from exposure and relevant history, then contrast predicted outcomes under exposure versus no exposure.",
    },
    ipw: {
      title: "Inverse probability weighting",
      equation: "E[Y_t(a)] = E[ I(A_t = a)Y_t / P(A_t = a | H_t) ]",
      latex: String.raw`\mathbb{E}\!\left[Y_t(a)\right] = \mathbb{E}\!\left[\frac{\mathbb{1}(A_t=a)Y_t}{P(A_t=a \mid H_t)}\right]`,
      detail: "Model the exposure probability from history and weight observations to balance measured time-varying confounders.",
    },
    doubly_robust: {
      title: "Doubly robust / AIPW",
      equation: "DR(a) = m(a,H_t) + I(A_t=a){Y_t - m(a,H_t)} / P(A_t=a|H_t)",
      latex: String.raw`\operatorname{DR}(a) = m(a,H_t) + \frac{\mathbb{1}(A_t=a)\left\{Y_t - m(a,H_t)\right\}}{P(A_t=a \mid H_t)}`,
      detail: "Combine the outcome model and exposure model so the estimate can remain consistent if one model is correctly specified.",
    },
  };
  return equations[method] || equations.g_formula;
}

function methodEquationExplanation(method, context) {
  const commonSymbols = [
    ["E[Y]", "Expected value, or average, of the outcome."],
    ["Y_t", "The outcome measured at time t."],
    ["A_t", "The exposure or treatment variable at time t."],
    ["a", "A specific exposure level, such as exposed versus not exposed."],
    ["H_t", "Relevant history before the outcome, including covariates and lagged variables from the DAG."],
  ];
  const commonInputs = [
    ["Outcome from DAG", "The node marked as the outcome."],
    ["Exposure from DAG", "The node marked as the exposure."],
    ["History and covariates from DAG", "Nodes marked as covariates, plus relevant lagged history."],
  ];
  const commonAssumptions = [
    "Consistency: the observed outcome matches the outcome under the exposure history that actually happened.",
    "Positivity: the data include enough exposed and unexposed days for similar histories.",
    "No unmeasured time-varying confounding: the DAG includes the important causes of both exposure and outcome.",
  ];
  const interpretation = "The final treatment effect is in outcome units. A positive estimate means the outcome is higher under the exposure condition; a negative estimate means it is lower. The confidence interval shows the uncertainty around that estimate.";
  const methodDetails = {
    dag_auto: {
      estimates: "An effect of the exposure on the outcome, using the roles and arrows in the DAG to choose an estimator workflow.",
      intuition: "The app reads your graph first, identifies the exposure, outcome, and adjustment history, then chooses the causal workflow that best matches those variables.",
      symbols: [
        ...commonSymbols,
        ["G", "Your causal DAG: the boxes and arrows you drew."],
        ["psi", "The target causal effect of the exposure on the outcome."],
      ],
      inputs: commonInputs,
      assumptions: commonAssumptions,
      technical: "The DAG-selected mode resolves to the estimator supported by the exposure, outcome, and adjustment variables currently present in the graph.",
    },
    g_formula: {
      estimates: "The average outcome that would be predicted if the exposure were set to a chosen level for comparable histories.",
      intuition: "First model the outcome using exposure and history. Then reuse that model to predict what would happen under each exposure scenario, and compare the predicted outcomes.",
      symbols: [
        ...commonSymbols,
        ["m(a, H_t)", "The model-predicted outcome if the exposure were set to level a for a day with history H_t."],
      ],
      inputs: commonInputs,
      assumptions: [
        ...commonAssumptions,
        "Outcome model correctness: the model for the expected outcome is a reasonable description of the data.",
      ],
      technical: "The G-formula standardizes predicted outcomes over the observed history distribution: average the modeled outcome m(a,H_t) across the histories in your dataset.",
    },
    ipw: {
      estimates: "The average outcome under a chosen exposure level, after weighting observations so the measured histories look more balanced.",
      intuition: "Days that were unlikely to have their observed exposure get more weight. This creates a weighted pseudo-population where exposure is less tied to the measured history.",
      symbols: [
        ...commonSymbols,
        ["I(A_t = a)", "An indicator that equals 1 when the exposure actually equals level a, and 0 otherwise."],
        ["P(A_t = a | H_t)", `The estimated probability of seeing that exposure level given the DAG history variables.`],
      ],
      inputs: commonInputs,
      assumptions: [
        ...commonAssumptions,
        "Exposure model correctness: the model for exposure probability is a reasonable description of the data.",
        "No extreme weights: exposure probabilities should not be too close to 0 or 1 for many days.",
      ],
      technical: "IPW estimates the counterfactual mean by reweighting observed outcomes by the inverse of the estimated exposure probability conditional on history.",
    },
    doubly_robust: {
      estimates: "The average outcome under a chosen exposure level, combining an outcome model with an exposure-probability model.",
      intuition: "This method starts with the outcome-model prediction, then corrects it using weighted residuals from days where the exposure level was actually observed.",
      symbols: [
        ...commonSymbols,
        ["m(a, H_t)", "The model-predicted outcome under exposure level a for history H_t."],
        ["I(A_t = a)", "An indicator that equals 1 when the exposure equals level a."],
        ["P(A_t = a | H_t)", "The probability of the exposure taking level a for the DAG history H_t."],
        ["Y_t - m(a, H_t)", "The outcome-model error for days with the relevant exposure level."],
      ],
      inputs: commonInputs,
      assumptions: [
        ...commonAssumptions,
        "At least one model should be reasonable: either the outcome model or the exposure model needs to be correctly specified for the doubly robust protection to help.",
        "Stable weights: exposure probabilities should not be too close to 0 or 1 for many days.",
      ],
      technical: "AIPW combines standardization and inverse probability weighting. The augmentation term uses weighted residuals to correct the outcome-model prediction.",
    },
  };
  const details = methodDetails[method] || methodDetails.g_formula;
  return {
    ...details,
    interpretation,
  };
}

function renderDefinitionList(rows) {
  return rows
    .map(([term, description]) => `
      <div>
        <dt>${escapeHtml(term)}</dt>
        <dd>${escapeHtml(description)}</dd>
      </div>
    `)
    .join("");
}

function renderBulletList(items) {
  return items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function renderEquationExplanation(method, context) {
  const explanation = methodEquationExplanation(method, context);
  return `
    <div class="equation-explanation-grid">
      <section>
        <h4>What this estimates</h4>
        <p>${escapeHtml(explanation.estimates)}</p>
      </section>
      <section>
        <h4>Method intuition</h4>
        <p>${escapeHtml(explanation.intuition)}</p>
      </section>
      <section>
        <h4>Symbols in plain English</h4>
        <dl class="equation-symbol-list">
          ${renderDefinitionList(explanation.symbols)}
        </dl>
      </section>
      <section>
        <h4>Inputs from your DAG</h4>
        <dl class="equation-symbol-list">
          ${renderDefinitionList(explanation.inputs)}
        </dl>
      </section>
      <section>
        <h4>Assumptions</h4>
        <ul>${renderBulletList(explanation.assumptions)}</ul>
      </section>
      <section>
        <h4>How to interpret the effect</h4>
        <p>${escapeHtml(explanation.interpretation)}</p>
      </section>
    </div>
    <details class="equation-technical-details">
      <summary>Technical details</summary>
      <p>${escapeHtml(explanation.technical)}</p>
    </details>
  `;
}

function updateMethodEquation() {
  const select = document.querySelector("[data-analysis-method]");
  const target = document.querySelector("[data-method-equation]");
  updateRunAnalysisButtonState();
  if (!select || !target) return;
  const method = select.value;
  const info = methodEquation(method);
  const context = equationContext();
  target.innerHTML = `
    <strong>${escapeHtml(info.title)}</strong>
    <span class="method-equation-summary">${escapeHtml(info.detail)}</span>
    <div class="method-equation-box" data-equation-box>
      <div class="method-equation-scroll" data-equation-scroll>
        <div class="method-equation-scale" data-equation-scale aria-label="${escapeHtml(info.equation)}">\\[${info.latex}\\]</div>
      </div>
    </div>
    <button class="equation-explain-toggle" type="button" aria-expanded="false" data-equation-explain-toggle>
      Explain equation
    </button>
    <div class="equation-explanation" data-equation-explanation hidden>
      ${renderEquationExplanation(method, context)}
    </div>
  `;
  typesetMath(target);
}

function hasDagNodeRole(role) {
  return Array.from(document.querySelectorAll("[data-dag-node]")).some((node) => {
    const nodeRole = nodeFieldValue(node, "node_role");
    const nodeId = dagNodeId(node);
    return nodeRole === role && nodeId.includes(".");
  });
}

function updateRunAnalysisButtonState() {
  const button = document.querySelector("[data-run-analysis-button]");
  if (!button || button.dataset.analysisRunning === "1") return;
  const hasExposure = hasDagNodeRole("exposure");
  const hasOutcome = hasDagNodeRole("outcome");
  const hasCycle = Boolean(findDagCycle().length);
  const ready = hasExposure && hasOutcome && !hasCycle;
  button.disabled = !ready;
  if (!hasExposure) {
    button.title = "Mark one DAG variable as the exposure before running analysis.";
  } else if (!hasOutcome) {
    button.title = "Mark one DAG variable as the outcome before running analysis.";
  } else if (hasCycle) {
    button.title = "Remove the directed cycle from the DAG before running analysis.";
  } else {
    button.removeAttribute("title");
  }
}

function syncAnalysisDayInputs(source) {
  if (!source?.matches?.("[data-analysis-days], [data-analysis-days-secondary]")) return;
  const selector = source.matches("[data-analysis-days]") ? "[data-analysis-days-secondary]" : "[data-analysis-days]";
  document.querySelectorAll(selector).forEach((input) => {
    input.value = source.value;
  });
}

function dagTimeLabelForRole(role) {
  if (role === "exposure" || role === "covariate") return "t-1";
  if (role === "outcome") return "t";
  return "";
}

function updateDagNodeTimeMarker(node, { applyDefaultLag = false } = {}) {
  if (!node) return;
  const role = node.querySelector("[data-node-role]")?.value || "";
  const marker = node.querySelector("[data-node-time-shadow]");
  const roleMarker = node.querySelector("[data-node-role-shadow]");
  if (marker) marker.textContent = dagTimeLabelForRole(role);
  if (roleMarker) roleMarker.textContent = role ? `(${role})` : "";
  const lagInput = node.querySelector("[data-node-day-lag]");
  if (applyDefaultLag && lagInput && (role === "exposure" || role === "covariate") && Number(lagInput.value || 0) === 0) {
    lagInput.value = "1";
  }
}

function updateDagTimeMarkers({ applyDefaultLags = false, autosave = false } = {}) {
  let changed = false;
  document.querySelectorAll("[data-dag-node]").forEach((node) => {
    const lagInput = node.querySelector("[data-node-day-lag]");
    const before = lagInput?.value;
    updateDagNodeTimeMarker(node, { applyDefaultLag: applyDefaultLags });
    changed = changed || Boolean(lagInput && before !== lagInput.value);
  });
  if (changed && autosave) scheduleDagAutosave();
}

function setDagNodeThresholdFromHistogram(bin) {
  const node = bin.closest("[data-dag-node]") || selectedDagNode();
  const target = bin.closest("[data-dag-node-histogram]");
  if (!node || !target) return;
  const thresholdInput = node.querySelector("[data-node-threshold]");
  const valueType = node.querySelector("[data-node-value-type]");
  const readout = target.querySelector("[data-dag-histogram-readout]");
  const midpoint = Number(bin.dataset.binMidpoint);
  if (!thresholdInput || !Number.isFinite(midpoint)) return;
  thresholdInput.value = formatNumber(midpoint);
  if (valueType) valueType.value = "binary_threshold";
  updateDagEncodingControls(node);
  if (readout) readout.textContent = `Binary threshold set to ${formatNumber(midpoint)} from ${bin.dataset.binLabel}.`;
  scheduleDagAutosave();
  const histogramData = target.histogramData;
  if (histogramData) {
    const nodeTarget = node.querySelector("[data-dag-node-histogram]");
    renderDagNodeHistogram(nodeTarget || target, histogramData);
  }
  syncDagPropertiesSidebar(node);
}

function updateDagEncodingControls(node) {
  if (!node) return;
  const isBinary = node.querySelector("[data-node-value-type]")?.value === "binary_threshold";
  node.querySelectorAll("[data-binary-threshold-control]").forEach((control) => {
    control.hidden = !isBinary;
  });
  if (node.classList.contains("is-selected")) updateSidebarBinaryControls();
}

function setSidebarStep(target) {
  const panel = document.querySelector("[data-dag-properties-panel]");
  if (!panel || !target) return;
  panel.querySelectorAll("[data-sidebar-tab]").forEach((button) => {
    const active = button.dataset.sidebarTab === target;
    button.classList.toggle("active", active);
    button.setAttribute("aria-expanded", String(active));
  });
  panel.querySelectorAll("[data-sidebar-panel]").forEach((section) => {
    section.hidden = section.dataset.sidebarPanel !== target;
  });
}

function updateAllDagEncodingControls() {
  document.querySelectorAll("[data-dag-node]").forEach(updateDagEncodingControls);
}

function selectedDagNode() {
  return document.querySelector("[data-dag-node].is-selected");
}

function dagNodeField(node, fieldName) {
  return node?.querySelector(DAG_NODE_FIELD_SELECTORS[fieldName] || "");
}

function sidebarField(fieldName) {
  return document.querySelector(`[data-sidebar-field="${fieldName}"]`);
}

function nodeFieldValue(node, fieldName) {
  return dagNodeField(node, fieldName)?.value || "";
}

function setNodeFieldValue(node, fieldName, value) {
  const field = dagNodeField(node, fieldName);
  if (field) field.value = value;
}

function roleLabel(role) {
  if (role === "exposure") return "Exposure";
  if (role === "outcome") return "Outcome";
  if (role === "latent") return "Latent";
  return "Covariate";
}

function humanReadableNodeLabel(node) {
  const label = nodeFieldValue(node, "node_label").trim();
  const sensor = dagNodeId(node);
  if (label && label !== sensor) return label;
  const raw = (label || sensor || "Variable").split(".").pop();
  return raw
    .replace(/^nick_r_/, "")
    .replace(/^sensor_/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function updateDagNodeSummary(node) {
  if (!node) return;
  const title = node.querySelector("[data-node-title]");
  const label = humanReadableNodeLabel(node);
  if (title) title.textContent = label;
}

function updateSidebarBinaryControls() {
  const isBinary = sidebarField("node_value_type")?.value === "binary_threshold";
  document.querySelectorAll("[data-sidebar-binary-control]").forEach((control) => {
    control.hidden = !isBinary;
  });
}

function selectedDagNodes() {
  return Array.from(document.querySelectorAll("[data-dag-node].is-selected"));
}

function edgeRowKey(row) {
  const source = row?.querySelector("[data-edge-source]")?.value.trim() || "";
  const target = row?.querySelector("[data-edge-target]")?.value.trim() || "";
  return `${source}→${target}`;
}

function selectedDagEdgeRows() {
  return Array.from(document.querySelectorAll("[data-dag-edge-row].is-selected"));
}

function clearSelectedDagEdges() {
  document.querySelectorAll("[data-dag-edge-row].is-selected").forEach((row) => {
    row.classList.remove("is-selected");
  });
  document.querySelectorAll("[data-rendered-edge].is-selected").forEach((edge) => {
    edge.classList.remove("is-selected");
    edge.setAttribute("marker-end", "url(#dag-arrow)");
  });
}

function clearDagSelection() {
  document.querySelectorAll("[data-dag-node].is-selected").forEach((node) => node.classList.remove("is-selected"));
  clearSelectedDagEdges();
  updateDagSelectionUi();
}

function updateDagSelectionUi() {
  const nodes = selectedDagNodes();
  const edges = selectedDagEdgeRows();
  const button = document.querySelector("[data-dag-delete-selected]");
  const count = document.querySelector("[data-dag-selection-count]");
  const itemCount = nodes.length + edges.length;
  if (button) button.disabled = itemCount === 0;
  if (count) {
    const pieces = [];
    if (nodes.length) pieces.push(`${nodes.length} node${nodes.length === 1 ? "" : "s"} selected`);
    if (edges.length) pieces.push(`${edges.length} edge${edges.length === 1 ? "" : "s"} selected`);
    count.textContent = pieces.join(" · ") || "No selection";
  }
  syncDagPropertiesSidebar(nodes.length === 1 && edges.length === 0 ? nodes[0] : null);
}

function setSelectedDagNode(node, { additive = false, toggle = false } = {}) {
  if (!additive) {
    document.querySelectorAll("[data-dag-node].is-selected").forEach((selected) => {
      if (selected !== node) selected.classList.remove("is-selected");
    });
    clearSelectedDagEdges();
  }
  if (node) {
    if (toggle && node.classList.contains("is-selected")) {
      node.classList.remove("is-selected");
    } else {
      node.classList.add("is-selected");
    }
  }
  updateDagSelectionUi();
}

function setSelectedDagEdge(row, { additive = false, toggle = false } = {}) {
  if (!additive) {
    document.querySelectorAll("[data-dag-node].is-selected").forEach((node) => node.classList.remove("is-selected"));
    clearSelectedDagEdges();
  }
  if (row) {
    if (toggle && row.classList.contains("is-selected")) {
      row.classList.remove("is-selected");
    } else {
      row.classList.add("is-selected");
    }
  }
  updateDagSelectionUi();
  updateDagEdges();
}

function confirmDagDelete(nodes, edges) {
  const total = nodes.length + edges.length;
  if (total <= 1) return true;
  return window.confirm(`Delete ${total} selected item${total === 1 ? "" : "s"}? This cannot be undone.`);
}

function deleteSelectedDagItems() {
  const nodes = selectedDagNodes();
  const edges = selectedDagEdgeRows();
  if (!nodes.length && !edges.length) return false;
  if (!confirmDagDelete(nodes, edges)) return false;
  const selectedEdgeSet = new Set(edges);
  const nodeIds = nodes.map(dagNodeId).filter(Boolean);
  removeDagEdgesForNodeIds(nodeIds);
  selectedEdgeSet.forEach((row) => {
    if (row.isConnected) row.remove();
  });
  nodes.forEach((node) => node.remove());
  updateDagEdges();
  updateDagDeleteButton();
  updateDagSelectionUi();
  updateMethodEquation();
  scheduleDagAutosave();
  return true;
}

function syncDagPropertiesSidebar(node = selectedDagNode()) {
  const placeholder = document.querySelector("[data-dag-properties-placeholder]");
  const panel = document.querySelector("[data-dag-properties-panel]");
  const title = document.querySelector("[data-sidebar-title]");
  const histogram = document.querySelector("[data-sidebar-histogram]");
  if (!placeholder || !panel) return;

  if (!node) {
    placeholder.hidden = false;
    panel.hidden = true;
    if (histogram) {
      histogram.innerHTML = "";
      histogram.histogramData = null;
    }
    return;
  }

  placeholder.hidden = true;
  panel.hidden = false;
  Object.keys(DAG_NODE_FIELD_SELECTORS).forEach((fieldName) => {
    const control = sidebarField(fieldName);
    if (control) control.value = nodeFieldValue(node, fieldName);
  });
  if (title) title.textContent = nodeFieldValue(node, "node_label") || "Variable properties";
  updateSidebarBinaryControls();
  const nodeHistogram = node.querySelector("[data-dag-node-histogram]");
  if (histogram) {
    histogram.innerHTML = nodeHistogram?.innerHTML || "";
    histogram.histogramData = nodeHistogram?.histogramData || null;
  }
  validateSidebarSensor();
  if (nodeHistogram && !nodeHistogram.innerHTML.trim()) {
    loadDagNodeHistogram(node);
  }
}

function applySidebarFieldToNode(control) {
  const node = selectedDagNode();
  if (!node) return;
  const fieldName = control.dataset.sidebarField;
  if (!fieldName) return;
  const previousId = fieldName === "node_id" ? dagNodeId(node) : "";
  setNodeFieldValue(node, fieldName, control.value);

  if (fieldName === "node_id") {
    node.dataset.nodeId = control.value.trim();
    updateDagEdgesForRenamedNode(previousId, control.value.trim());
    validateSidebarSensor();
  }
  if (fieldName === "node_role") {
    node.classList.remove("exposure", "covariate", "outcome", "latent");
    node.classList.add(control.value);
    updateDagNodeTimeMarker(node, { applyDefaultLag: true });
    const lagControl = sidebarField("node_day_lag");
    if (lagControl) lagControl.value = nodeFieldValue(node, "node_day_lag");
  }
  if (fieldName === "node_value_type") {
    updateDagEncodingControls(node);
  }
  updateDagNodeSummary(node);
  syncDagPropertiesSidebar(node);
  refreshSelectedNodeHistogram(node, control);
  updateDagEdges();
  if (["node_label", "node_id", "node_role"].includes(fieldName)) {
    updateMethodEquation();
  }
  scheduleDagAutosave();
}

function updateDagEdgesForRenamedNode(previousId, nextId) {
  if (!previousId || !nextId || previousId === nextId) return;
  document.querySelectorAll("[data-dag-edge-row]").forEach((row) => {
    const source = row.querySelector("[data-edge-source]");
    const target = row.querySelector("[data-edge-target]");
    if (source?.value.trim() === previousId) source.value = nextId;
    if (target?.value.trim() === previousId) target.value = nextId;
  });
}

function dagEdgeRows() {
  return Array.from(document.querySelectorAll("[data-dag-edge-row]")).map((row) => ({
    row,
    source: row.querySelector("[data-edge-source]")?.value.trim() || "",
    target: row.querySelector("[data-edge-target]")?.value.trim() || "",
  })).filter((edge) => edge.source && edge.target && edge.source !== edge.target);
}

function findDagCycle() {
  const nodes = new Set(Array.from(document.querySelectorAll("[data-dag-node]")).map(dagNodeId).filter(Boolean));
  const edges = dagEdgeRows().filter((edge) => nodes.has(edge.source) && nodes.has(edge.target));
  const adjacency = new Map(Array.from(nodes).map((node) => [node, []]));
  edges.forEach((edge) => adjacency.get(edge.source)?.push(edge.target));
  const visiting = new Set();
  const visited = new Set();
  const stack = [];

  function visit(node) {
    if (visiting.has(node)) {
      const start = stack.indexOf(node);
      return stack.slice(start).concat(node);
    }
    if (visited.has(node)) return null;
    visiting.add(node);
    stack.push(node);
    for (const next of adjacency.get(node) || []) {
      const cycle = visit(next);
      if (cycle) return cycle;
    }
    stack.pop();
    visiting.delete(node);
    visited.add(node);
    return null;
  }

  for (const node of nodes) {
    const cycle = visit(node);
    if (cycle) return cycle;
  }
  return [];
}

function updateDagCycleWarning(cycle = findDagCycle()) {
  const warning = document.querySelector("[data-dag-cycle-warning]");
  const cyclicNodes = new Set(cycle);
  const cyclicEdgeKeys = new Set();
  for (let index = 0; index < cycle.length - 1; index += 1) {
    cyclicEdgeKeys.add(`${cycle[index]}|||${cycle[index + 1]}`);
  }
  document.querySelectorAll("[data-dag-node]").forEach((node) => {
    node.classList.toggle("has-cycle", cyclicNodes.has(dagNodeId(node)));
  });
  document.querySelectorAll("[data-dag-edge-row]").forEach((row) => {
    const source = row.querySelector("[data-edge-source]")?.value.trim();
    const target = row.querySelector("[data-edge-target]")?.value.trim();
    row.dataset.cyclic = cyclicEdgeKeys.has(`${source}|||${target}`) ? "1" : "0";
  });
  if (warning) {
    warning.hidden = !cycle.length;
    if (cycle.length) warning.textContent = `This graph has a directed cycle: ${cycle.join(" -> ")}. Causal DAGs must be acyclic. Remove one highlighted arrow before saving or running analysis.`;
  }
  updateRunAnalysisButtonState();
  return cycle;
}

function refreshSelectedNodeHistogram(node, control) {
  const target = node.querySelector("[data-dag-node-histogram]");
  if (!target?.innerHTML.trim()) return;
  if (control.matches('[data-sidebar-field="node_threshold"], [data-sidebar-field="node_value_type"], [data-sidebar-field="node_threshold_operator"]') && target.histogramData) {
    renderDagNodeHistogram(target, target.histogramData);
    syncDagPropertiesSidebar(node);
    return;
  }
  if (control.matches('[data-sidebar-field="node_aggregation"], [data-sidebar-field="node_time_start"], [data-sidebar-field="node_time_end"], [data-sidebar-field="node_day_lag"], [data-sidebar-field="node_id"]')) {
    loadDagNodeHistogram(node);
  }
}

function thresholdRatio(value, data) {
  if (!Number.isFinite(value) || data.maximum === data.minimum) return 0.5;
  return Math.min(1, Math.max(0, (value - data.minimum) / (data.maximum - data.minimum)));
}

function updateExposureCounts(data) {
  const thresholdInput = document.querySelector("[data-threshold-input]");
  const operator = document.querySelector("[data-outcome-operator]");
  const exposedCount = document.querySelector("[data-exposed-count]");
  const notExposedCount = document.querySelector("[data-not-exposed-count]");
  const exposedPercent = document.querySelector("[data-exposed-percent]");
  const notExposedPercent = document.querySelector("[data-not-exposed-percent]");
  if (!thresholdInput || !operator || !exposedCount || !notExposedCount || !data.values) return;

  const threshold = Number(thresholdInput.value);
  const exposed = data.values.filter((value) =>
    operator.value === "more_than" ? value > threshold : value < threshold
  ).length;
  const total = data.values.length || 1;
  const notExposed = data.values.length - exposed;
  exposedCount.textContent = exposed;
  notExposedCount.textContent = notExposed;
  if (exposedPercent) exposedPercent.textContent = `${Math.round((exposed / total) * 100)}%`;
  if (notExposedPercent) notExposedPercent.textContent = `${Math.round((notExposed / total) * 100)}%`;
}

function setCommittedThreshold(value, data, marker, readout) {
  const thresholdInput = document.querySelector("[data-threshold-input]");
  const rounded = Number(value.toFixed(1));
  thresholdInput.value = formatNumber(rounded);
  updateThresholdHours();
  updateExposureCounts(data);
  marker.style.left = `${thresholdRatio(rounded, data) * 100}%`;
  marker.classList.add("committed");
  readout.textContent = `Threshold set to ${formatMinutesWithHours(rounded)} · ${data.rows}/${data.requested_days} days available`;
  refreshOpenCovariateDistributions();
}

function renderHistogram(container, data) {
  container.histogramData = data;
  const bars = container.querySelector("[data-histogram-bars]");
  const stat = container.querySelector("[data-histogram-stat]");
  const readout = container.querySelector("[data-histogram-readout]");
  const thresholdInput = document.querySelector("[data-threshold-input]");

  if (!data.ok || !data.bins.length) {
    bars.innerHTML = `<p class="empty-state">${data.message || "No histogram data available."}</p>`;
    stat.textContent = "No data";
    readout.textContent = "Choose a numeric outcome sensor with history.";
    return;
  }

  const maxCount = Math.max(...data.bins.map((bin) => bin.count), 1);
  const yTicks = [maxCount, Math.round(maxCount / 2), 0];
  const xTicks = [
    data.minimum,
    data.minimum + (data.maximum - data.minimum) / 2,
    data.maximum,
  ];

  bars.innerHTML = `
    <div class="histogram-y-axis" aria-hidden="true">
      ${yTicks.map((tick) => `<span>${formatNumber(tick)}</span>`).join("")}
    </div>
    <div class="histogram-plot">
      <span class="axis-label y-label">Days</span>
      <div class="histogram-bars">
        ${densitySvg(data.values || [], data.minimum, data.maximum, "density-overlay outcome-density", "Fitted outcome distribution line")}
        ${data.bins
    .map((bin) => {
      const height = Math.max(6, Math.round((bin.count / maxCount) * 100));
      return `
        <div
          class="histogram-bar"
          style="height: ${height}%"
          data-lower="${bin.lower}"
          data-upper="${bin.upper}"
          title="${formatNumber(bin.lower)} to ${formatNumber(bin.upper)}: ${bin.count}">
          <span>${bin.count}</span>
        </div>
      `;
    })
    .join("")}
      </div>
      <div class="histogram-x-axis" aria-hidden="true">
        ${xTicks.map((tick) => `<span>${formatNumber(tick)}<small>${formatHours(tick)}</small></span>`).join("")}
      </div>
      <span class="axis-label x-label">Outcome value, minutes with hours</span>
    </div>
  `;

  const plotBars = bars.querySelector(".histogram-bars");

  const marker = document.createElement("div");
  marker.className = "histogram-marker";
  plotBars.appendChild(marker);

  stat.textContent = `${data.rows}/${data.requested_days} days`;
  readout.classList.toggle("coverage-warning", data.missing_days > 0);
  readout.textContent =
    data.missing_days > 0
      ? `${data.missing_days} requested days are missing for this sensor. Range ${formatMinutesWithHours(data.minimum)} to ${formatMinutesWithHours(data.maximum)}.`
      : `Full ${data.requested_days}-day coverage. Range ${formatMinutesWithHours(data.minimum)} to ${formatMinutesWithHours(data.maximum)}.`;
  updateThresholdHours();
  updateExposureCounts(data);
  const initialThreshold = Number(thresholdInput.value);
  if (Number.isFinite(initialThreshold)) {
    marker.style.left = `${thresholdRatio(initialThreshold, data) * 100}%`;
    marker.classList.add("committed");
  }

  plotBars.onmousemove = (event) => {
    const rect = plotBars.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    const value = data.minimum + ratio * (data.maximum - data.minimum);
    const rounded = Number(value.toFixed(1));
    readout.textContent = `Click to set threshold to ${formatMinutesWithHours(rounded)} · current threshold stays ${formatMinutesWithHours(thresholdInput.value)}`;
  };

  plotBars.onclick = (event) => {
    const rect = plotBars.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    const value = data.minimum + ratio * (data.maximum - data.minimum);
    setCommittedThreshold(value, data, marker, readout);
  };
}

async function loadHistogram() {
  const container = document.querySelector("[data-histogram]");
  if (!container) return;

  const sensorSelect = document.querySelector("[data-outcome-sensor]");
  const analysisDays = document.querySelector("[data-analysis-days]");
  const bars = container.querySelector("[data-histogram-bars]");
  const stat = container.querySelector("[data-histogram-stat]");

  const url = new URL(container.dataset.url, window.location.origin);
  url.searchParams.set("sensor", sensorSelect.value || container.dataset.initialSensor || "");
  url.searchParams.set("analysis_window_days", analysisDays.value || "90");

  stat.textContent = "Loading";
  bars.innerHTML = '<p class="empty-state">Loading outcome history.</p>';

  try {
    const response = await fetch(url);
    renderHistogram(container, await response.json());
  } catch {
    renderHistogram(container, {
      ok: false,
      message: "Unable to load outcome histogram.",
      bins: [],
    });
  }
}

function renderMiniHistogram(target, data) {
  if (!data.ok || !data.bins.length) {
    target.innerHTML = `<p class="empty-state">${data.message || "No covariate distribution available."}</p>`;
    return;
  }
  const maxCount = Math.max(...data.bins.map((bin) => bin.count), 1);
  const isBinary = data.kind === "binary";
  const correlation = data.association?.correlation;
  const associationText = correlation === null || correlation === undefined
    ? `Correlation unavailable · ${data.association?.available_days || 0} paired days`
    : `Single-variable correlation with exposed outcome: ${correlation} · ${data.association?.available_days || 0} paired days`;
  target.innerHTML = `
    <div class="mini-histogram-header">
      <strong>${data.sensor}</strong>
      <span>${data.rows}/${data.requested_days} days · ${data.aggregation} · ${data.day_offset ? "day before" : "same day"}</span>
    </div>
    ${data.outcome_label ? `<p class="mini-histogram-subtitle">Outcome rate by covariate level: ${data.outcome_label}</p>` : ""}
    <div class="mini-association">${associationText}</div>
    <div class="mini-histogram-bars grouped-covariate-bars ${isBinary ? "binary-level-bars" : ""}">
      ${
        isBinary
          ? ""
          : `${densitySvg(data.values || [], data.minimum, data.maximum, "density-overlay covariate-density", "Fitted covariate distribution line")}
             ${densitySvg(data.exposed_values || [], data.minimum, data.maximum, "density-overlay exposed-density", "Fitted exposed outcome distribution line")}`
      }
      ${data.bins.map((bin) => {
        const height = Math.max(6, Math.round((bin.count / maxCount) * 100));
        const exposedPercent = Number(bin.exposed_percent || 0);
        const label = bin.label || `${formatNumber(bin.lower)} to ${formatNumber(bin.upper)}`;
        const exposedHeight = Math.max(0, Math.round(((bin.exposed_count || 0) / maxCount) * 100));
        return `
          <div
            class="mini-histogram-bin"
            title="${label}: ${bin.count} days; ${formatNumber(exposedPercent)}% exposed outcome (${bin.exposed_count || 0}/${bin.outcome_days || 0})">
            <span class="mini-bar-pair">
              <span class="mini-count-bar" style="height:${height}%"><b>${bin.count}</b></span>
              <span class="mini-exposed-bar" style="height:${exposedHeight}%"><b>${bin.exposed_count || 0}</b></span>
            </span>
            <strong class="mini-exposed-count">${formatNumber(exposedPercent)}% exposed</strong>
            ${isBinary ? `<strong class="binary-level-label">${label}</strong>` : ""}
          </div>
        `;
      }).join("")}
    </div>
    <div class="mini-histogram-legend">
      <span><i class="count"></i> covariate days</span>
      <span><i class="exposed"></i> exposed outcome days</span>
      ${isBinary ? "" : "<span><i class=\"curve\"></i> fitted distribution lines</span>"}
    </div>
    <div class="mini-histogram-axis">
      ${
        isBinary
          ? "<span>Off / closed / 0</span><span>On / open / 1</span>"
          : `<span>${formatNumber(data.minimum)}</span><span>${formatNumber((data.minimum + data.maximum) / 2)}</span><span>${formatNumber(data.maximum)}</span>`
      }
    </div>
  `;
}

function renderDistributionStats(data) {
  const stats = data.statistics;
  if (!stats) return "";
  const statItems = [
    ["Mean", stats.mean],
    ["Median", stats.median],
    ["Std dev", stats.std_dev],
  ];
  const binarySummary = Number.isFinite(Number(stats.on_percent))
    ? `
      <div class="dag-stat-chip wide">
        <span>On / 1</span>
        <strong>${formatNumber(stats.on_percent)}%</strong>
        <small>${stats.on_count || 0} of ${data.rows || 0} days</small>
      </div>
    `
    : "";

  return `
    <div class="dag-distribution-stats" aria-label="Distribution statistics">
      ${binarySummary}
      ${statItems.map(([label, value]) => `
        <div class="dag-stat-chip">
          <span>${label}</span>
          <strong>${formatNumber(value)}</strong>
        </div>
      `).join("")}
    </div>
  `;
}

function renderDagNodeHistogram(target, data) {
  if (!target) return;
  target.histogramData = data;
  if (!data.ok || !data.bins?.length) {
    target.innerHTML = `<p class="empty-state">${data.message || "No histogram data available."}</p>`;
    return;
  }
  const node = target.closest("[data-dag-node]");
  const thresholdValue = Number(node?.querySelector("[data-node-threshold]")?.value);
  const hasThreshold = Number.isFinite(thresholdValue) && data.maximum !== data.minimum;
  const thresholdLeft = hasThreshold ? thresholdRatio(thresholdValue, data) * 100 : null;
  const maxCount = Math.max(...data.bins.map((bin) => bin.count), 1);
  const bins = data.bins.map((bin) => {
    const height = Math.max(8, Math.round((bin.count / maxCount) * 100));
    const label = bin.label || `${formatNumber(bin.lower)}-${formatNumber(bin.upper)}`;
    const midpoint = Number.isFinite(Number(bin.lower)) && Number.isFinite(Number(bin.upper))
      ? (Number(bin.lower) + Number(bin.upper)) / 2
      : "";
    return `
      <span
        class="dag-histogram-bin"
        title="${label}: ${bin.count}"
        data-bin-label="${label}"
        data-bin-count="${bin.count}"
        data-bin-midpoint="${midpoint}">
        <i style="height:${height}%"></i>
      </span>
    `;
  }).join("");
  target.innerHTML = `
    <div class="dag-histogram-meta">
      <strong>${data.rows}/${data.requested_days} days</strong>
      <span>${data.aggregation || "last"}${Number(data.day_lag || 0) ? ` · lag ${data.day_lag}d` : ""}${data.time_start || data.time_end ? ` · ${data.time_start || "00:00"}-${data.time_end || "23:59"}` : ""}</span>
    </div>
    <div class="dag-histogram-bars">
      ${densitySvg(data.values || [], data.minimum, data.maximum, "density-overlay dag-density", "Fitted DAG box distribution line")}
      ${hasThreshold ? `<span class="dag-threshold-marker" style="left:${thresholdLeft}%"></span>` : ""}
      ${bins}
    </div>
    <div class="dag-histogram-axis">
      <span>${formatNumber(data.minimum)}</span>
      <span>${formatNumber(data.maximum)}</span>
    </div>
    ${renderDistributionStats(data)}
  `;
}

async function loadDagNodeHistogram(node) {
  const form = document.querySelector("[data-dag-form]");
  const target = node?.querySelector("[data-dag-node-histogram]");
  if (!form?.dataset.dagHistogramUrl || !node || !target) return;
  const sensor = dagNodeId(node);
  const aggregation = node.querySelector("[data-node-aggregation]")?.value || "last";
  const timeStart = node.querySelector("[data-node-time-start]")?.value || "";
  const timeEnd = node.querySelector("[data-node-time-end]")?.value || "";
  const dayLag = node.querySelector("[data-node-day-lag]")?.value || "0";
  const analysisDays = document.querySelector("[data-analysis-days]")?.value || "90";
  const requestKey = JSON.stringify([sensor, aggregation, timeStart, timeEnd, dayLag, analysisDays]);
  target.dataset.histogramRequestKey = requestKey;
  target.innerHTML = `
    <div class="empty-state dag-histogram-loading" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span>
      <span>Loading histogram.</span>
    </div>
  `;
  target.histogramData = null;
  if (node.classList.contains("is-selected")) syncDagPropertiesSidebar(node);
  const url = new URL(form.dataset.dagHistogramUrl, window.location.origin);
  url.searchParams.set("sensor", sensor);
  url.searchParams.set("aggregation", aggregation);
  url.searchParams.set("time_start", timeStart);
  url.searchParams.set("time_end", timeEnd);
  url.searchParams.set("day_lag", dayLag);
  url.searchParams.set("analysis_window_days", analysisDays);
  try {
    const response = await fetch(url);
    const data = await response.json();
    if (target.dataset.histogramRequestKey !== requestKey) return;
    renderDagNodeHistogram(target, data);
  } catch {
    if (target.dataset.histogramRequestKey !== requestKey) return;
    renderDagNodeHistogram(target, { ok: false, message: "Unable to load histogram.", bins: [] });
  }
  if (node.classList.contains("is-selected")) syncDagPropertiesSidebar(node);
}

function incomingDagSourceIds(targetId) {
  return Array.from(document.querySelectorAll("[data-dag-edge-row]"))
    .filter((row) => row.querySelector("[data-edge-target]")?.value.trim() === targetId)
    .map((row) => row.querySelector("[data-edge-source]")?.value.trim())
    .filter(Boolean);
}

function dagNodeById(nodeId) {
  return Array.from(document.querySelectorAll("[data-dag-node]")).find((node) => dagNodeId(node) === nodeId);
}

async function fetchDagHistogramForNode(node) {
  const form = document.querySelector("[data-dag-form]");
  if (!form?.dataset.dagHistogramUrl || !node) return null;
  const url = new URL(form.dataset.dagHistogramUrl, window.location.origin);
  url.searchParams.set("sensor", dagNodeId(node));
  url.searchParams.set("aggregation", node.querySelector("[data-node-aggregation]")?.value || "last");
  url.searchParams.set("time_start", node.querySelector("[data-node-time-start]")?.value || "");
  url.searchParams.set("time_end", node.querySelector("[data-node-time-end]")?.value || "");
  url.searchParams.set("day_lag", node.querySelector("[data-node-day-lag]")?.value || "0");
  url.searchParams.set("analysis_window_days", document.querySelector("[data-analysis-days]")?.value || "90");
  const response = await fetch(url);
  return response.json();
}

function renderIncomingHistogramOverlay(target, targetNode, series) {
  const valid = series.filter((row) => row.data?.ok && row.data.values?.length);
  if (!valid.length) {
    target.innerHTML = '<p class="empty-state">No incoming parent histograms are available for this box.</p>';
    return;
  }
  target.innerHTML = `
    <div class="dag-histogram-meta">
      <strong>Incoming overlays</strong>
      <span>${valid.length} parent${valid.length === 1 ? "" : "s"} -> ${dagNodeId(targetNode)}</span>
    </div>
    <div class="dag-overlay-plot">
      ${valid.map((row, index) => {
        const values = row.data.values || [];
        return `
          <div class="dag-overlay-layer color-${index % 5}">
            ${densitySvg(values, row.data.minimum, row.data.maximum, "density-overlay dag-overlay-density", `${row.id} distribution`)}
          </div>
        `;
      }).join("")}
    </div>
    <div class="dag-overlay-legend">
      ${valid.map((row, index) => `
        <span class="color-${index % 5}">
          <i></i>
          ${row.id}
          <small>${row.data.aggregation || "last"}${Number(row.data.day_lag || 0) ? `, lag ${row.data.day_lag}d` : ""}</small>
        </span>
      `).join("")}
    </div>
    <p class="dag-histogram-readout">Each line is scaled to that sensor's own daily distribution.</p>
  `;
}

async function loadIncomingHistogramOverlay(node) {
  const target = node?.querySelector("[data-dag-node-histogram]");
  if (!node || !target) return;
  const targetId = dagNodeId(node);
  const sourceIds = incomingDagSourceIds(targetId);
  if (!sourceIds.length) {
    target.innerHTML = '<p class="empty-state">No arrows point into this box.</p>';
    return;
  }
  target.innerHTML = '<p class="empty-state">Loading incoming histograms.</p>';
  const series = await Promise.all(sourceIds.map(async (id) => {
    const sourceNode = dagNodeById(id);
    if (!sourceNode) return { id, data: { ok: false } };
    try {
      return { id, data: await fetchDagHistogramForNode(sourceNode) };
    } catch {
      return { id, data: { ok: false } };
    }
  }));
  renderIncomingHistogramOverlay(target, node, series);
  if (node.classList.contains("is-selected")) syncDagPropertiesSidebar(node);
}

async function toggleCovariateDistribution(button) {
  const row = button.closest("tr");
  const distributionRow = row.nextElementSibling;
  if (!distributionRow?.matches(".covariate-distribution-row")) return;

  const target = distributionRow.querySelector("[data-covariate-distribution]");
  const table = row.closest("[data-covariate-histogram-url]");
  const sensor = row.querySelector('input[name="predictor_entities"]').value.trim();
  const aggregation = row.querySelector('select[name="predictor_aggregations"]').value;
  const dayOffset = row.querySelector('select[name="predictor_day_offsets"]').value;
  const analysisDays = document.querySelector("[data-analysis-days]").value || "90";
  const sleepSensor = document.querySelector("[data-outcome-sensor]")?.value || "";
  const outcomeOperator = document.querySelector("[data-outcome-operator]")?.value || "less_than";
  const outcomeThreshold = document.querySelector("[data-threshold-input]")?.value || "";
  const sleepStartSensor = document.querySelector('select[name="sleep_start_sensor"]')?.value || "";
  const excludeMinutes = document.querySelector('input[name="exclude_before_sleep_start_minutes"]')?.value || "0";

  distributionRow.hidden = !distributionRow.hidden;
  button.textContent = distributionRow.hidden ? "Show" : "Hide";
  if (distributionRow.hidden) return;

  target.innerHTML = '<p class="empty-state">Loading covariate distribution.</p>';
  const url = new URL(table.dataset.covariateHistogramUrl, window.location.origin);
  url.searchParams.set("sensor", sensor);
  url.searchParams.set("aggregation", aggregation);
  url.searchParams.set("day_offset", dayOffset);
  url.searchParams.set("analysis_window_days", analysisDays);
  url.searchParams.set("sleep_sensor", sleepSensor);
  url.searchParams.set("outcome_operator", outcomeOperator);
  url.searchParams.set("outcome_threshold", outcomeThreshold);
  url.searchParams.set("sleep_start_sensor", sleepStartSensor);
  url.searchParams.set("exclude_before_sleep_start_minutes", excludeMinutes);

  try {
    const response = await fetch(url);
    renderMiniHistogram(target, await response.json());
  } catch {
    renderMiniHistogram(target, {
      ok: false,
      message: "Unable to load covariate distribution.",
      bins: [],
    });
  }
}

function refreshOpenCovariateDistributions() {
  document.querySelectorAll("[data-covariate-toggle]").forEach((button) => {
    const row = button.closest("tr");
    const distributionRow = row?.nextElementSibling;
    if (distributionRow?.matches(".covariate-distribution-row") && !distributionRow.hidden) {
      toggleCovariateDistribution(button);
      toggleCovariateDistribution(button);
    }
  });
}

document.addEventListener("click", (event) => {
  const explainEquation = event.target.closest("[data-equation-explain-toggle]");
  if (explainEquation) {
    const panel = explainEquation.parentElement?.querySelector("[data-equation-explanation]");
    const expanded = explainEquation.getAttribute("aria-expanded") === "true";
    explainEquation.setAttribute("aria-expanded", String(!expanded));
    explainEquation.textContent = expanded ? "Explain equation" : "Hide explanation";
    if (panel) panel.hidden = expanded;
    return;
  }

  const addDagNode = event.target.closest("[data-dag-add-node]");
  if (addDagNode) {
    const canvas = document.querySelector("[data-dag-canvas]");
    if (!canvas) return;
    setDagZoom(canvas, 1);
    const count = canvas.querySelectorAll("[data-dag-node]").length + 1;
    canvas.insertAdjacentHTML("beforeend", dagNodeTemplate(count));
    const node = canvas.querySelector("[data-dag-node]:last-child");
    updateDagNodeSummary(node);
    setSelectedDagNode(node);
    resolveDagNodeOverlaps({ autosave: true });
    updateDagEdges();
    updateMethodEquation();
    scheduleDagAutosave();
    return;
  }

  const fitDag = event.target.closest("[data-dag-fit]");
  if (fitDag) {
    fitDagToScreen();
    return;
  }

  const addDagEdge = event.target.closest("[data-dag-add-edge]");
  if (addDagEdge) {
    const list = document.querySelector("[data-dag-edge-list]");
    if (!list) return;
    list.insertAdjacentHTML("beforeend", dagEdgeTemplate());
    updateDagEdges();
    return;
  }

  const removeDagNode = event.target.closest("[data-dag-remove-node]");
  if (removeDagNode) {
    deleteDagNodes([removeDagNode.closest("[data-dag-node]")]);
    return;
  }

  const deleteSelectedDagNodes = event.target.closest("[data-dag-delete-selected]");
  if (deleteSelectedDagNodes) {
    deleteSelectedDagItems();
    return;
  }

  const sidebarDeleteNode = event.target.closest("[data-sidebar-delete-node]");
  if (sidebarDeleteNode) {
    deleteDagNodes([selectedDagNode()]);
    return;
  }

  const removeDagEdge = event.target.closest("[data-dag-remove-edge]");
  if (removeDagEdge) {
    removeDagEdge.closest("[data-dag-edge-row]")?.remove();
    updateDagEdges();
    scheduleDagAutosave();
    return;
  }

  const renderedEdge = event.target.closest("[data-rendered-edge]");
  if (renderedEdge) {
    const row = Array.from(document.querySelectorAll("[data-dag-edge-row]"))
      .find((candidate) => edgeRowKey(candidate) === renderedEdge.dataset.edgeKey);
    const additive = event.ctrlKey || event.metaKey;
    setSelectedDagEdge(row || null, { additive, toggle: additive });
    return;
  }

  const loadHistogramButton = event.target.closest("[data-dag-load-histogram]");
  if (loadHistogramButton) {
    loadDagNodeHistogram(loadHistogramButton.closest("[data-dag-node]"));
    return;
  }

  const overlayIncomingButton = event.target.closest("[data-dag-overlay-incoming]");
  if (overlayIncomingButton) {
    loadIncomingHistogramOverlay(overlayIncomingButton.closest("[data-dag-node]"));
    return;
  }

  const histogramBin = event.target.closest("[data-bin-label]");
  if (histogramBin) {
    const readout = histogramBin.closest("[data-dag-node-histogram]")?.querySelector("[data-dag-histogram-readout]");
    if (readout) readout.textContent = `${histogramBin.dataset.binLabel}: ${histogramBin.dataset.binCount} days`;
  }

  const thresholdBin = event.target.closest("[data-bin-midpoint]");
  if (thresholdBin) {
    setDagNodeThresholdFromHistogram(thresholdBin);
    return;
  }

  const sidebarTab = event.target.closest("[data-sidebar-tab]");
  if (sidebarTab) {
    setSidebarStep(sidebarTab.dataset.sidebarTab);
    return;
  }

  const sidebarNext = event.target.closest("[data-sidebar-next]");
  if (sidebarNext) {
    setSidebarStep(sidebarNext.dataset.sidebarNext);
    return;
  }

  const dagNodeTab = event.target.closest("[data-dag-node-tab]");
  if (dagNodeTab) {
    const node = dagNodeTab.closest("[data-dag-node]");
    if (!node) return;
    const targetPanel = dagNodeTab.dataset.dagNodeTab;
    node.querySelectorAll("[data-dag-node-tab]").forEach((button) => {
      button.classList.toggle("active", button === dagNodeTab);
    });
    node.querySelectorAll("[data-dag-node-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.dagNodePanel !== targetPanel;
    });
    return;
  }

  const sectionToggle = event.target.closest("[data-toggle-section]");
  if (sectionToggle) {
    const section = document.getElementById(sectionToggle.dataset.toggleSection);
    if (!section) return;
    const expanded = sectionToggle.getAttribute("aria-expanded") !== "false";
    const nextExpanded = !expanded;
    sectionToggle.setAttribute("aria-expanded", String(nextExpanded));
    sectionToggle.setAttribute(
      "aria-label",
      expanded ? "Expand causal DAG" : "Collapse causal DAG"
    );
    sectionToggle.title = expanded ? "Expand causal DAG" : "Collapse causal DAG";
    section.hidden = false;
    section.closest(".workflow-dag-section")?.classList.toggle("is-dag-collapsed", !nextExpanded);
    if (nextExpanded) {
      requestAnimationFrame(() => {
        updateDagEdges();
        requestAnimationFrame(updateDagEdges);
      });
    }
    return;
  }

  const addButton = event.target.closest("[data-add-row]");
  if (addButton) {
    const table = document.getElementById(addButton.dataset.addRow);
    if (!table) return;
    table.querySelector("tbody").appendChild(createEntityRow(table));
    table.querySelector("tbody tr:last-child input").focus();
    scheduleAutosaveSettings();
    return;
  }

  const removeButton = event.target.closest("[data-remove-row]");
  if (removeButton) {
    const row = removeButton.closest("tr");
    const distributionRow = row.nextElementSibling;
    if (distributionRow?.matches(".covariate-distribution-row")) {
      distributionRow.remove();
    }
    row.remove();
    scheduleAutosaveSettings();
    return;
  }

  const covariateToggle = event.target.closest("[data-covariate-toggle]");
  if (covariateToggle) {
    toggleCovariateDistribution(covariateToggle);
  }
});

document.addEventListener("mouseover", (event) => {
  const histogramBin = event.target.closest("[data-bin-label]");
  if (!histogramBin) return;
  const readout = histogramBin.closest("[data-dag-node-histogram]")?.querySelector("[data-dag-histogram-readout]");
  if (readout) readout.textContent = `${histogramBin.dataset.binLabel}: ${histogramBin.dataset.binCount} days`;
});

function dagNodeCenter(node, canvas) {
  const nodeRect = node.getBoundingClientRect();
  const canvasRect = canvas.getBoundingClientRect();
  return {
    x: ((nodeRect.left - canvasRect.left + nodeRect.width / 2) / canvasRect.width) * 1000,
    y: ((nodeRect.top - canvasRect.top + nodeRect.height / 2) / canvasRect.height) * 620,
  };
}

function dagNodePort(node, canvas, port = "auto", targetNode = null) {
  const nodeRect = node.getBoundingClientRect();
  const canvasRect = canvas.getBoundingClientRect();
  let side = port;
  if (side === "auto" && targetNode) {
    const targetRect = targetNode.getBoundingClientRect();
    const dx = targetRect.left + targetRect.width / 2 - (nodeRect.left + nodeRect.width / 2);
    const dy = targetRect.top + targetRect.height / 2 - (nodeRect.top + nodeRect.height / 2);
    side = Math.abs(dx) > Math.abs(dy) ? (dx >= 0 ? "right" : "left") : (dy >= 0 ? "bottom" : "top");
  }
  const points = {
    top: { x: nodeRect.left + nodeRect.width / 2, y: nodeRect.top },
    right: { x: nodeRect.right, y: nodeRect.top + nodeRect.height / 2 },
    bottom: { x: nodeRect.left + nodeRect.width / 2, y: nodeRect.bottom },
    left: { x: nodeRect.left, y: nodeRect.top + nodeRect.height / 2 },
  };
  const point = points[side] || points.right;
  return {
    x: ((point.x - canvasRect.left) / canvasRect.width) * 1000,
    y: ((point.y - canvasRect.top) / canvasRect.height) * 620,
    side,
  };
}

function dagCanvasPoint(event, canvas) {
  const canvasRect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - canvasRect.left) / canvasRect.width) * 1000,
    y: ((event.clientY - canvasRect.top) / canvasRect.height) * 620,
  };
}

function dagNodeId(node) {
  return node?.querySelector("[data-node-id-input]")?.value.trim() || node?.dataset.nodeId || "";
}

function ensureDagEdgeRow(source, target, label = "") {
  if (!source || !target || source === target) return false;
  const list = document.querySelector("[data-dag-edge-list]");
  if (!list) return false;
  const exists = Array.from(list.querySelectorAll("[data-dag-edge-row]")).some((row) => {
    return (
      row.querySelector("[data-edge-source]")?.value.trim() === source
      && row.querySelector("[data-edge-target]")?.value.trim() === target
    );
  });
  if (exists) return false;
  list.insertAdjacentHTML("beforeend", dagEdgeTemplate());
  const row = list.querySelector("[data-dag-edge-row]:last-child");
  row.querySelector("[data-edge-source]").value = source;
  row.querySelector("[data-edge-target]").value = target;
  row.querySelector("[data-edge-label]").value = label;
  return true;
}

function updateDagDeleteButton() {
  updateDagSelectionUi();
}

function removeDagEdgesForNodeIds(nodeIds) {
  const ids = new Set(nodeIds);
  document.querySelectorAll("[data-dag-edge-row]").forEach((row) => {
    const source = row.querySelector("[data-edge-source]")?.value.trim();
    const target = row.querySelector("[data-edge-target]")?.value.trim();
    if (ids.has(source) || ids.has(target)) row.remove();
  });
}

function deleteDagNodes(nodes) {
  const selectedNodes = Array.from(nodes || []).filter(Boolean);
  if (!selectedNodes.length) return false;
  const nodeIds = selectedNodes.map(dagNodeId).filter(Boolean);
  removeDagEdgesForNodeIds(nodeIds);
  selectedNodes.forEach((node) => node.remove());
  updateDagEdges();
  updateDagSelectionUi();
  updateMethodEquation();
  scheduleDagAutosave();
  return true;
}

function deleteDagEdges(rows) {
  const selectedRows = Array.from(rows || []).filter(Boolean);
  if (!selectedRows.length) return false;
  selectedRows.forEach((row) => row.remove());
  updateDagEdges();
  updateDagSelectionUi();
  scheduleDagAutosave();
  return true;
}

function dagPortVector(side) {
  if (side === "left") return { x: -1, y: 0 };
  if (side === "top") return { x: 0, y: -1 };
  if (side === "bottom") return { x: 0, y: 1 };
  return { x: 1, y: 0 };
}

function dagCurvedEdgePath(start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const distance = Math.hypot(dx, dy) || 1;
  const sourceVector = dagPortVector(start.side);
  const targetVector = dagPortVector(end.side);
  const controlDistance = Math.min(180, Math.max(52, distance * 0.38));
  const softBend = Math.min(36, Math.max(10, distance * 0.065));
  const horizontalEdge = Math.abs(dx) > Math.abs(dy) * 1.25;
  const verticalEdge = Math.abs(dy) > Math.abs(dx) * 1.25;
  const bend = {
    x: verticalEdge ? (dx >= 0 ? softBend : -softBend) : 0,
    y: horizontalEdge ? (dy >= 0 ? softBend : -softBend) : 0,
  };
  const c1 = {
    x: start.x + sourceVector.x * controlDistance + bend.x,
    y: start.y + sourceVector.y * controlDistance + bend.y,
  };
  const c2 = {
    x: end.x + targetVector.x * controlDistance + bend.x,
    y: end.y + targetVector.y * controlDistance + bend.y,
  };
  return `M ${start.x} ${start.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${end.x} ${end.y}`;
}

function updateDagEdges() {
  const canvas = document.querySelector("[data-dag-canvas]");
  const svg = document.querySelector("[data-dag-edges-svg]");
  if (!canvas || !svg) return;
  svg.querySelectorAll("[data-rendered-edge]").forEach((line) => line.remove());
  updateDagCycleWarning();
  const nodes = new Map();
  canvas.querySelectorAll("[data-dag-node]").forEach((node) => {
    const input = node.querySelector("[data-node-id-input]");
    const id = input?.value.trim() || node.dataset.nodeId;
    node.dataset.nodeId = id;
    nodes.set(id, node);
  });
  document.querySelectorAll("[data-dag-edge-row]").forEach((row) => {
    const source = row.querySelector("[data-edge-source]")?.value.trim();
    const target = row.querySelector("[data-edge-target]")?.value.trim();
    if (!nodes.has(source) || !nodes.has(target) || source === target) return;
    const sourceNode = nodes.get(source);
    const targetNode = nodes.get(target);
    const start = dagNodePort(sourceNode, canvas, "auto", targetNode);
    const end = dagNodePort(targetNode, canvas, "auto", sourceNode);
    const edge = document.createElementNS("http://www.w3.org/2000/svg", "path");
    edge.setAttribute("d", dagCurvedEdgePath(start, end));
    edge.setAttribute("tabindex", "0");
    edge.setAttribute("role", "button");
    edge.setAttribute("aria-label", `Select edge from ${source} to ${target}`);
    if (row.dataset.cyclic === "1") edge.classList.add("dag-cycle-edge");
    const selected = row.classList.contains("is-selected");
    edge.setAttribute("marker-end", selected ? "url(#dag-arrow-selected)" : "url(#dag-arrow)");
    if (selected) edge.classList.add("is-selected");
    edge.dataset.renderedEdge = "1";
    edge.dataset.edgeKey = edgeRowKey(row);
    svg.appendChild(edge);
  });
}

function dagBounds(nodes) {
  return nodes.reduce(
    (acc, node) => {
      const left = node.offsetLeft;
      const top = node.offsetTop;
      const right = left + node.offsetWidth;
      const bottom = top + node.offsetHeight;
      return {
        left: Math.min(acc.left, left),
        top: Math.min(acc.top, top),
        right: Math.max(acc.right, right),
        bottom: Math.max(acc.bottom, bottom),
      };
    },
    { left: Infinity, top: Infinity, right: -Infinity, bottom: -Infinity }
  );
}

function centerDagNodesInView({ autosave = false } = {}) {
  const scroll = document.querySelector("[data-dag-scroll]");
  const canvas = document.querySelector("[data-dag-canvas]");
  if (!scroll || !canvas) return null;
  const nodes = Array.from(canvas.querySelectorAll("[data-dag-node]"));
  if (!nodes.length) return null;

  const padding = DAG_NODE_MARGIN;
  const bounds = dagBounds(nodes);
  const graphWidth = bounds.right - bounds.left;
  const viewportWidth = scroll.clientWidth || scroll.parentElement?.clientWidth || 900;
  const logicalViewportWidth = viewportWidth / dagZoom(canvas);
  if (graphWidth + padding * 2 > logicalViewportWidth) return bounds;

  const desiredLeft = Math.max(padding, (logicalViewportWidth - graphWidth) / 2);
  const deltaX = desiredLeft - bounds.left;
  if (Math.abs(deltaX) < 1) return bounds;

  nodes.forEach((node) => {
    const nextX = Math.max(padding, node.offsetLeft + deltaX);
    node.style.left = `${Math.round(nextX)}px`;
    node.querySelector("[data-node-x]").value = Math.round(nextX);
  });
  updateDagEdges();
  if (autosave) scheduleDagAutosave();
  return dagBounds(nodes);
}

function clampDagNodeIntoView(node) {
  if (!node) return false;
  const canvas = document.querySelector("[data-dag-canvas]");
  const padding = DAG_NODE_MARGIN;
  const maxLeft = Math.max(padding, (canvas?.offsetWidth || 1800) - node.offsetWidth - padding);
  const maxTop = Math.max(padding, (canvas?.offsetHeight || 620) - node.offsetHeight - padding);
  const nextX = Math.min(maxLeft, Math.max(padding, node.offsetLeft));
  const nextY = Math.min(maxTop, Math.max(padding, node.offsetTop));
  if (Math.abs(nextX - node.offsetLeft) < 1 && Math.abs(nextY - node.offsetTop) < 1) return false;
  node.style.left = `${Math.round(nextX)}px`;
  node.style.top = `${Math.round(nextY)}px`;
  node.querySelector("[data-node-x]").value = Math.round(nextX);
  node.querySelector("[data-node-y]").value = Math.round(nextY);
  return true;
}

function clampAllDagNodesIntoView({ autosave = false } = {}) {
  let changed = false;
  document.querySelectorAll("[data-dag-node]").forEach((node) => {
    changed = clampDagNodeIntoView(node) || changed;
  });
  if (changed) {
    updateDagEdges();
    if (autosave) scheduleDagAutosave();
  }
  return changed;
}

function dagNodesOverlap(first, second, margin = 24) {
  return !(
    first.offsetLeft + first.offsetWidth + margin <= second.offsetLeft
    || second.offsetLeft + second.offsetWidth + margin <= first.offsetLeft
    || first.offsetTop + first.offsetHeight + margin <= second.offsetTop
    || second.offsetTop + second.offsetHeight + margin <= first.offsetTop
  );
}

function moveDagNode(node, x, y) {
  const nextX = Math.round(x);
  const nextY = Math.round(y);
  node.style.left = `${nextX}px`;
  node.style.top = `${nextY}px`;
  node.querySelector("[data-node-x]").value = nextX;
  node.querySelector("[data-node-y]").value = nextY;
}

function resolveDagNodeOverlaps({ autosave = false } = {}) {
  const canvas = document.querySelector("[data-dag-canvas]");
  if (!canvas) return false;
  const nodes = Array.from(canvas.querySelectorAll("[data-dag-node]"))
    .sort((first, second) => first.offsetTop - second.offsetTop || first.offsetLeft - second.offsetLeft);
  let changed = false;
  const margin = 28;

  for (let pass = 0; pass < Math.max(1, nodes.length); pass += 1) {
    let movedThisPass = false;
    for (let index = 0; index < nodes.length; index += 1) {
      const node = nodes[index];
      let nextY = node.offsetTop;
      for (let otherIndex = 0; otherIndex < nodes.length; otherIndex += 1) {
        if (index === otherIndex) continue;
        const other = nodes[otherIndex];
        if (!dagNodesOverlap(node, other, margin)) continue;
        if (node.offsetTop < other.offsetTop || (node.offsetTop === other.offsetTop && node.offsetLeft < other.offsetLeft)) {
          continue;
        }
        nextY = Math.max(nextY, other.offsetTop + other.offsetHeight + margin);
      }
      if (nextY !== node.offsetTop) {
        moveDagNode(node, node.offsetLeft, nextY);
        movedThisPass = true;
        changed = true;
      }
    }
    if (!movedThisPass) break;
  }

  if (changed) {
    const bottom = Math.max(...nodes.map((node) => node.offsetTop + node.offsetHeight), 620);
    canvas.style.minHeight = `${Math.max(canvas.offsetHeight, bottom + DAG_NODE_MARGIN)}px`;
    clampAllDagNodesIntoView();
    updateDagEdges();
    if (autosave) scheduleDagAutosave();
  }
  return changed;
}

function fitDagToScreen() {
  const scroll = document.querySelector("[data-dag-scroll]");
  const canvas = document.querySelector("[data-dag-canvas]");
  if (!scroll || !canvas) return;
  const nodes = Array.from(canvas.querySelectorAll("[data-dag-node]"));
  if (!nodes.length) {
    scroll.classList.add("is-fit");
    scroll.style.height = "320px";
    setDagZoom(canvas, 1);
    return;
  }
  let bounds = dagBounds(nodes);
  const padding = DAG_NODE_MARGIN;
  const width = Math.max(1, bounds.right - bounds.left + padding * 2);
  const height = Math.max(1, bounds.bottom - bounds.top + padding * 2);
  const viewportTop = scroll.getBoundingClientRect().top;
  const maxViewportHeight = Math.max(320, window.innerHeight - viewportTop - 72);
  const availableWidth = scroll.parentElement?.clientWidth || scroll.clientWidth || 900;
  const zoom = Math.min(1, Math.max(0.35, Math.min(
    1,
    (availableWidth - 24) / width,
    (maxViewportHeight - 24) / height
  )));
  const fittedHeight = Math.ceil(height * zoom + 24);
  scroll.classList.add("is-fit", "is-resizing");
  scroll.style.height = `${Math.max(260, Math.min(fittedHeight, maxViewportHeight))}px`;
  canvas.style.width = `${Math.max(availableWidth / zoom, bounds.right + padding)}px`;
  canvas.style.minHeight = `${Math.max(bounds.bottom + padding, scroll.clientHeight / zoom)}px`;
  setDagZoom(canvas, zoom);
  requestAnimationFrame(() => {
    bounds = centerDagNodesInView({ autosave: true }) || bounds;
    scroll.scrollTo({
      left: Math.max(0, ((bounds.left + bounds.right) / 2) * dagZoom(canvas) - scroll.clientWidth / 2),
      top: Math.max(0, (bounds.top - padding) * dagZoom(canvas)),
      behavior: "smooth",
    });
  });
}

function ensureDagNodeFits(node) {
  const scroll = document.querySelector("[data-dag-scroll]");
  const canvas = document.querySelector("[data-dag-canvas]");
  if (!scroll || !canvas || !node) return;
  const zoom = dagZoom(canvas);
  const padding = DAG_NODE_MARGIN;
  if (!node.dataset.preExpandY) {
    node.dataset.preExpandY = String(node.offsetTop);
  }
  if (node.offsetTop < padding) {
    const shift = padding - node.offsetTop;
    node.style.top = `${padding}px`;
    node.querySelector("[data-node-y]").value = padding;
    canvas.querySelectorAll("[data-dag-node]").forEach((otherNode) => {
      if (otherNode === node) return;
      const nextTop = otherNode.offsetTop + shift;
      otherNode.style.top = `${nextTop}px`;
      otherNode.querySelector("[data-node-y]").value = Math.round(nextTop);
    });
    canvas.style.minHeight = `${Math.max(canvas.offsetHeight, canvas.scrollHeight + shift)}px`;
  }
  const nodeTop = Math.max(0, Math.floor((node.offsetTop - padding) * zoom));
  const neededHeight = Math.ceil((node.offsetTop + node.offsetHeight + 64) * zoom);
  if (neededHeight > scroll.clientHeight) {
    scroll.style.height = `${neededHeight}px`;
    scroll.classList.add("is-resizing");
  }
  const neededCanvasWidth = Math.ceil(node.offsetLeft + node.offsetWidth + 140);
  if (neededCanvasWidth > canvas.offsetWidth) {
    canvas.style.width = `${neededCanvasWidth}px`;
  }
  if (nodeTop < scroll.scrollTop || (node.offsetTop + node.offsetHeight + padding) * zoom > scroll.scrollTop + scroll.clientHeight) {
    scroll.scrollTo({
      top: nodeTop,
      left: Math.max(0, Math.floor((node.offsetLeft - padding) * zoom)),
      behavior: "smooth",
    });
  }
  updateDagEdges();
}

function initializeDagEditor() {
  const canvas = document.querySelector("[data-dag-canvas]");
  const scroll = document.querySelector("[data-dag-scroll]");
  const svg = document.querySelector("[data-dag-edges-svg]");
  if (!canvas) return;
  let activeNode = null;
  let activeConnector = null;
  let activePan = null;
  let activeSelection = null;
  let activeGroupDrag = null;
  let activeResize = null;
  let activeSectionResize = null;
  let activeSidebarResize = null;
  let startX = 0;
  let startY = 0;
  let nodeStartX = 0;
  let nodeStartY = 0;

  canvas.addEventListener("pointerdown", (event) => {
    const resizeHandle = event.target.closest("[data-dag-resize-handle]");
    if (resizeHandle) {
      const node = resizeHandle.closest("[data-dag-node]");
      if (!node || node.classList.contains("is-collapsed")) return;
      event.preventDefault();
      activeResize = {
        node,
        startX: event.clientX,
        startY: event.clientY,
        width: node.offsetWidth,
        height: node.offsetHeight,
        zoom: dagZoom(canvas),
        axis: resizeHandle.dataset.resizeAxis || "both",
      };
      node.classList.add("is-resizing");
      resizeHandle.setPointerCapture(event.pointerId);
      return;
    }

    if (event.target.closest("[data-rendered-edge]")) {
      return;
    }

    const handle = event.target.closest("[data-dag-connect-handle]");
    if (handle) {
      const node = handle.closest("[data-dag-node]");
      if (!node || !svg) return;
      event.preventDefault();
      event.stopPropagation();
      const start = dagNodePort(node, canvas, handle.dataset.port || "right");
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.classList.add("dag-draft-edge");
      line.setAttribute("x1", start.x);
      line.setAttribute("y1", start.y);
      line.setAttribute("x2", start.x);
      line.setAttribute("y2", start.y);
      line.setAttribute("marker-end", "url(#dag-arrow)");
      svg.appendChild(line);
      activeConnector = { sourceNode: node, sourcePort: handle.dataset.port || "right", line };
      handle.setPointerCapture(event.pointerId);
      canvas.classList.add("is-connecting");
      return;
    }

    const node = event.target.closest("[data-dag-node]");
    if (!node && scroll) {
      if (event.button === 1 || event.altKey || event.metaKey || event.ctrlKey) {
        event.preventDefault();
        activePan = {
          startX: event.clientX,
          startY: event.clientY,
          scrollLeft: scroll.scrollLeft,
          scrollTop: scroll.scrollTop,
        };
        canvas.setPointerCapture(event.pointerId);
        scroll.classList.add("is-panning");
        return;
      }
      if (event.button === 0) {
        event.preventDefault();
        clearDagSelection();
        const point = dagCanvasPoint(event, canvas);
        const rect = document.createElement("div");
        rect.className = "dag-selection-rect";
        rect.style.left = `${(point.x / 1000) * canvas.offsetWidth}px`;
        rect.style.top = `${(point.y / 620) * canvas.offsetHeight}px`;
        rect.style.width = "0px";
        rect.style.height = "0px";
        canvas.appendChild(rect);
        activeSelection = { startX: point.x, startY: point.y, rect };
        canvas.setPointerCapture(event.pointerId);
        return;
      }
    }
    if (!node || event.target.matches("input, select, button, textarea")) return;
    event.preventDefault();
    const additive = event.ctrlKey || event.metaKey;
    if (additive) {
      setSelectedDagNode(node, { additive: true, toggle: true });
      return;
    }
    const selectedNodes = selectedDagNodes();
    if (node.classList.contains("is-selected") && selectedNodes.length > 1) {
      activeGroupDrag = {
        startX: event.clientX,
        startY: event.clientY,
        zoom: dagZoom(canvas),
        nodes: selectedNodes.map((selectedNode) => ({
          node: selectedNode,
          x: Number(selectedNode.querySelector("[data-node-x]").value) || selectedNode.offsetLeft,
          y: Number(selectedNode.querySelector("[data-node-y]").value) || selectedNode.offsetTop,
        })),
      };
      node.setPointerCapture(event.pointerId);
      canvas.classList.add("is-group-dragging");
      return;
    }
    setSelectedDagNode(node);
    activeNode = node;
    startX = event.clientX;
    startY = event.clientY;
    nodeStartX = Number(node.querySelector("[data-node-x]").value) || node.offsetLeft;
    nodeStartY = Number(node.querySelector("[data-node-y]").value) || node.offsetTop;
    node.setPointerCapture(event.pointerId);
    node.classList.add("is-dragging");
  });

  document.addEventListener("pointerdown", (event) => {
    const sidebarHandle = event.target.closest("[data-sidebar-resize-handle]");
    if (sidebarHandle) {
      const sidebar = document.querySelector("[data-dag-properties-sidebar]");
      const layout = document.querySelector(".dag-editor-layout");
      if (!sidebar || !layout) return;
      event.preventDefault();
      activeSidebarResize = {
        startX: event.clientX,
        width: sidebar.offsetWidth,
        sidebar,
        layout,
      };
      sidebarHandle.setPointerCapture(event.pointerId);
      sidebar.classList.add("is-resizing");
      return;
    }

    const sectionHandle = event.target.closest("[data-dag-section-resize-handle]");
    if (!sectionHandle) return;
    const scrollArea = document.querySelector("[data-dag-scroll]");
    if (!scrollArea || scrollArea.closest(".workflow-dag-section")?.classList.contains("is-dag-collapsed")) return;
    event.preventDefault();
    activeSectionResize = {
      startY: event.clientY,
      height: scrollArea.offsetHeight,
      scrollArea,
    };
    sectionHandle.setPointerCapture(event.pointerId);
    scrollArea.classList.add("is-resizing");
  });

  document.addEventListener("pointermove", (event) => {
    if (activeSidebarResize) {
      const width = Math.max(280, activeSidebarResize.width - (event.clientX - activeSidebarResize.startX));
      activeSidebarResize.layout.style.setProperty("--dag-sidebar-width", `${Math.round(width)}px`);
      updateDagEdges();
      return;
    }
    if (!activeSectionResize) return;
    activeSectionResize.scrollArea.style.height = `${Math.max(360, activeSectionResize.height + event.clientY - activeSectionResize.startY)}px`;
    updateDagEdges();
  });

  document.addEventListener("pointerup", () => {
    if (activeSidebarResize) {
      activeSidebarResize.sidebar.classList.remove("is-resizing");
      activeSidebarResize = null;
      updateDagEdges();
      return;
    }
    if (!activeSectionResize) return;
    activeSectionResize.scrollArea.classList.remove("is-resizing");
    activeSectionResize = null;
    updateDagEdges();
  });

  document.addEventListener("keydown", (event) => {
    if (!["Backspace", "Delete"].includes(event.key)) return;
    if (event.target.matches("input, select, textarea") || event.target.isContentEditable) return;
    if (!selectedDagNodes().length && !selectedDagEdgeRows().length) return;
    event.preventDefault();
    deleteSelectedDagItems();
  });

  canvas.addEventListener("pointermove", (event) => {
    if (activeResize) {
      const width = Math.max(420, activeResize.width + (event.clientX - activeResize.startX) / activeResize.zoom);
      const height = Math.max(220, activeResize.height + (event.clientY - activeResize.startY) / activeResize.zoom);
      if (activeResize.axis === "x" || activeResize.axis === "both") {
        activeResize.node.style.width = `${Math.round(width)}px`;
      }
      if (activeResize.axis === "y" || activeResize.axis === "both") {
        activeResize.node.style.height = `${Math.round(height)}px`;
      }
      updateDagEdges();
      return;
    }

    if (activeSelection) {
      const point = dagCanvasPoint(event, canvas);
      const left = Math.min(activeSelection.startX, point.x);
      const top = Math.min(activeSelection.startY, point.y);
      const width = Math.abs(point.x - activeSelection.startX);
      const height = Math.abs(point.y - activeSelection.startY);
      activeSelection.rect.style.left = `${(left / 1000) * canvas.offsetWidth}px`;
      activeSelection.rect.style.top = `${(top / 620) * canvas.offsetHeight}px`;
      activeSelection.rect.style.width = `${(width / 1000) * canvas.offsetWidth}px`;
      activeSelection.rect.style.height = `${(height / 620) * canvas.offsetHeight}px`;
      const selectionBox = activeSelection.rect.getBoundingClientRect();
      canvas.querySelectorAll("[data-dag-node]").forEach((candidate) => {
        const box = candidate.getBoundingClientRect();
        const intersects = !(box.right < selectionBox.left || box.left > selectionBox.right || box.bottom < selectionBox.top || box.top > selectionBox.bottom);
        candidate.classList.toggle("is-selected", intersects);
      });
      updateDagDeleteButton();
      return;
    }

    if (activeGroupDrag) {
      const dx = (event.clientX - activeGroupDrag.startX) / activeGroupDrag.zoom;
      const dy = (event.clientY - activeGroupDrag.startY) / activeGroupDrag.zoom;
      const canvasRect = canvas.getBoundingClientRect();
      const logicalWidth = canvasRect.width / activeGroupDrag.zoom;
      const logicalHeight = canvasRect.height / activeGroupDrag.zoom;
      activeGroupDrag.nodes.forEach((entry) => {
        const maxX = Math.max(DAG_NODE_MARGIN, logicalWidth - entry.node.offsetWidth - DAG_NODE_MARGIN);
        const maxY = Math.max(DAG_NODE_MARGIN, logicalHeight - entry.node.offsetHeight - DAG_NODE_MARGIN);
        const nextX = Math.min(maxX, Math.max(DAG_NODE_MARGIN, entry.x + dx));
        const nextY = Math.min(maxY, Math.max(DAG_NODE_MARGIN, entry.y + dy));
        entry.node.style.left = `${nextX}px`;
        entry.node.style.top = `${nextY}px`;
        entry.node.querySelector("[data-node-x]").value = Math.round(nextX);
        entry.node.querySelector("[data-node-y]").value = Math.round(nextY);
      });
      updateDagEdges();
      return;
    }

    if (activePan && scroll) {
      scroll.scrollLeft = activePan.scrollLeft - (event.clientX - activePan.startX);
      scroll.scrollTop = activePan.scrollTop - (event.clientY - activePan.startY);
      return;
    }

    if (activeConnector) {
      const point = dagCanvasPoint(event, canvas);
      activeConnector.line.setAttribute("x2", point.x);
      activeConnector.line.setAttribute("y2", point.y);
      canvas.querySelectorAll("[data-dag-node]").forEach((node) => {
        node.classList.toggle("is-connect-target", node !== activeConnector.sourceNode && node.matches(":hover"));
      });
      return;
    }

    if (!activeNode) return;
    const canvasRect = canvas.getBoundingClientRect();
    const zoom = dagZoom(canvas);
    const logicalWidth = canvasRect.width / zoom;
    const logicalHeight = canvasRect.height / zoom;
    const maxX = Math.max(DAG_NODE_MARGIN, logicalWidth - activeNode.offsetWidth - DAG_NODE_MARGIN);
    const maxY = Math.max(DAG_NODE_MARGIN, logicalHeight - activeNode.offsetHeight - DAG_NODE_MARGIN);
    const nextX = Math.min(maxX, Math.max(DAG_NODE_MARGIN, nodeStartX + (event.clientX - startX) / zoom));
    const nextY = Math.min(maxY, Math.max(DAG_NODE_MARGIN, nodeStartY + (event.clientY - startY) / zoom));
    activeNode.style.left = `${nextX}px`;
    activeNode.style.top = `${nextY}px`;
    activeNode.querySelector("[data-node-x]").value = Math.round(nextX);
    activeNode.querySelector("[data-node-y]").value = Math.round(nextY);
    updateDagEdges();
  });

  canvas.addEventListener("pointerup", (event) => {
    if (activeResize) {
      activeResize.node.classList.remove("is-resizing");
      activeResize = null;
      updateDagEdges();
      return;
    }

    if (activeSelection) {
      activeSelection.rect.remove();
      activeSelection = null;
      updateDagDeleteButton();
      return;
    }

    if (activeGroupDrag) {
      activeGroupDrag = null;
      canvas.classList.remove("is-group-dragging");
      scheduleDagAutosave();
      return;
    }

    if (activePan) {
      activePan = null;
      scroll?.classList.remove("is-panning");
      return;
    }

    if (activeConnector) {
      const targetNode = document.elementFromPoint(event.clientX, event.clientY)?.closest("[data-dag-node]");
      const source = dagNodeId(activeConnector.sourceNode);
      const target = dagNodeId(targetNode);
      activeConnector.line.remove();
      if (targetNode && targetNode !== activeConnector.sourceNode && ensureDagEdgeRow(source, target)) {
        updateDagEdges();
        scheduleDagAutosave();
      }
      canvas.querySelectorAll(".is-connect-target").forEach((node) => node.classList.remove("is-connect-target"));
      canvas.classList.remove("is-connecting");
      activeConnector = null;
      return;
    }

    activeNode?.classList.remove("is-dragging");
    if (activeNode) scheduleDagAutosave();
    activeNode = null;
  });

  canvas.addEventListener("pointercancel", () => {
    activePan = null;
    activeSidebarResize?.sidebar.classList.remove("is-resizing");
    activeSidebarResize = null;
    activeResize?.node.classList.remove("is-resizing");
    activeResize = null;
    activeGroupDrag = null;
    activeSelection?.rect.remove();
    activeSelection = null;
    activeNode?.classList.remove("is-dragging");
    activeNode = null;
    if (activeConnector) {
      activeConnector.line.remove();
      activeConnector = null;
      canvas.classList.remove("is-connecting");
    }
    canvas.classList.remove("is-group-dragging");
    scroll?.classList.remove("is-panning");
  });

  document.addEventListener("input", (event) => {
    if (event.target.matches("[data-sidebar-field]")) {
      applySidebarFieldToNode(event.target);
      return;
    }
    if (event.target.closest("[data-dag-form]")) {
      const node = event.target.closest("[data-dag-node]");
      if (event.target.matches("[data-node-role]") && node) {
        node.classList.remove("exposure", "covariate", "outcome", "latent");
        node.classList.add(event.target.value);
        updateDagNodeTimeMarker(node, { applyDefaultLag: true });
      }
      if (event.target.matches("[data-node-label-input]") && node) {
        const title = node.querySelector("[data-node-title]");
        if (title) title.textContent = event.target.value.trim() || "Untitled";
      }
      if (event.target.matches("[data-node-value-type]") && node) {
        updateDagEncodingControls(node);
      }
      if (event.target.matches("[data-node-aggregation], [data-node-time-start], [data-node-time-end], [data-node-day-lag], [data-node-id-input], [data-node-threshold], [data-node-value-type], [data-node-threshold-operator]") && node) {
        const target = node.querySelector("[data-dag-node-histogram]");
        if (target?.innerHTML.trim()) {
          if (event.target.matches("[data-node-threshold], [data-node-value-type], [data-node-threshold-operator]") && target.histogramData) {
            renderDagNodeHistogram(target, target.histogramData);
          } else {
            loadDagNodeHistogram(node);
          }
        }
      }
      updateDagEdges();
      scheduleDagAutosave();
    }
  });
  document.addEventListener("change", (event) => {
    if (event.target.matches("[data-sidebar-field]")) {
      applySidebarFieldToNode(event.target);
      return;
    }
    if (event.target.closest("[data-dag-form]")) {
      const node = event.target.closest("[data-dag-node]");
      if (event.target.matches("[data-node-value-type]") && node) {
        updateDagEncodingControls(node);
      }
      if (node && event.target.matches("[data-node-aggregation], [data-node-time-start], [data-node-time-end], [data-node-day-lag], [data-node-id-input], [data-node-threshold], [data-node-value-type], [data-node-threshold-operator]")) {
        const target = node.querySelector("[data-dag-node-histogram]");
        if (target?.innerHTML.trim()) {
          if (event.target.matches("[data-node-threshold], [data-node-value-type], [data-node-threshold-operator]") && target.histogramData) {
            renderDagNodeHistogram(target, target.histogramData);
          } else {
            loadDagNodeHistogram(node);
          }
        }
      }
      updateDagEdges();
      scheduleDagAutosave();
    }
  });
  window.addEventListener("resize", updateDagEdges);
  requestAnimationFrame(() => {
    clampAllDagNodesIntoView({ autosave: true });
    resolveDagNodeOverlaps({ autosave: true });
    updateDagTimeMarkers({ applyDefaultLags: true, autosave: true });
    updateAllDagEncodingControls();
    document.querySelectorAll("[data-dag-node]").forEach(updateDagNodeSummary);
    syncDagPropertiesSidebar(null);
    updateDagDeleteButton();
    updateDagEdges();
    requestAnimationFrame(() => {
      fitDagToScreen();
    });
  });
}

document.addEventListener("change", (event) => {
  syncAnalysisDayInputs(event.target);
  if (event.target.matches("[data-outcome-sensor], [data-analysis-days]")) {
    loadHistogram();
    refreshOpenCovariateDistributions();
    scheduleAutosaveSettings();
  }
  if (event.target.matches("[data-analysis-days-secondary]")) {
    scheduleAutosaveSettings();
  }
  if (event.target.matches("[data-threshold-input], [data-outcome-operator]")) {
    updateThresholdHours();
    const container = document.querySelector("[data-histogram]");
    if (container?.histogramData) updateExposureCounts(container.histogramData);
    refreshOpenCovariateDistributions();
    scheduleAutosaveSettings();
  }
  if (event.target.closest("[data-analysis-form]")) {
    if (event.target.matches("[data-analysis-method]")) {
      updateMethodEquation();
    }
    scheduleAutosaveSettings();
  }
  if (event.target.matches("[data-entity-search-input], [data-node-id-input]")) {
    scheduleEntitySuggestionRefresh(event.target.value.trim());
  }
  if (event.target.matches('[data-sidebar-field="node_id"]')) {
    validateSidebarSensor();
  }
});

document.addEventListener("input", (event) => {
  syncAnalysisDayInputs(event.target);
  if (event.target.matches("[data-threshold-input]")) {
    updateThresholdHours();
    const container = document.querySelector("[data-histogram]");
    if (container?.histogramData) updateExposureCounts(container.histogramData);
  }
  if (event.target.closest("[data-analysis-form]")) {
    scheduleAutosaveSettings();
  }
  if (event.target.matches('[data-sidebar-field="node_id"]')) {
    validateSidebarSensor();
  }
});

document.addEventListener("submit", async (event) => {
  const dagForm = event.target.closest("[data-dag-form]");
  if (dagForm) {
    const cycle = updateDagCycleWarning();
    if (cycle.length) {
      event.preventDefault();
      document.querySelector("[data-dag-cycle-warning]")?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    return;
  }

  const form = event.target.closest("[data-analysis-form]");
  if (!form) return;
  event.preventDefault();

  if (form.dataset.dagSyncedForSubmit !== "1") {
    updateRunAnalysisButtonState();
    const button = form.querySelector("[data-run-analysis-button]");
    const previousContent = button?.innerHTML;
    if (button) {
      button.disabled = true;
      button.innerHTML = '<span class="button-spinner" aria-hidden="true"></span> Saving DAG';
    }
    const saved = await autosaveDagNow(false);
    if (!saved) {
      if (button) {
        button.innerHTML = previousContent || "Run Analysis";
        delete button.dataset.analysisRunning;
      }
      updateRunAnalysisButtonState();
      document.querySelector("[data-dag-cycle-warning]")?.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    form.dataset.dagSyncedForSubmit = "1";
    await runAnalysisInPage(form);
    return;
  }

  await runAnalysisInPage(form);
});

function startAnalysisProgress(output) {
  window.clearInterval(analysisProgressTimer);
  const progress = output.querySelector("[data-analysis-progress]");
  if (!progress) return;
  const tabs = Array.from(progress.querySelectorAll("[data-progress-step]"));
  const message = progress.querySelector("[data-analysis-progress-message]");
  let activeIndex = 0;

  const render = () => {
    tabs.forEach((tab, index) => {
      tab.classList.toggle("active", index === activeIndex);
      tab.classList.toggle("complete", index < activeIndex);
    });
    if (message) {
      message.textContent = ANALYSIS_PROGRESS_STEPS[activeIndex] || "Finishing analysis.";
    }
  };

  render();
  analysisProgressTimer = window.setInterval(() => {
    activeIndex = Math.min(activeIndex + 1, tabs.length - 1);
    render();
    if (activeIndex === tabs.length - 1) {
      window.clearInterval(analysisProgressTimer);
    }
  }, 1100);
}

function restoreAnalysisScroll() {
  if (sessionStorage.getItem("analysisSubmitted") !== "1") return;

  const savedY = Number(sessionStorage.getItem("analysisScrollY"));
  sessionStorage.removeItem("analysisSubmitted");
  sessionStorage.removeItem("analysisScrollY");

  if (Number.isFinite(savedY)) {
    requestAnimationFrame(() => {
      window.scrollTo({ top: savedY, behavior: "auto" });
    });
  }
}

function fitMethodEquations(root = document) {
  const scope = root instanceof Element || root instanceof Document ? root : document;
  scope.querySelectorAll("[data-equation-box]").forEach((box) => {
    const scroll = box.querySelector("[data-equation-scroll]");
    const scale = box.querySelector("[data-equation-scale]");
    if (!scroll || !scale) return;
    scale.style.fontSize = "1em";
    scroll.classList.remove("is-scrollable");
    const available = Math.max(1, scroll.clientWidth - 2);
    const naturalWidth = Math.max(scale.scrollWidth, scale.getBoundingClientRect().width);
    if (naturalWidth <= available) return;
    const nextScale = Math.max(0.72, Math.min(1, available / naturalWidth));
    scale.style.fontSize = `${nextScale.toFixed(3)}em`;
    requestAnimationFrame(() => {
      scroll.classList.toggle("is-scrollable", scale.scrollWidth > scroll.clientWidth + 2);
    });
  });
}

function ensureMathJaxLoaded() {
  if (window.MathJax?.typesetPromise) return Promise.resolve(window.MathJax);
  if (mathJaxLoadPromise) return mathJaxLoadPromise;
  mathJaxLoadPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    const timeout = window.setTimeout(() => {
      script.remove();
      mathJaxLoadPromise = null;
      reject(new Error("MathJax load timed out."));
    }, 3500);
    script.src = MATHJAX_SRC;
    script.async = true;
    script.dataset.mathjaxLoader = "1";
    script.onload = () => {
      window.clearTimeout(timeout);
      resolve(window.MathJax);
    };
    script.onerror = () => {
      window.clearTimeout(timeout);
      mathJaxLoadPromise = null;
      reject(new Error("MathJax failed to load."));
    };
    document.head.appendChild(script);
  });
  return mathJaxLoadPromise;
}

function typesetMath(root = document) {
  const target = root instanceof Element || root instanceof Document ? root : document;
  ensureMathJaxLoaded().then(() => {
    window.MathJax.typesetPromise([target])
      .then(() => fitMethodEquations(target))
      .catch(() => fitMethodEquations(target));
  }).catch(() => {
    fitMethodEquations(target);
  });
}

function clampWorkbenchWidth(width, workbench) {
  const bounds = workbench.getBoundingClientRect();
  const minimum = 320;
  const minimumOutputWidth = 420;
  const maximum = Math.max(minimum, bounds.width - 12 - minimumOutputWidth);
  return Math.min(maximum, Math.max(minimum, width));
}

function updateWorkbenchOutputMode(workbench, width) {
  const bounds = workbench.getBoundingClientRect();
  const availableOutputWidth = bounds.width - width - 20;
  workbench.classList.toggle("is-output-condensed", availableOutputWidth < 560);
}

function setWorkbenchWidth(workbench, width) {
  const clamped = clampWorkbenchWidth(width, workbench);
  workbench.style.setProperty("--workbench-left-width", `${clamped}px`);
  localStorage.setItem("workbenchLeftWidth", String(Math.round(clamped)));
  updateWorkbenchOutputMode(workbench, clamped);
}

function updateAnalysisToolsToggleState(toggle, expanded) {
  toggle.setAttribute("aria-expanded", String(expanded));
  toggle.setAttribute(
    "aria-label",
    expanded ? "Collapse causal analysis" : "Expand causal analysis"
  );
  toggle.title = expanded ? "Collapse causal analysis" : "Expand causal analysis";
}

function setAnalysisToolsExpanded(expanded) {
  const section = document.querySelector("[data-analysis-section]");
  const body = document.querySelector("[data-analysis-tools-body]");
  const toggle = document.querySelector("[data-analysis-tools-toggle]");
  if (!section || !body || !toggle) return;

  updateAnalysisToolsToggleState(toggle, expanded);
  body.hidden = false;
  section.classList.toggle("is-analysis-collapsed", !expanded);

  if (!expanded) return;
  requestAnimationFrame(() => {
    const workbench = document.querySelector("[data-analysis-workbench]");
    if (!workbench) return;
    updateWorkbenchOutputMode(
      workbench,
      workbench.querySelector(".workbench-controls").getBoundingClientRect().width
    );
  });
}

function initializeAnalysisToolsToggle() {
  const toggle = document.querySelector("[data-analysis-tools-toggle]");
  const body = document.querySelector("[data-analysis-tools-body]");
  if (!toggle || !body) return;

  const expanded = toggle.getAttribute("aria-expanded") !== "false";
  updateAnalysisToolsToggleState(toggle, expanded);
  if (!expanded) {
    document.querySelector("[data-analysis-section]")?.classList.add("is-analysis-collapsed");
  }
  body.hidden = false;

  toggle.addEventListener("click", () => {
    setAnalysisToolsExpanded(toggle.getAttribute("aria-expanded") === "false");
  });
}

function initializeWorkbenchResizer() {
  const workbench = document.querySelector("[data-analysis-workbench]");
  const resizer = document.querySelector("[data-workbench-resizer]");
  if (!workbench || !resizer) return;

  const savedWidth = Number(localStorage.getItem("workbenchLeftWidth"));
  if (Number.isFinite(savedWidth)) {
    setWorkbenchWidth(workbench, savedWidth);
  } else {
    updateWorkbenchOutputMode(
      workbench,
      workbench.querySelector(".workbench-controls").getBoundingClientRect().width
    );
  }

  let startX = 0;
  let startWidth = 0;

  function stopResize() {
    workbench.classList.remove("is-resizing");
    document.removeEventListener("pointermove", resize);
    document.removeEventListener("pointerup", stopResize);
  }

  function resize(event) {
    setWorkbenchWidth(workbench, startWidth + event.clientX - startX);
  }

  resizer.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    startX = event.clientX;
    startWidth = workbench.querySelector(".workbench-controls").getBoundingClientRect().width;
    workbench.classList.add("is-resizing");
    resizer.setPointerCapture(event.pointerId);
    document.addEventListener("pointermove", resize);
    document.addEventListener("pointerup", stopResize);
  });

  resizer.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const currentWidth = workbench.querySelector(".workbench-controls").getBoundingClientRect().width;
    if (event.key === "Home") {
      setWorkbenchWidth(workbench, 320);
    } else if (event.key === "End") {
      setWorkbenchWidth(workbench, workbench.getBoundingClientRect().width - 432);
    } else {
      setWorkbenchWidth(workbench, currentWidth + (event.key === "ArrowRight" ? 32 : -32));
    }
  });

  window.addEventListener("resize", () => {
    const currentWidth = workbench.querySelector(".workbench-controls").getBoundingClientRect().width;
    setWorkbenchWidth(workbench, currentWidth);
  });
}

function setDagBuilderExpanded(expanded) {
  const toggle = document.querySelector("[data-toggle-section='dag-section-body']");
  const section = document.getElementById("dag-section-body");
  if (!toggle || !section) return;
  toggle.setAttribute("aria-expanded", String(expanded));
  toggle.setAttribute("aria-label", expanded ? "Collapse causal DAG" : "Expand causal DAG");
  toggle.title = expanded ? "Collapse causal DAG" : "Expand causal DAG";
  section.hidden = false;
  section.closest(".workflow-dag-section")?.classList.toggle("is-dag-collapsed", !expanded);
  if (expanded) {
    requestAnimationFrame(() => {
      updateDagEdges();
      requestAnimationFrame(updateDagEdges);
    });
  }
}

function findDagNodeForGuide({ role, text }) {
  const nodes = Array.from(document.querySelectorAll("[data-dag-node]"));
  return nodes.find((node) => {
    const nodeRole = node.querySelector("[data-node-role]")?.value || "";
    const label = humanReadableNodeLabel(node).toLowerCase();
    const id = dagNodeId(node).toLowerCase();
    return (!role || nodeRole === role) && (!text || label.includes(text) || id.includes(text));
  }) || nodes.find((node) => !role || node.querySelector("[data-node-role]")?.value === role) || nodes[0];
}

function prepareDagNodeGuide(options = {}, sidebarStep = "definition") {
  setDagBuilderExpanded(true);
  const node = findDagNodeForGuide(options);
  if (node) {
    setSelectedDagNode(node);
    setSidebarStep(sidebarStep);
    if (sidebarStep === "advanced") {
      loadDagNodeHistogram(node);
    }
  }
}

function submitAnalysisFromGuide() {
  const form = document.querySelector("[data-analysis-form]");
  if (!form) return;
  onboardingResumeAfterAnalysis = true;
  if (typeof setAnalysisToolsExpanded === "function") setAnalysisToolsExpanded(true);
  form.requestSubmit();
}

const ONBOARDING_STEPS = [
  {
    selector: "[data-toggle-section='dag-section-body']",
    title: "Open Step 1",
    copy: "Start by opening the DAG builder. This is where you review variables and causal arrows before running an analysis.",
  },
  {
    selector: ".dag-node.exposure",
    title: "Select sleep exposure",
    copy: "Sleep is the exposure in this time-varying example. Click it to inspect the sensor mapping, role, aggregation, and prior-day timing.",
    prepare: () => {
      prepareDagNodeGuide({ role: "exposure", text: "sleep" }, "definition");
    },
  },
  {
    selector: "[data-sidebar-panel='definition']",
    title: "Confirm the variable definition",
    copy: "Use this panel to check the sensor mapping, role, daily aggregate, time window, and lag. Exposure and covariate nodes usually use prior-day timing.",
    prepare: () => {
      prepareDagNodeGuide({ role: "exposure", text: "sleep" }, "definition");
    },
  },
  {
    selector: "[data-sidebar-panel='preparation']",
    title: "Set the sleep threshold",
    copy: "For sleep exposure, use Binary from threshold with more-than 420 minutes, so exposed days mean roughly seven or more hours of sleep.",
    prepare: () => {
      prepareDagNodeGuide({ role: "exposure", text: "sleep" }, "preparation");
    },
  },
  {
    selector: "[data-sidebar-histogram]",
    title: "Show the sleep histogram",
    copy: "This histogram shows the daily sleep distribution. Use it to choose a threshold with enough days on both sides.",
    prepare: () => {
      prepareDagNodeGuide({ role: "exposure", text: "sleep" }, "advanced");
    },
  },
  {
    selector: ".dag-node.covariate",
    title: "Review steps or other covariates",
    copy: "Select a covariate such as steps to decide whether it should stay continuous or become a thresholded adjustment variable.",
    prepare: () => {
      prepareDagNodeGuide({ role: "covariate", text: "step" }, "definition");
    },
  },
  {
    selector: "[data-sidebar-panel='preparation']",
    title: "Set the steps threshold",
    copy: "Steps start as high activity versus lower activity using more-than 8000 steps, a readable cutoff for daily movement.",
    prepare: () => {
      prepareDagNodeGuide({ role: "covariate", text: "step" }, "preparation");
    },
  },
  {
    selector: "[data-sidebar-histogram]",
    title: "Show the steps histogram",
    copy: "The steps histogram helps decide whether activity should stay continuous or be thresholded into low versus high activity days.",
    prepare: () => {
      prepareDagNodeGuide({ role: "covariate", text: "step" }, "advanced");
    },
  },
  {
    selector: ".dag-node.outcome",
    title: "Select the mood outcome",
    copy: "Mood is the outcome the analysis tries to explain. Confirm the mapped mood sensor and make sure outcome timing is measured at t.",
    prepare: () => {
      prepareDagNodeGuide({ role: "outcome", text: "mood" }, "definition");
    },
  },
  {
    selector: "[data-sidebar-panel='preparation']",
    title: "Set the mood threshold",
    copy: "Mood starts as a low-mood outcome using less-than 5.7 on the 0 to 10 scale, which gives the demo data enough days on both sides.",
    prepare: () => {
      prepareDagNodeGuide({ role: "outcome", text: "mood" }, "preparation");
    },
  },
  {
    selector: "[data-sidebar-histogram]",
    title: "Show the mood histogram",
    copy: "The mood histogram shows the outcome distribution and where the low-mood threshold sits before the model runs.",
    prepare: () => {
      prepareDagNodeGuide({ role: "outcome", text: "mood" }, "advanced");
    },
  },
  {
    selector: "[data-analysis-tools-toggle]",
    title: "Open Step 2",
    copy: "After the DAG structure looks right, open Causal Analysis to choose an estimation method and review model settings.",
  },
  {
    selector: "[data-analysis-method]",
    title: "Choose the causal method",
    copy: "Pick the estimator for the current DAG. The method equation updates here so the analysis setup stays connected to the causal assumptions.",
    prepare: () => {
      if (typeof setAnalysisToolsExpanded === "function") setAnalysisToolsExpanded(true);
    },
  },
  {
    selector: "[data-run-analysis-button]",
    title: "Run the analysis",
    copy: "Click Run Analysis to use the current DAG, thresholds, timing, and selected method. The demo data will produce results for this mood example.",
    prepare: () => {
      if (typeof setAnalysisToolsExpanded === "function") setAnalysisToolsExpanded(true);
    },
    nextLabel: "Run Analysis",
    action: submitAnalysisFromGuide,
  },
];

let onboardingIndex = 0;
let onboardingActive = false;
let onboardingCurrentTarget = null;
let onboardingRenderTimer;
let onboardingPositionFrame;
let onboardingResumeAfterAnalysis = false;

async function runAnalysisInPage(form) {
  const button = form.querySelector("[data-run-analysis-button]");
  const loading = form.querySelector("[data-analysis-loading]");
  const output = document.querySelector("[data-analysis-results]");
  const previousContent = button?.innerHTML;

  window.clearInterval(analysisProgressTimer);
  if (button) {
    button.dataset.analysisRunning = "1";
    button.disabled = true;
    button.innerHTML = '<span class="button-spinner" aria-hidden="true"></span> Running analysis';
  }
  if (loading) {
    loading.classList.add("active");
  }
  if (output) {
    output.classList.add("is-running");
    startAnalysisProgress(output);
  }

  try {
    const response = await fetch(form.action, {
      method: form.method || "POST",
      body: new FormData(form),
      headers: { "X-Requested-With": "fetch" },
    });
    const html = await response.text();
    if (!response.ok) {
      throw new Error(`Analysis request failed with ${response.status}`);
    }

    const parsed = new DOMParser().parseFromString(html, "text/html");
    const nextOutput = parsed.querySelector("[data-analysis-results]");
    if (!nextOutput || !output) {
      throw new Error("Analysis response did not include a results panel.");
    }

    window.clearInterval(analysisProgressTimer);
    output.replaceWith(document.importNode(nextOutput, true));
    const updatedOutput = document.querySelector("[data-analysis-results]");
    updatedOutput?.classList.remove("is-running");
    fitMethodEquations(updatedOutput || document);
    typesetMath();

    if (onboardingResumeAfterAnalysis) {
      onboardingResumeAfterAnalysis = false;
      finishOnboarding();
    }
  } catch (error) {
    window.clearInterval(analysisProgressTimer);
    if (output) {
      output.classList.remove("is-running");
      output.innerHTML = `
        <section class="panel result-card">
          <h2>Analysis could not run</h2>
          <p>${escapeHtml(error.message || "Unable to run the analysis without refreshing.")}</p>
        </section>
      `;
    }
  } finally {
    delete form.dataset.dagSyncedForSubmit;
    if (button) {
      delete button.dataset.analysisRunning;
      button.disabled = false;
      button.innerHTML = previousContent || "Run Analysis";
    }
    if (loading) {
      loading.classList.remove("active");
    }
    updateRunAnalysisButtonState();
  }
}

function positionOnboardingTarget(target) {
  const overlay = document.querySelector("[data-onboarding-overlay]");
  const spotlight = document.querySelector("[data-onboarding-spotlight]");
  const card = document.querySelector("[data-onboarding-card]");
  const step = ONBOARDING_STEPS[onboardingIndex];
  if (!overlay || !spotlight || !card || !target || !step) return;

  const rect = target.getBoundingClientRect();
  const pad = 12;
  const top = Math.max(12, rect.top - pad);
  const left = Math.max(12, rect.left - pad);
  const width = Math.min(window.innerWidth - left - 12, rect.width + pad * 2);
  const height = Math.min(window.innerHeight - top - 12, rect.height + pad * 2);
  spotlight.style.top = `${top}px`;
  spotlight.style.left = `${left}px`;
  spotlight.style.width = `${width}px`;
  spotlight.style.height = `${height}px`;

  const cardWidth = Math.min(360, window.innerWidth - 32);
  const prefersRight = left + width + 24 + cardWidth < window.innerWidth;
  const cardLeft = prefersRight ? left + width + 24 : Math.max(16, Math.min(window.innerWidth - cardWidth - 16, left));
  const cardTop = prefersRight
    ? Math.max(16, Math.min(window.innerHeight - 260, top))
    : Math.min(window.innerHeight - 260, top + height + 18);
  card.style.left = `${cardLeft}px`;
  card.style.top = `${Math.max(16, cardTop)}px`;
}

function scheduleOnboardingPosition() {
  if (!onboardingActive || !onboardingCurrentTarget) return;
  window.cancelAnimationFrame(onboardingPositionFrame);
  onboardingPositionFrame = window.requestAnimationFrame(() => {
    positionOnboardingTarget(onboardingCurrentTarget);
  });
}

function renderOnboardingStep() {
  const overlay = document.querySelector("[data-onboarding-overlay]");
  const title = document.querySelector("[data-onboarding-title]");
  const copy = document.querySelector("[data-onboarding-copy]");
  const count = document.querySelector("[data-onboarding-count]");
  const next = document.querySelector("[data-onboarding-next]");
  const step = ONBOARDING_STEPS[onboardingIndex];
  if (!overlay || !title || !copy || !count || !next || !step) return;

  window.clearTimeout(onboardingRenderTimer);
  onboardingCurrentTarget = null;
  title.textContent = step.title;
  copy.textContent = step.copy;
  count.textContent = `Guide ${onboardingIndex + 1} of ${ONBOARDING_STEPS.length}`;
  next.textContent = step.nextLabel || (onboardingIndex === ONBOARDING_STEPS.length - 1 ? "Done" : "Next");
  step.prepare?.();

  onboardingRenderTimer = window.setTimeout(() => {
    const target = document.querySelector(step.selector);
    if (!target) {
      finishOnboarding();
      return;
    }
    onboardingCurrentTarget = target;
    target.scrollIntoView({ block: "center", inline: "center", behavior: "smooth" });
    positionOnboardingTarget(target);
    onboardingRenderTimer = window.setTimeout(() => {
      positionOnboardingTarget(target);
    }, 420);
  }, 90);
}

function startOnboarding() {
  if (!document.querySelector("[data-onboarding-overlay]")) return;
  onboardingIndex = 0;
  onboardingActive = true;
  document.querySelector("[data-onboarding-overlay]").hidden = false;
  document.body.classList.add("is-onboarding-active");
  renderOnboardingStep();
}

function finishOnboarding() {
  const overlay = document.querySelector("[data-onboarding-overlay]");
  onboardingActive = false;
  onboardingCurrentTarget = null;
  window.clearTimeout(onboardingRenderTimer);
  window.cancelAnimationFrame(onboardingPositionFrame);
  if (overlay) overlay.hidden = true;
  document.body.classList.remove("is-onboarding-active");
}

function initializeOnboarding() {
  const start = document.querySelector("[data-onboarding-start]");
  const next = document.querySelector("[data-onboarding-next]");
  const skipButtons = document.querySelectorAll("[data-onboarding-skip]");
  if (!start || !next) return;

  start.addEventListener("click", startOnboarding);
  next.addEventListener("click", () => {
    const step = ONBOARDING_STEPS[onboardingIndex];
    if (step?.action) {
      step.action();
      return;
    }
    if (onboardingIndex >= ONBOARDING_STEPS.length - 1) {
      finishOnboarding();
      return;
    }
    onboardingIndex += 1;
    renderOnboardingStep();
  });
  skipButtons.forEach((button) => button.addEventListener("click", finishOnboarding));
  window.addEventListener("resize", scheduleOnboardingPosition);
  window.addEventListener("scroll", scheduleOnboardingPosition, { passive: true });

  if (!window.personalWebsiteDisableOnboarding) {
    window.setTimeout(startOnboarding, 750);
  }
}

document.addEventListener("DOMContentLoaded", loadHistogram);
document.addEventListener("DOMContentLoaded", updateThresholdHours);
document.addEventListener("DOMContentLoaded", restoreAnalysisScroll);
document.addEventListener("DOMContentLoaded", typesetMath);
document.addEventListener("DOMContentLoaded", updateMethodEquation);
document.addEventListener("DOMContentLoaded", initializeAnalysisToolsToggle);
document.addEventListener("DOMContentLoaded", initializeWorkbenchResizer);
document.addEventListener("DOMContentLoaded", initializeDagEditor);
document.addEventListener("DOMContentLoaded", initializeOnboarding);
window.addEventListener("resize", () => fitMethodEquations(document));
window.addEventListener("beforeunload", saveSettingsBeforeUnload);
window.addEventListener("beforeunload", saveDagBeforeUnload);

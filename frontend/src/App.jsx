
import { useEffect, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:5000";

const EMPTY_SIMULATION = {
  steps_last_hour: "",
  sleep_hours_last_night: "",
  carbs_previous_2h_g: "",
};

function formatLabel(value) {
  if (value === null || value === undefined || value === "") {
    return "Not available";
  }

  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function displayValue(value, suffix = "") {
  if (value === null || value === undefined || value === "") {
    return "Not available";
  }

  return `${value}${suffix}`;
}

function getArray(data, keys = []) {
  if (Array.isArray(data)) return data;

  for (const key of keys) {
    if (Array.isArray(data?.[key])) return data[key];
  }

  return [];
}

function getNumber(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function getPatientId(patient) {
  return patient?.patient_id ?? patient?.id ?? patient?.patientId ?? "";
}

function getGlucoseValue(prediction) {
  return getNumber(prediction?.predicted_glucose_2h_mg_dl);
}

function SectionHeading({ eyebrow, title, tag }) {
  return (
    <div className="section-heading">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h2>{title}</h2>
      </div>
      {tag && <span className="demo-tag">{tag}</span>}
    </div>
  );
}

function MetricCard({ label, value, description }) {
  return (
    <article className="metric-card">
      <p>{label}</p>
      <h2>{value}</h2>
      <span>{description}</span>
    </article>
  );
}

function ProfileRow({ label, value }) {
  return (
    <div className="profile-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function GlucoseChart({ history, prediction }) {
  const width = 760;
  const height = 290;
  const left = 58;
  const right = 25;
  const top = 30;
  const bottom = 48;

  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;

  const validHistory = history
    .map((item) => ({
      ...item,
      value: getNumber(item.glucose_mg_dl),
      time: new Date(item.timestamp),
    }))
    .filter(
      (item) =>
        item.value !== null &&
        !Number.isNaN(item.time.getTime())
    );

  const predictedValue = getGlucoseValue(prediction);

  if (validHistory.length === 0 && predictedValue === null) {
    return (
      <div className="chart-empty">
        No glucose history or forecast is available yet.
      </div>
    );
  }

  const allValues = [
    ...validHistory.map((item) => item.value),
    ...(predictedValue === null ? [] : [predictedValue]),
  ];

  const minimum = Math.min(...allValues, 70);
  const maximum = Math.max(...allValues, 180);
  const yMin = Math.floor((minimum - 10) / 10) * 10;
  const yMax = Math.ceil((maximum + 10) / 10) * 10;
  const yRange = Math.max(yMax - yMin, 20);

  // Keep the forecast separate from the historical readings.
  const historyWidth = plotWidth * 0.78;

  const xFor = (index, count) => {
    if (count <= 1) return left;
    return left + (index / (count - 1)) * historyWidth;
  };

  const yFor = (value) =>
    top + ((yMax - value) / yRange) * plotHeight;

  const historyPoints = validHistory.map((item, index) => ({
    ...item,
    x: xFor(index, validHistory.length),
    y: yFor(item.value),
  }));

  const forecastX = left + plotWidth;
  const forecastY =
    predictedValue === null ? null : yFor(predictedValue);

  const historyPath = historyPoints
    .map(
      (point, index) =>
        `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`
    )
    .join(" ");

  return (
    <div className="chart-wrapper">
      <div className="chart-legend">
        <span>
          <i className="legend-dot recorded" />
          Recorded glucose
        </span>
        <span>
          <i className="legend-dot predicted" />
          Two-hour prediction
        </span>
      </div>

      <div className="chart-scroll">
        <svg
          className="glucose-chart"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="Historical glucose readings and two-hour forecast"
        >
          {Array.from({ length: 6 }, (_, index) => {
            const value = yMin + (yRange * index) / 5;
            const y = yFor(value);

            return (
              <g key={index}>
                <line
                  x1={left}
                  y1={y}
                  x2={width - right}
                  y2={y}
                  stroke="#e5e9e6"
                  strokeDasharray="4 5"
                />
                <text
                  x={left - 10}
                  y={y + 4}
                  fontSize="11"
                  fill="#68766e"
                  textAnchor="end"
                >
                  {Math.round(value)}
                </text>
              </g>
            );
          })}

          <text
            x="15"
            y={height / 2}
            fontSize="11"
            fill="#68766e"
            textAnchor="middle"
            transform={`rotate(-90 15 ${height / 2})`}
          >
            Glucose (mg/dL)
          </text>

          {historyPath && (
            <path
              d={historyPath}
              fill="none"
              stroke="#17815c"
              strokeWidth="2.8"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          )}

          {historyPoints.map((point, index) => (
            <circle
              key={`${point.time.toISOString()}-${index}`}
              cx={point.x}
              cy={point.y}
              r="3.8"
              fill="#17815c"
              stroke="#ffffff"
              strokeWidth="1.5"
            >
              <title>
                {point.time.toLocaleString()}: {point.value} mg/dL
              </title>
            </circle>
          ))}

          {historyPoints.length > 0 && (
            <>
              <text
                x={left}
                y={height - 13}
                fontSize="11"
                fill="#68766e"
                textAnchor="start"
              >
                {historyPoints[0].time.toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </text>

              <text
                x={left + historyWidth}
                y={height - 13}
                fontSize="11"
                fill="#68766e"
                textAnchor="middle"
              >
                Latest
              </text>
            </>
          )}

          {predictedValue !== null && (
            <>
              <line
                x1={forecastX}
                y1={top}
                x2={forecastX}
                y2={height - bottom}
                stroke="#d58b28"
                strokeDasharray="5 5"
                strokeWidth="1.5"
              />

              <circle
                cx={forecastX}
                cy={forecastY}
                r="7"
                fill="#d58b28"
                stroke="#ffffff"
                strokeWidth="2"
              >
                <title>
                  Predicted in two hours:{" "}
                  {predictedValue.toFixed(1)} mg/dL
                </title>
              </circle>

              <text
                x={forecastX - 5}
                y={Math.max(top + 12, forecastY - 13)}
                fontSize="12"
                fontWeight="600"
                fill="#a9650d"
                textAnchor="end"
              >
                {predictedValue.toFixed(1)}
              </text>

              <text
                x={forecastX - 5}
                y={height - bottom + 22}
                fontSize="11"
                fill="#a9650d"
                textAnchor="end"
              >
                +2 hours
              </text>
            </>
          )}
        </svg>
      </div>

      <p className="chart-note">
        Historical points are recorded synthetic sensor values. The
        forecast is a separate model estimate, not an observed reading.
      </p>
    </div>
  );
}

function App() {
  const [currentPage, setCurrentPage] = useState("dashboard");

  const [patients, setPatients] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [patient, setPatient] = useState(null);
  const [sensor, setSensor] = useState(null);
  const [features, setFeatures] = useState(null);
  const [history, setHistory] = useState([]);

  const [prediction, setPrediction] = useState(null);
  const [simulationInputs, setSimulationInputs] =
    useState(EMPTY_SIMULATION);
  const [simulationPrediction, setSimulationPrediction] =
    useState(null);

  const [loadingPatients, setLoadingPatients] = useState(true);
  const [loadingPatient, setLoadingPatient] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [simulating, setSimulating] = useState(false);
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState("Checking connection");

  // Load the patient directory once.
  useEffect(() => {
    let cancelled = false;

    async function loadPatients() {
      try {
        const response = await fetch(`${API}/patients`);

        if (!response.ok) {
          throw new Error("Could not load the patient directory.");
        }

        const data = await response.json();
        const list = getArray(data, [
          "patients",
          "data",
          "results",
        ]);

        if (cancelled) return;

        setPatients(list);
        setApiStatus("Connected");

        if (list.length > 0) {
          const firstId = getPatientId(list[0]);
          if (firstId !== "") setSelectedId(String(firstId));
        }
      } catch (err) {
        if (!cancelled) {
          setApiStatus("Disconnected");
          setError(
            `${err.message} Check that Flask is running at ${API}.`
          );
        }
      } finally {
        if (!cancelled) setLoadingPatients(false);
      }
    }

    loadPatients();

    return () => {
      cancelled = true;
    };
  }, []);

  // Load the selected patient's profile, features, and history.
  useEffect(() => {
    if (!selectedId) return undefined;

    let cancelled = false;

    async function loadPatientData() {
      setLoadingPatient(true);
      setError("");
      setPatient(null);
      setSensor(null);
      setFeatures(null);
      setHistory([]);
      setPrediction(null);
      setSimulationPrediction(null);

      try {
        const encodedId = encodeURIComponent(selectedId);

        const responses = await Promise.all([
          fetch(`${API}/patients/${encodedId}`),
          fetch(`${API}/patients/${encodedId}/features`),
          fetch(`${API}/patients/${encodedId}/history`),
        ]);

        if (!responses[0].ok) {
          throw new Error("Could not load this patient's profile.");
        }

        const profileData = await responses[0].json();
        const profile = profileData.patient ?? profileData;

        let loadedFeatures = null;
        let loadedHistory = [];

        if (responses[1].ok) {
          const data = await responses[1].json();
          loadedFeatures =
            data.features ?? data.model_features ?? null;
        }

        if (responses[2].ok) {
          const data = await responses[2].json();
          loadedHistory = getArray(data, [
            "history",
            "data",
            "readings",
          ]);
        }

        if (cancelled) return;

        setPatient(profile);
        setSensor(
          profileData.latest_sensor ??
            profileData.sensor ??
            profileData.latestSensor ??
            null
        );
        setFeatures(loadedFeatures);
        setHistory(loadedHistory);
        setApiStatus("Connected");
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
        }
      } finally {
        if (!cancelled) setLoadingPatient(false);
      }
    }

    loadPatientData();

    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  // Fill the simulation form with the selected patient's baseline values.
  useEffect(() => {
    if (!features) {
      setSimulationInputs(EMPTY_SIMULATION);
      setSimulationPrediction(null);
      return;
    }

    setSimulationInputs({
      steps_last_hour: features.steps_last_hour ?? "",
      sleep_hours_last_night:
        features.sleep_hours_last_night ?? "",
      carbs_previous_2h_g: features.carbs_previous_2h_g ?? "",
    });

    setSimulationPrediction(null);
  }, [features, selectedId]);

  async function postPrediction(inputFeatures, purpose) {
    const response = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(inputFeatures),
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(
        data.error || `${purpose} request failed.`
      );
    }

    return data;
  }

  async function generateForecast() {
    setError("");
    setPrediction(null);

    if (!features || Object.keys(features).length === 0) {
      setError("Prepared model features are unavailable.");
      return;
    }

    setPredicting(true);

    try {
      const data = await postPrediction(features, "Forecast");
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setPredicting(false);
    }
  }

  async function runWhatIfSimulation() {
    setError("");
    setSimulationPrediction(null);

    if (!features) {
      setError("Patient model features are unavailable.");
      return;
    }

    const simulatedFeatures = { ...features };

    for (const key of [
      "steps_last_hour",
      "sleep_hours_last_night",
      "carbs_previous_2h_g",
    ]) {
      const rawValue = simulationInputs[key];

      if (rawValue === "" || rawValue === null) {
        setError(
          `Please enter a value for ${key.replaceAll("_", " ")}.`
        );
        return;
      }

      const numericValue = Number(rawValue);

      if (!Number.isFinite(numericValue) || numericValue < 0) {
        setError("Simulation values must be valid non-negative numbers.");
        return;
      }

      simulatedFeatures[key] = numericValue;
    }

    if (simulatedFeatures.sleep_hours_last_night > 24) {
      setError("Sleep hours cannot exceed 24.");
      return;
    }

    setSimulating(true);

    try {
      const data = await postPrediction(
        simulatedFeatures,
        "Simulation"
      );
      setSimulationPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setSimulating(false);
    }
  }

  const patientId = patient?.patient_id ?? selectedId;
  const glucose = sensor?.glucose_mg_dl ?? null;
  const heartRate = sensor?.heart_rate_bpm ?? null;
  const hrv = sensor?.hrv_ms ?? null;
  const spo2 = sensor?.spo2_percent ?? null;

  const currentGlucose = getNumber(glucose);
  const predictedGlucose = getGlucoseValue(prediction);
  const simulatedGlucose = getGlucoseValue(simulationPrediction);

  const glucoseChange =
    currentGlucose !== null && predictedGlucose !== null
      ? predictedGlucose - currentGlucose
      : null;

  const simulationDifference =
    simulatedGlucose !== null && predictedGlucose !== null
      ? simulatedGlucose - predictedGlucose
      : null;

  const changeLabel =
    glucoseChange === null
      ? "Not available"
      : glucoseChange > 0
        ? "Increase predicted"
        : glucoseChange < 0
          ? "Decrease predicted"
          : "No change predicted";

  const formattedChange =
    glucoseChange === null
      ? "Not available"
      : `${glucoseChange > 0 ? "+" : ""}${glucoseChange.toFixed(1)} mg/dL`;

  const validHistory = history.filter(
    (item) => getNumber(item.glucose_mg_dl) !== null
  );

  const pageTitles = {
    dashboard: {
      title: "Patient Dashboard",
      subtitle: "An overview of your synthetic patient and model results.",
    },
    monitoring: {
      title: "Patient Monitoring",
      subtitle: "Review patient details and available sensor readings.",
    },
    forecast: {
      title: "Glucose Forecast",
      subtitle: "Explore the two-hour prediction and hypothetical scenarios.",
    },
    reports: {
      title: "Reports",
      subtitle: "Review model evaluation and print a synthetic patient report.",
    },
  };

  const pageInfo = pageTitles[currentPage];

  function updateSimulationField(field, value) {
    setSimulationInputs((previous) => ({
      ...previous,
      [field]: value,
    }));
    setSimulationPrediction(null);
  }

  function renderPatientProfile() {
    return (
      <section className="content-grid">
        <div className="profile-panel">
          <SectionHeading eyebrow="PATIENT RECORD" title="Patient Profile" />

          <ProfileRow label="Patient ID" value={displayValue(patientId)} />
          <ProfileRow
            label="Age"
            value={displayValue(patient?.age_years, " years")}
          />
          <ProfileRow label="Sex" value={formatLabel(patient?.sex)} />
          <ProfileRow label="BMI" value={displayValue(patient?.bmi)} />
          <ProfileRow
            label="Diabetes status"
            value={formatLabel(patient?.diabetes_status)}
          />
          <ProfileRow
            label="Baseline HbA1c"
            value={displayValue(patient?.hba1c_baseline_percent, "%")}
          />
          <ProfileRow
            label="Risk category"
            value={formatLabel(patient?.baseline_risk_category)}
          />
        </div>

        <div className="profile-panel">
          <SectionHeading eyebrow="SENSOR DATA" title="Latest Readings" />

          <ProfileRow
            label="Glucose"
            value={displayValue(glucose, " mg/dL")}
          />
          <ProfileRow
            label="Heart rate"
            value={displayValue(heartRate, " bpm")}
          />
          <ProfileRow
            label="Heart-rate variability"
            value={displayValue(hrv, " ms")}
          />
          <ProfileRow label="SpO₂" value={displayValue(spo2, "%")} />
          <ProfileRow
            label="Steps last hour"
            value={displayValue(sensor?.steps_last_hour)}
          />
          <ProfileRow
            label="Activity"
            value={formatLabel(sensor?.activity_intensity)}
          />
          <ProfileRow
            label="Sleep last night"
            value={displayValue(sensor?.sleep_hours_last_night, " hours")}
          />
          <ProfileRow
            label="Sensor quality"
            value={formatLabel(sensor?.sensor_quality)}
          />
          <ProfileRow
            label="Recorded at"
            value={
              sensor?.timestamp
                ? new Date(sensor.timestamp).toLocaleString()
                : "Not available"
            }
          />
        </div>
      </section>
    );
  }

  function renderForecast() {
    return (
      <>
        <section className="chart-panel">
          <SectionHeading
            eyebrow="GLUCOSE TRENDS"
            title="History and Forecast"
            tag="SYNTHETIC DATA"
          />

          <p className="panel-description">
            Historical sensor readings and the model's separate two-hour estimate.
          </p>

          <GlucoseChart history={history} prediction={prediction} />

          <button
            className="primary-button"
            onClick={generateForecast}
            disabled={predicting || loadingPatient || !features}
          >
            {predicting ? "Generating forecast..." : "Generate Forecast"}
          </button>

          {prediction && (
            <div className="result-box">
              <p className="result-label">Two-hour glucose forecast</p>

              <div className="forecast-comparison">
                <div className="forecast-value">
                  <span>Latest recorded glucose</span>
                  <h2>
                    {currentGlucose === null
                      ? "—"
                      : `${currentGlucose.toFixed(1)} mg/dL`}
                  </h2>
                </div>

                <div className="forecast-value predicted-value">
                  <span>Predicted in two hours</span>
                  <h2>
                    {predictedGlucose === null
                      ? "—"
                      : `${predictedGlucose.toFixed(1)} mg/dL`}
                  </h2>
                </div>
              </div>

              <div className="forecast-change">
                <span>Predicted difference</span>
                <h3>{formattedChange}</h3>
                <p>{changeLabel}</p>
              </div>

              <p>
                Demo category: {formatLabel(prediction.demo_category)}
              </p>
              {prediction.notice && <p>{prediction.notice}</p>}

              <p className="chart-note">
                The difference compares the latest available sensor reading
                with the forecast. It is not a guaranteed change.
              </p>
            </div>
          )}

          <p className="disclaimer">
            Synthetic data and model-generated estimates only. Do not use
            these results for diagnosis, treatment, or emergency decisions.
          </p>
        </section>

        <section className="simulation-panel">
          <SectionHeading
            eyebrow="DIGITAL TWIN"
            title="What-if Simulation"
            tag="HYPOTHETICAL"
          />

          <p className="panel-description">
            Change these inputs to explore how the model responds. The
            original patient record is not modified.
          </p>

          <div className="simulation-grid">
            <label className="simulation-field">
              Steps during the last hour
              <input
                type="number"
                min="0"
                step="1"
                value={simulationInputs.steps_last_hour}
                onChange={(event) =>
                  updateSimulationField("steps_last_hour", event.target.value)
                }
              />
            </label>

            <label className="simulation-field">
              Sleep hours last night
              <input
                type="number"
                min="0"
                max="24"
                step="0.5"
                value={simulationInputs.sleep_hours_last_night}
                onChange={(event) =>
                  updateSimulationField(
                    "sleep_hours_last_night",
                    event.target.value
                  )
                }
              />
            </label>

            <label className="simulation-field">
              Carbohydrates in previous 2 hours (g)
              <input
                type="number"
                min="0"
                step="1"
                value={simulationInputs.carbs_previous_2h_g}
                onChange={(event) =>
                  updateSimulationField(
                    "carbs_previous_2h_g",
                    event.target.value
                  )
                }
              />
            </label>
          </div>

          <button
            className="primary-button"
            onClick={runWhatIfSimulation}
            disabled={simulating || loadingPatient || !features}
          >
            {simulating ? "Running simulation..." : "Run What-if Simulation"}
          </button>

          {simulationPrediction && (
            <div className="simulation-result">
              <p className="result-label">Simulated two-hour forecast</p>
              <h3>
                {simulatedGlucose === null
                  ? "Prediction unavailable"
                  : `${simulatedGlucose.toFixed(1)} mg/dL`}
              </h3>

              {simulationDifference !== null && (
                <p>
                  Difference from original forecast:{" "}
                  {simulationDifference > 0 ? "+" : ""}
                  {simulationDifference.toFixed(1)} mg/dL
                </p>
              )}

              <p>
                Demo category:{" "}
                {formatLabel(simulationPrediction.demo_category)}
              </p>
            </div>
          )}

          <p className="disclaimer">
            This hypothetical simulation uses synthetic data. It does not
            establish cause and effect or provide medical advice.
          </p>
        </section>
      </>
    );
  }

  function renderPerformance() {
    return (
      <section className="performance-panel">
        <SectionHeading
          eyebrow="MODEL EVALUATION"
          title="Model Performance"
          tag="VALIDATION RESULTS"
        />

        <p className="panel-description">
          Previously calculated results on the synthetic test dataset.
        </p>

        <div className="performance-grid">
          <article className="performance-card">
            <p className="performance-label">MAE</p>
            <h3>11.129</h3>
            <span className="performance-unit">mg/dL</span>
            <p>
              Mean Absolute Error: the average absolute difference between
              predicted and actual glucose values.
            </p>
          </article>

          <article className="performance-card">
            <p className="performance-label">RMSE</p>
            <h3>15.996</h3>
            <span className="performance-unit">mg/dL</span>
            <p>
              Root Mean Squared Error: larger prediction errors affect
              this metric more strongly.
            </p>
          </article>

          <article className="performance-card">
            <p className="performance-label">R² Score</p>
            <h3>0.7878</h3>
            <span className="performance-unit">78.78% of variance</span>
            <p>
              The proportion of target variation explained by the model
              on that test dataset. This is not a clinical accuracy score.
            </p>
          </article>
        </div>

        <p className="performance-note">
          These are offline results on synthetic data. Errors were higher
          for low and high glucose readings. This model is a research
          demonstration only and must not be used for clinical decisions.
        </p>
      </section>
    );
  }

  function renderReport() {
    return (
      <section className="report-panel" id="patient-report">
        <div className="section-heading">
          <div>
            <p className="eyebrow">PATIENT SUMMARY</p>
            <h2>Patient Report</h2>
          </div>

          <button
            className="primary-button no-print"
            onClick={() => window.print()}
            disabled={!patient}
          >
            Print / Save as PDF
          </button>
        </div>

        <p className="panel-description">
          Synthetic patient summary for demonstration purposes.
        </p>

        <div className="report-subsection">
          <h3>Patient Profile</h3>
          <div className="report-grid">
            <div>
              <span>Patient ID</span>
              <strong>{displayValue(patientId)}</strong>
            </div>
            <div>
              <span>Age</span>
              <strong>{displayValue(patient?.age_years, " years")}</strong>
            </div>
            <div>
              <span>Sex</span>
              <strong>{formatLabel(patient?.sex)}</strong>
            </div>
            <div>
              <span>BMI</span>
              <strong>{displayValue(patient?.bmi)}</strong>
            </div>
            <div>
              <span>Diabetes status</span>
              <strong>{formatLabel(patient?.diabetes_status)}</strong>
            </div>
            <div>
              <span>Baseline HbA1c</span>
              <strong>
                {displayValue(patient?.hba1c_baseline_percent, "%")}
              </strong>
            </div>
            <div>
              <span>Risk category</span>
              <strong>{formatLabel(patient?.baseline_risk_category)}</strong>
            </div>
          </div>
        </div>

        <div className="report-subsection">
          <h3>Latest Sensor Readings</h3>
          <div className="report-grid">
            <div>
              <span>Glucose</span>
              <strong>{displayValue(glucose, " mg/dL")}</strong>
            </div>
            <div>
              <span>Heart rate</span>
              <strong>{displayValue(heartRate, " bpm")}</strong>
            </div>
            <div>
              <span>HRV</span>
              <strong>{displayValue(hrv, " ms")}</strong>
            </div>
            <div>
              <span>SpO₂</span>
              <strong>{displayValue(spo2, "%")}</strong>
            </div>
            <div>
              <span>Steps last hour</span>
              <strong>{displayValue(sensor?.steps_last_hour)}</strong>
            </div>
            <div>
              <span>Sleep last night</span>
              <strong>
                {displayValue(sensor?.sleep_hours_last_night, " hours")}
              </strong>
            </div>
            <div>
              <span>Recorded at</span>
              <strong>
                {sensor?.timestamp
                  ? new Date(sensor.timestamp).toLocaleString()
                  : "Not available"}
              </strong>
            </div>
          </div>
        </div>

        <div className="report-subsection">
          <h3>Two-hour Forecast</h3>
          {prediction ? (
            <>
              <p className="report-forecast-value">
                {predictedGlucose === null
                  ? "Prediction unavailable"
                  : `${predictedGlucose.toFixed(1)} mg/dL`}
              </p>
              <p>
                Demo category: {formatLabel(prediction.demo_category)}
              </p>
            </>
          ) : (
            <p>Generate a forecast on the Glucose Forecast page to include it here.</p>
          )}
        </div>

        <div className="report-subsection">
          <h3>Recent Glucose History</h3>
          {validHistory.length === 0 ? (
            <p>No historical glucose readings are available.</p>
          ) : (
            <div className="report-table-wrap">
              <table className="report-table">
                <thead>
                  <tr>
                    <th>Recorded time</th>
                    <th>Glucose (mg/dL)</th>
                  </tr>
                </thead>
                <tbody>
                  {validHistory.slice(-10).reverse().map((item, index) => (
                    <tr key={`${item.timestamp}-${index}`}>
                      <td>
                        {item.timestamp
                          ? new Date(item.timestamp).toLocaleString()
                          : "Not available"}
                      </td>
                      <td>
                        {getNumber(item.glucose_mg_dl).toFixed(1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="report-disclaimer">
          <strong>Important limitations</strong>
          <p>
            This report contains synthetic data and model-generated
            estimates. It is not a medical diagnosis or treatment
            recommendation and must not be used for clinical decisions
            or emergency response.
          </p>
        </div>
      </section>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">D</div>
          <div>
            <h2>Drishti Health</h2>
            <p>Healthcare Digital Twin</p>
          </div>
        </div>

        <nav className="sidebar-nav" aria-label="Main navigation">
          <button
            type="button"
            className={`nav-item ${currentPage === "dashboard" ? "active" : ""}`}
            aria-current={currentPage === "dashboard" ? "page" : undefined}
            onClick={() => {
              setCurrentPage("dashboard");
              setError("");
            }}
          >
            <span>▦</span> Dashboard
          </button>

          <button
            type="button"
            className={`nav-item ${currentPage === "monitoring" ? "active" : ""}`}
            aria-current={currentPage === "monitoring" ? "page" : undefined}
            onClick={() => {
              setCurrentPage("monitoring");
              setError("");
            }}
          >
            <span>♡</span> Patient Monitoring
          </button>

          <button
            type="button"
            className={`nav-item ${currentPage === "forecast" ? "active" : ""}`}
            aria-current={currentPage === "forecast" ? "page" : undefined}
            onClick={() => {
              setCurrentPage("forecast");
              setError("");
            }}
          >
            <span>⌁</span> Glucose Forecast
          </button>

          <button
            type="button"
            className={`nav-item ${currentPage === "reports" ? "active" : ""}`}
            aria-current={currentPage === "reports" ? "page" : undefined}
            onClick={() => {
              setCurrentPage("reports");
              setError("");
            }}
          >
            <span>▤</span> Reports
          </button>
        </nav>

        <div className="sidebar-footer">
          <span
            className={`status-dot ${apiStatus === "Connected" ? "online" : ""}`}
          />
          <span>{apiStatus}</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="page-header">
          <div>
            <p className="eyebrow">PREDICTIVE HEALTHCARE</p>
            <h1>{pageInfo.title}</h1>
            <p className="subtitle">{pageInfo.subtitle}</p>
          </div>

          <div className="patient-picker">
            <label htmlFor="patient-select">Select patient</label>
            <select
              id="patient-select"
              value={selectedId}
              onChange={(event) => setSelectedId(event.target.value)}
              disabled={loadingPatients || patients.length === 0}
            >
              {loadingPatients && (
                <option value="">Loading patients...</option>
              )}

              {!loadingPatients && patients.length === 0 && (
                <option value="">No patients found</option>
              )}

              {patients.map((item, index) => {
                const id = getPatientId(item);

                return (
                  <option key={id || index} value={String(id)}>
                    {id || `Patient ${index + 1}`}
                  </option>
                );
              })}
            </select>
          </div>
        </header>

        {error && (
          <div className="error-banner" role="alert">
            {error}
          </div>
        )}

        {loadingPatient && (
          <div className="loading-message">Loading patient data...</div>
        )}

        {!loadingPatients && patients.length === 0 && (
          <div className="chart-empty">
            No patients were returned by the backend. Check the Flask
            terminal and the GET /patients endpoint.
          </div>
        )}

        {/* DASHBOARD PAGE */}
        {currentPage === "dashboard" && (
          <>
            <section className="metrics-grid">
              <MetricCard
                label="Patient ID"
                value={patientId || "—"}
                description="Selected synthetic patient"
              />
              <MetricCard
                label="Latest Glucose"
                value={glucose == null ? "—" : `${glucose} mg/dL`}
                description="Latest available sensor record"
              />
              <MetricCard
                label="Heart Rate"
                value={heartRate == null ? "—" : `${heartRate} bpm`}
                description="Latest available sensor record"
              />
              <MetricCard
                label="SpO₂"
                value={spo2 == null ? "—" : `${spo2}%`}
                description="Latest available sensor record"
              />
            </section>

            <section className="chart-panel">
              <SectionHeading
                eyebrow="PATIENT OVERVIEW"
                title="Recent Glucose History"
                tag="SYNTHETIC DATA"
              />
              <GlucoseChart history={history} prediction={prediction} />

              <button
                className="primary-button"
                onClick={() => setCurrentPage("forecast")}
              >
                Open Glucose Forecast
              </button>
            </section>

            {renderPatientProfile()}

            <section className="quick-links">
              <h2>Explore Drishti Health</h2>
              <div className="quick-links-grid">
                <button onClick={() => setCurrentPage("monitoring")}>
                  <strong>Patient Monitoring</strong>
                  <span>View patient details and sensor readings</span>
                </button>
                <button onClick={() => setCurrentPage("forecast")}>
                  <strong>Glucose Forecast</strong>
                  <span>Generate predictions and run simulations</span>
                </button>
                <button onClick={() => setCurrentPage("reports")}>
                  <strong>Reports</strong>
                  <span>Review evaluation results and print a report</span>
                </button>
              </div>
            </section>
          </>
        )}

        {/* PATIENT MONITORING PAGE */}
        {currentPage === "monitoring" && (
          <>
            <section className="metrics-grid">
              <MetricCard
                label="Patient ID"
                value={patientId || "—"}
                description="Currently selected patient"
              />
              <MetricCard
                label="Glucose"
                value={glucose == null ? "—" : `${glucose} mg/dL`}
                description="Latest available record"
              />
              <MetricCard
                label="Heart Rate"
                value={heartRate == null ? "—" : `${heartRate} bpm`}
                description="Beats per minute"
              />
              <MetricCard
                label="History Readings"
                value={validHistory.length}
                description="Readings with glucose values"
              />
            </section>

            {renderPatientProfile()}

            <section className="chart-panel">
              <SectionHeading
                eyebrow="RECORDED SENSOR DATA"
                title="Glucose History"
                tag="HISTORICAL"
              />
              <p className="panel-description">
                These are historical values, not future predictions.
              </p>
              <GlucoseChart history={history} prediction={null} />

              <div className="report-table-wrap">
                <table className="report-table">
                  <thead>
                    <tr>
                      <th>Recorded time</th>
                      <th>Glucose (mg/dL)</th>
                      <th>Heart rate (bpm)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.slice(-24).reverse().map((item, index) => (
                      <tr key={`${item.timestamp}-${index}`}>
                        <td>
                          {item.timestamp
                            ? new Date(item.timestamp).toLocaleString()
                            : "Not available"}
                        </td>
                        <td>{displayValue(item.glucose_mg_dl)}</td>
                        <td>{displayValue(item.heart_rate_bpm)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {history.length === 0 && (
                <p>No historical sensor readings were returned for this patient.</p>
              )}
            </section>
          </>
        )}

        {/* GLUCOSE FORECAST PAGE */}
        {currentPage === "forecast" && renderForecast()}

        {/* REPORTS PAGE */}
        {currentPage === "reports" && (
          <>
            {renderPerformance()}
            {renderReport()}
          </>
        )}

        <footer className="page-footer">
          Drishti Health · Synthetic patient data · Research demonstration only
        </footer>
      </main>
    </div>
  );
}

export default App;

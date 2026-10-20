# Drishti Health
### Predictive Healthcare Digital Twin for Type 2 Diabetes

**Team:** Deepthink  
**Institution:** Pune Institute of Computer Technology (PICT), Pune, Maharashtra, India  
**Team Member:** Srushti Malvadkar

---

## 1. Project Title

**Drishti Health: A Predictive Healthcare Digital Twin Proof of Concept**

Drishti Health explores the use of digital patient representations and machine learning to support predictive health monitoring for Type 2 Diabetes.

## 2. Problem Statement

Traditional health monitoring often focuses on current and historical measurements. Drishti Health explores how patient-specific health data and predictive analytics can help identify possible future changes in glucose levels and investigate potential health scenarios.

## 3. Healthcare Use Case

The initial use case focuses on Type 2 Diabetes.

The prototype demonstrates:
- Patient profiles and health information.
- Simulated sensor readings and historical glucose trends.
- Machine learning-based glucose forecasting.
- What-if simulations using selected health-related inputs.
- Model performance information and printable reports.

## 4. Technical Stack

| Component | Technology |
|---|---|
| Frontend | React, Vite |
| Backend | Python, Flask |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Data Source | Synthetic patient and simulated sensor datasets |

## 5. AI/ML Model Details

The project uses a machine learning model to predict future glucose levels from selected patient and sensor-related features.

### Current reported evaluation metrics

| Metric | Value |
|---|---:|
| Mean Absolute Error (MAE) | 11.129 mg/dL |
| Root Mean Squared Error (RMSE) | 15.996 mg/dL |
| R² Score | 0.7878 |

These metrics describe the model's reported performance on the project's evaluation data. They do not establish clinical validity.

## 6. Dataset Details

The project uses synthetic datasets representing patient profiles, sensor readings, glucose predictions, meal logs, laboratory observations, and medication records.

The datasets are processed to construct model inputs and demonstrate the workflow from health data to glucose forecasting.

No real patient records are required for the current prototype.

## 7. Prototype Demonstration

The working application includes four main pages:

- **Dashboard:** Patient overview and key health indicators.
- **Monitoring:** Sensor readings and historical glucose data.
- **Forecast:** Glucose predictions and what-if simulations.
- **Reports:** Model performance information and printable reports.

**Demo video (minimum 20 minutes):** [Add recording link]

The demonstration will explain the problem, architecture, datasets, model, working prototype, results, limitations, and future scope.

## 9. Architecture Diagram

**Architecture diagram (PDF/PPT):** [Add public link]

The diagram will show the frontend, backend, synthetic datasets, machine learning model, and prediction workflow.

## 10. Project Presentation

**Project presentation (PDF/PPT):** [Add public link]

The presentation will cover the project overview, problem statement, healthcare use case, architecture, technical stack, AI/ML approach, results, prototype demonstration, limitations, and future scope.

## Limitations

Drishti Health is an educational proof of concept that uses synthetic patient data and simulated sensor readings. It is not clinically validated and must not be used for diagnosis or medical decision-making.

## Future Scope

- Integration with wearable health-monitoring devices.
- Continuous synchronization of the patient's digital representation.
- Comparison of predicted and observed outcomes.
- Expansion to additional health conditions.
- Further model validation and exploration of health-alert workflows.

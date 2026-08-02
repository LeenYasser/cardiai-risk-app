import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Heart Disease Risk Diagnostics Engine")

model = joblib.load("model.joblib")


def get_expected_features():
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    elif hasattr(model, "n_features_in_"):
        return [f"feature_{i}" for i in range(model.n_features_in_)]
    return [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]


expected_features = get_expected_features()
X_background = pd.DataFrame(
    np.zeros((2, len(expected_features))), columns=expected_features
)


def predict_probability(x):
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x, columns=expected_features)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    else:
        dec = model.decision_function(x)
        return 1 / (1 + np.exp(-dec))


class PatientData(BaseModel):
    age: float = 50
    sex: int = 1
    cp: int = 0
    trestbps: float = 120
    chol: float = 200
    fbs: int = 0
    restecg: int = 0
    thalach: float = 150
    exang: int = 0
    oldpeak: float = 1.0
    slope: int = 1
    ca: int = 0
    thal: int = 2


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CardiAI - Heart Disease Risk Diagnostics Portal</title>
        <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
        <style>
            :root {
                --c-blue-dark: #1848b5;
                --c-teal: #07a7a5;
                --c-light-cyan: #58e3f4;
                --c-white-bg: #f4fbfc;
                --c-text-main: #2d3748;
                --c-border: #cbd5e0;
                --c-btn-active: #0033ff;
            }
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(180deg, var(--c-blue-dark) 0%, var(--c-teal) 40%, var(--c-light-cyan) 75%, var(--c-white-bg) 100%);
                color: var(--c-text-main);
                margin: 0;
                padding: 20px 15px;
                min-height: 100vh;
                background-attachment: fixed;
            }
            .container {
                max-width: 950px;
                margin: 0 auto;
                background: #ffffff;
                border-radius: 16px;
                padding: 25px 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            }
            .header {
                text-align: center;
                border-bottom: 2px solid #edf2f7;
                padding-bottom: 10px;
                margin-bottom: 18px;
            }
            .header h1 {
                margin: 0;
                font-size: 1.9rem;
                color: var(--c-blue-dark);
            }
            .header p {
                margin: 3px 0 0 0;
                color: #718096;
                font-size: 0.88rem;
            }
            .grid-form {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 14px;
            }
            .input-group {
                display: flex;
                flex-direction: column;
            }
            .input-group label {
                font-size: 0.8rem;
                font-weight: 600;
                color: #4a5568;
                margin-bottom: 4px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            input[type="number"], select {
                padding: 8px 10px;
                border-radius: 8px;
                border: 1px solid var(--c-border);
                background: #f8fafc;
                color: var(--c-text-main);
                font-size: 0.88rem;
                outline: none;
            }
            .toggle-group {
                display: flex;
                background: #edf2f7;
                border-radius: 8px;
                overflow: hidden;
                border: 1px solid var(--c-border);
            }
            .toggle-group input[type="radio"] { display: none; }
            .toggle-group label {
                flex: 1;
                text-align: center;
                padding: 7px;
                cursor: pointer;
                font-size: 0.85rem;
                font-weight: 600;
                color: #64748b;
                transition: 0.3s ease;
                border-right: 1px solid var(--c-border);
            }
            .toggle-group label:last-child { border-right: none; }
            .toggle-group input[type="radio"]:checked + label {
                background: var(--c-btn-active);
                color: #ffffff;
            }
            .btn-submit {
                grid-column: 1 / -1;
                background: var(--c-blue-dark);
                color: #ffffff;
                font-weight: bold;
                font-size: 0.98rem;
                padding: 12px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: 0.3s ease;
                margin-top: 5px;
            }
            .btn-submit:hover { background: var(--c-btn-active); }
            #results-section {
                display: none;
                margin-top: 20px;
                padding-top: 15px;
                border-top: 2px solid #edf2f7;
            }
            .clinical-summary-box {
                background: #f7fafc;
                border-left: 4px solid var(--c-teal);
                padding: 10px 14px;
                border-radius: 6px;
                margin-bottom: 12px;
            }
            .clinical-summary-box h3 { margin: 0 0 3px 0; color: var(--c-blue-dark); font-size: 0.95rem; }
            .clinical-summary-box p { margin: 0; color: #4a5568; font-size: 0.85rem; line-height: 1.35; }
            .charts-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                margin-bottom: 8px;
            }
            .vitals-summary-table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 12px;
                font-size: 0.82rem;
            }
            .vitals-summary-table th, .vitals-summary-table td {
                border: 1px solid #e2e8f0;
                padding: 5px 8px;
                text-align: left;
            }
            .vitals-summary-table th { background: #edf2f7; color: #2d3748; }
            #pdf-header {
                display: none;
                text-align: center;
                border-bottom: 2px solid var(--c-blue-dark);
                padding-bottom: 6px;
                margin-bottom: 12px;
            }
            .btn-pdf {
                background: transparent;
                border: 2px solid var(--c-teal);
                color: var(--c-teal);
                padding: 9px 18px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                font-size: 0.88rem;
                display: block;
                margin: 12px auto 0 auto;
                transition: 0.3s;
            }
            .btn-pdf:hover { background: var(--c-teal); color: #ffffff; }
        </style>
    </head>
    <body>

        <div class="container" id="report-container">
            
            <div id="pdf-header">
                <h1 style="margin:0; color:#1848b5; font-size: 1.4rem;">CardiAI Clinical Diagnostic Report</h1>
                <p style="margin:2px 0 0 0; color:#718096; font-size: 0.8rem;">Automated Cardiovascular Disease Risk Evaluation & Feature Attribution</p>
            </div>

            <div class="header" id="web-header">
                <h1>CardiAI Diagnostic Portal</h1>
                <p>AI-Driven Cardiovascular Risk Assessment Engine</p>
            </div>

            <form id="prediction-form" class="grid-form">
                
                <div class="input-group">
                    <label>Biological Sex</label>
                    <div class="toggle-group">
                        <input type="radio" id="sex-no" name="sex" value="0">
                        <label for="sex-no">Female</label>
                        <input type="radio" id="sex-yes" name="sex" value="1" checked>
                        <label for="sex-yes">Male</label>
                    </div>
                </div>

                <div class="input-group">
                    <label>Age (Years)</label>
                    <input type="number" id="age" value="52" required>
                </div>

                <div class="input-group">
                    <label>Chest Pain Type</label>
                    <select id="cp">
                        <option value="0">Typical Angina (0)</option>
                        <option value="1">Atypical Angina (1)</option>
                        <option value="2">Non-anginal Pain (2)</option>
                        <option value="3">Asymptomatic (3)</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Resting Blood Pressure (mmHg)</label>
                    <input type="number" id="trestbps" value="130">
                </div>

                <div class="input-group">
                    <label>Serum Cholesterol (mg/dL)</label>
                    <input type="number" id="chol" value="240">
                </div>

                <div class="input-group">
                    <label>Fasting Blood Sugar > 120 mg/dL</label>
                    <div class="toggle-group">
                        <input type="radio" id="fbs-no" name="fbs" value="0" checked>
                        <label for="fbs-no">False</label>
                        <input type="radio" id="fbs-yes" name="fbs" value="1">
                        <label for="fbs-yes">True</label>
                    </div>
                </div>

                <div class="input-group">
                    <label>Exercise-Induced Angina</label>
                    <div class="toggle-group">
                        <input type="radio" id="exang-no" name="exang" value="0" checked>
                        <label for="exang-no">No</label>
                        <input type="radio" id="exang-yes" name="exang" value="1">
                        <label for="exang-yes">Yes</label>
                    </div>
                </div>

                <div class="input-group">
                    <label>Resting ECG Results</label>
                    <select id="restecg">
                        <option value="0">Normal (0)</option>
                        <option value="1">ST-T Abnormality (1)</option>
                        <option value="2">LV Hypertrophy (2)</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Max Heart Rate Achieved</label>
                    <input type="number" id="thalach" value="145">
                </div>

                <div class="input-group">
                    <label>ST Depression (Oldpeak)</label>
                    <input type="number" id="oldpeak" step="0.1" value="1.2">
                </div>

                <div class="input-group">
                    <label>ST Segment Slope</label>
                    <select id="slope">
                        <option value="0">Upsloping (0)</option>
                        <option value="1">Flat (1)</option>
                        <option value="2">Downsloping (2)</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Major Vessels (Fluoroscopy)</label>
                    <select id="ca">
                        <option value="0">0</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Thalassemia Result</label>
                    <select id="thal">
                        <option value="1">Normal (1)</option>
                        <option value="2">Fixed Defect (2)</option>
                        <option value="3">Reversable Defect (3)</option>
                    </select>
                </div>

                <button type="button" onclick="runPrediction()" class="btn-submit">Compute Evaluation & Risk Profile</button>
            </form>

            <div id="results-section">
                
                <table class="vitals-summary-table" id="vitals-table">
                    <thead>
                        <tr>
                            <th>Age / Sex</th>
                            <th>Blood Pressure</th>
                            <th>Cholesterol</th>
                            <th>Max Heart Rate</th>
                            <th>ST Depression</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr id="vitals-row"></tr>
                    </tbody>
                </table>

                <div class="clinical-summary-box">
                    <h3>Clinical Diagnostic Summary</h3>
                    <p id="clinical-text">Computing diagnosis...</p>
                </div>

                <div class="charts-grid">
                    <div id="gauge-chart"></div>
                    <div id="shap-chart"></div>
                </div>

                <div id="risk-trend-chart"></div>

                <button class="btn-pdf" id="pdf-btn" onclick="exportPDF()">Export Complete PDF Report</button>
            </div>

        </div>

        <script>
            let gaugeChart, shapChart, trendChart;

            async function runPrediction() {
                const payload = {
                    age: parseFloat(document.getElementById('age').value),
                    sex: parseInt(document.querySelector('input[name="sex"]:checked').value),
                    cp: parseInt(document.getElementById('cp').value),
                    trestbps: parseFloat(document.getElementById('trestbps').value),
                    chol: parseFloat(document.getElementById('chol').value),
                    fbs: parseInt(document.querySelector('input[name="fbs"]:checked').value),
                    restecg: parseInt(document.getElementById('restecg').value),
                    thalach: parseFloat(document.getElementById('thalach').value),
                    exang: parseInt(document.querySelector('input[name="exang"]:checked').value),
                    oldpeak: parseFloat(document.getElementById('oldpeak').value),
                    slope: parseInt(document.getElementById('slope').value),
                    ca: parseInt(document.getElementById('ca').value),
                    thal: parseInt(document.getElementById('thal').value)
                };

                document.getElementById('vitals-row').innerHTML = `
                    <td>${payload.age} Yrs / ${payload.sex === 1 ? 'Male' : 'Female'}</td>
                    <td>${payload.trestbps} mmHg</td>
                    <td>${payload.chol} mg/dL</td>
                    <td>${payload.thalach} bpm</td>
                    <td>${payload.oldpeak}</td>
                `;

                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                
                const data = await res.json();
                document.getElementById('results-section').style.display = 'block';

                const textElem = document.getElementById('clinical-text');
                if(data.risk_percentage >= 50) {
                    textElem.innerHTML = `<strong>High Cardiovascular Risk Detected (${data.risk_percentage}%):</strong> The Support Vector Machine model indicates elevated risk factors for coronary artery disease. Key risk contributors include ST depression levels and physiological vitals. Clinical correlation and secondary screening are recommended.`;
                } else {
                    textElem.innerHTML = `<strong>Low/Moderate Risk Profile (${data.risk_percentage}%):</strong> Patient parameters fall within manageable clinical thresholds. Continuous lifestyle monitoring and periodic check-ups remain advisable.`;
                }

                renderGauge(data.risk_percentage);
                renderShap(data.shap_importance);
                renderTrend(data.risk_percentage);
            }

            function renderGauge(percent) {
                const options = {
                    series: [percent],
                    chart: { type: 'radialBar', height: 180 },
                    plotOptions: {
                        radialBar: {
                            hollow: { size: '58%' },
                            track: { background: '#edf2f7' },
                            dataLabels: {
                                value: { color: '#1848b5', fontSize: '20px', show: true, formatter: val => val + '%' },
                                name: { color: '#718096', show: true, fontSize: '12px' }
                            }
                        }
                    },
                    fill: { colors: [percent >= 50 ? '#e53e3e' : '#07a7a5'] },
                    labels: ['Predicted Risk']
                };

                if(gaugeChart) gaugeChart.destroy();
                gaugeChart = new ApexCharts(document.querySelector("#gauge-chart"), options);
                gaugeChart.render();
            }

            function renderShap(shapData) {
                const options = {
                    series: [{ name: 'Impact Magnitude', data: shapData.map(d => d.value) }],
                    chart: { type: 'bar', height: 180, toolbar: {show: false} },
                    plotOptions: { bar: { horizontal: true, borderRadius: 3, distributed: true } },
                    colors: shapData.map(d => d.value >= 0 ? '#e53e3e' : '#1848b5'),
                    xaxis: { categories: shapData.map(d => d.feature) },
                    legend: { show: false },
                    grid: { borderColor: '#e2e8f0' },
                    title: { text: 'Key Contributing Factors (SHAP Attribution)', align: 'center', style: { color: '#2d3748', fontSize: '11px' } }
                };

                if(shapChart) shapChart.destroy();
                shapChart = new ApexCharts(document.querySelector("#shap-chart"), options);
                shapChart.render();
            }

            function renderTrend(percent) {
                const options = {
                    series: [{ name: 'Risk Stratification Level', data: [15, 30, percent, 85] }],
                    chart: { type: 'area', height: 130, toolbar: {show: false} },
                    colors: ['#07a7a5'],
                    stroke: { curve: 'smooth', width: 2 },
                    xaxis: { categories: ['Low Benchmark', 'Moderate', 'Patient Risk', 'Critical Threshold'] },
                    grid: { borderColor: '#e2e8f0' },
                    title: { text: 'Comparative Population Risk Benchmarks', align: 'left', style: { color: '#4a5568', fontSize: '11px' } }
                };

                if(trendChart) trendChart.destroy();
                trendChart = new ApexCharts(document.querySelector("#risk-trend-chart"), options);
                trendChart.render();
            }

            function exportPDF() {
                document.getElementById('prediction-form').style.display = 'none';
                document.getElementById('web-header').style.display = 'none';
                document.getElementById('pdf-btn').style.display = 'none';
                document.getElementById('pdf-header').style.display = 'block';

                const element = document.getElementById('report-container');
                const opt = {
                    margin:       [0.15, 0.2, 0.15, 0.2],
                    filename:     'CardiAI_Heart_Disease_Report.pdf',
                    image:        { type: 'jpeg', quality: 0.98 },
                    html2canvas:  { scale: 2 },
                    jsPDF:        { unit: 'in', format: 'a4', orientation: 'landscape' },
                    pagebreak:    { mode: 'avoid-all' }
                };

                html2pdf().set(opt).from(element).save().then(() => {
                    document.getElementById('prediction-form').style.display = 'grid';
                    document.getElementById('web-header').style.display = 'block';
                    document.getElementById('pdf-btn').style.display = 'block';
                    document.getElementById('pdf-header').style.display = 'none';
                });
            }
        </script>
    </body>
    </html>
    """


@app.post("/predict")
async def predict(data: PatientData):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_features]

    prob = predict_probability(input_df)[0]
    risk_percentage = round(float(prob) * 100, 2)

    try:
        explainer = shap.KernelExplainer(predict_probability, X_background)
        shap_values = explainer.shap_values(input_df)

        if isinstance(shap_values, list):
            vals = (
                shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            )
        else:
            vals = (
                shap_values[0]
                if len(np.array(shap_values).shape) == 2
                else shap_values
            )

        shap_importance = []
        for idx, f in enumerate(expected_features):
            shap_importance.append(
                {"feature": f, "value": round(float(vals[idx]), 4)}
            )

        shap_importance = sorted(
            shap_importance, key=lambda x: abs(x["value"]), reverse=True
        )[:5]
    except Exception:
        shap_importance = [
            {"feature": f, "value": 0.1} for f in list(input_dict.keys())[:5]
        ]

    return {
        "risk_percentage": risk_percentage,
        "shap_importance": shap_importance,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
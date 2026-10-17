# 🌿 DigitalWell AI
### A digital-wellness machine-learning prototype for IBM Z / LinuxONE

DigitalWell AI explores how smartphone-use and lifestyle signals can be used to estimate a **model-based digital-habit risk score** and compare hypothetical habit scenarios. The project combines a scikit-learn classification pipeline, a Streamlit interface, and IBM Z / LinuxONE (s390x) environment experimentation.

> **Status: Datathon prototype.** The repository contains the cleaned model-development notebook, Streamlit dashboard, dependency list, and setup guidance. The source dataset and trained model artifact are intentionally not committed; train the model from an authorized copy of the dataset before launching the app.

## ✨ What the project demonstrates

- **End-to-end ML pipeline:** numeric and categorical preprocessing inside a scikit-learn Pipeline.
- **Leakage-aware split:** stratified 80/20 training and holdout test split, with preprocessing fitted only on training data.
- **Classification evaluation:** accuracy, ROC-AUC, precision/recall/F1 report, and confusion matrix.
- **Model interpretation:** permutation feature importance measured using test-set ROC-AUC.
- **What-if exploration:** compare model outputs after hypothetical changes to daily screen time, social-media time, and sleep.
- **Interactive demo:** Streamlit dashboard with editable habit inputs and side-by-side scenario results.
- **Platform exploration:** project work recorded on IBM Z / LinuxONE using the s390x architecture.

## 🧠 Recorded prototype results

The original uploaded notebook recorded these results for a 7,500-row dataset and a 1,500-row holdout set:

| Metric | Recorded result |
|---|---:|
| Dataset rows | 7,500 |
| Predictor features | 12 |
| Training examples | 6,000 |
| Test examples | 1,500 |
| Accuracy | 87.53% |
| ROC-AUC | 0.9540 |

Recorded confusion matrix (actual rows, predicted columns):

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 386 | 52 |
| Actual 1 | 135 | 927 |

These figures are transcribed from the notebook's saved output; they have **not** been independently reproduced from the files currently in this repository because the dataset is not included. Re-run the workflow against the same authorized dataset before using the metrics in a submission or presentation.

In the recorded permutation-importance output, daily_screen_time_hours and social_media_hours had the largest importance scores. This describes model behavior on the recorded test set; it does not show that either factor causes addiction.

## 🧰 Tech stack

- Python
- pandas and NumPy
- scikit-learn: preprocessing, Logistic Regression, evaluation, permutation importance
- joblib: model serialization
- Matplotlib: analysis plots
- Streamlit: interactive dashboard
- IBM Z / LinuxONE (s390x): target environment explored

## 📁 Repository structure

~~~text
DigitalWell-AI/
├── README.md
├── Digitalwell_Ai.ipynb       # Cleaned model training and evaluation workflow
├── app.py                     # Standalone Streamlit dashboard
├── requirements.txt           # Core dependency ranges
└── .gitignore                 # Excludes local data, model artifacts, logs, secrets
~~~

## 📊 Dataset and model inputs

The notebook expects a CSV named:

~~~text
Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
~~~

The target is addicted_label. The training code excludes transaction_id, user_id, and addiction_level from the predictors.

**Numeric predictors:** age, daily_screen_time_hours, social_media_hours, gaming_hours, work_study_hours, sleep_hours, notifications_per_day, app_opens_per_day, weekend_screen_time.

**Categorical predictors:** gender, stress_level, academic_work_impact.

The notebook generates two local artifacts when its export cell runs:

- digitalwell_model.pkl — the fitted preprocessing and classification pipeline.
- digitalwell_features.json — metadata describing model input columns.

The dataset and serialized model are ignored by Git so they are not accidentally committed. Add data or model artifacts only after checking ownership, license, privacy, file size, and the datathon's submission rules.

## 🚀 Run locally

### 1. Clone and install dependencies

~~~bash
git clone https://github.com/Pavankumarng/DigitalWell-AI.git
cd DigitalWell-AI

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

The requirements file specifies a tested Streamlit version from the notebook's environment and compatible version ranges for core libraries. It is not a fully locked environment; confirm package availability for your chosen Python version and s390x image.

### 2. Add the dataset and train

Obtain the CSV from a source you are authorized to use. Place it in the repository root beside Digitalwell_Ai.ipynb, open the notebook, and run its cells from top to bottom. The training and export cells will write digitalwell_model.pkl and digitalwell_features.json to the working directory.

Do not publish the dataset unless you have explicit permission to redistribute it and have checked for personal or sensitive information.

### 3. Start the dashboard

After training, make sure digitalwell_model.pkl is in the same directory as app.py, then run:

~~~bash
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
~~~

Use the local URL printed by Streamlit, or your hosted notebook's supported preview / port-forwarding URL. The address 0.0.0.0 is a bind address and usually should not be typed directly into a browser. In a hosted IBM environment, localhost access may require the platform's own port proxy.

## 🏗️ IBM Z / LinuxONE notes

The original notebook recorded Python 3.12 running on Linux s390x and experiments to install and troubleshoot Streamlit dependencies. Availability can vary between images and package versions, so successful operation in one environment does not guarantee portability to every LinuxONE system.

For a datathon demo, check the environment architecture, install dependencies supported by the target image, train and serve the model in compatible environments where practical, and verify the platform's required way of exposing Streamlit ports.

## ⚠️ Limitations and responsible use

- **Not a diagnosis.** This prototype is not a medical, psychological, or clinical assessment and should not be used to label people or make high-impact decisions.
- **Probability is not automatically calibrated.** The dashboard displays the classifier's predicted probability for the positive label; that value should not automatically be interpreted as a true clinical or population-level risk percentage.
- **What-if is not causal.** The simulator changes input features and asks the same model for another output. It cannot establish that making the change will cause the predicted change in real life.
- **Generalization is unverified.** The recorded holdout metrics are from one dataset. The dataset provenance, label definition, external validation, calibration, and subgroup performance should be documented before making real-world claims.
- **Data and model provenance matter.** Verify permission to use the dataset and model, minimize retained personal data, and document limitations and bias risks.
- **Reproducibility is in progress.** Dataset and trained model files are not in the repository, and requirements are not fully version-locked.

## 🗺️ Suggested next steps

- [ ] Document dataset source, license, target-label meaning, and collection methodology.
- [ ] Reproduce the recorded metrics from a clean environment.
- [ ] Add tested environment locking for the intended IBM Z / LinuxONE image.
- [ ] Add automated tests for input-schema compatibility and model-file errors.
- [ ] Evaluate probability calibration and subgroup performance.
- [ ] Add clear explanations of model uncertainty and privacy practices.
- [ ] Capture an approved screenshot of the running dashboard for the datathon submission.

## Acknowledgement

Built as a digital-wellness machine-learning prototype in an IBM Z Datathon context. IBM Z / LinuxONE is the platform explored by this project; this repository does not imply IBM endorsement.

## License

No license has been added. Until a license is chosen and committed, others should not assume they have permission to reuse, modify, or redistribute the project's code. The dataset has its own separate licensing and privacy requirements.

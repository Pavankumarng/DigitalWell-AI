# 🌿 DigitalWell AI
### Digital wellness risk estimation and habit what-if simulation on IBM Z / LinuxONE

DigitalWell AI is a machine-learning prototype that estimates a user's digital-habit risk from smartphone-usage and lifestyle inputs. It pairs a scikit-learn classification pipeline with a Streamlit dashboard and simple scenario simulations, with an implementation path explored on IBM Z / LinuxONE (`s390x`).

> **Project status:** Datathon prototype / research demo. The notebook includes model-training and evaluation code plus the dashboard implementation. The training CSV and serialized model are referenced by the notebook but are not bundled in this repository yet. The app therefore needs those artifacts (or a retraining step) before it can run end to end.

## What it does

- Trains a **Logistic Regression** classifier wrapped in a scikit-learn preprocessing pipeline.
- Handles numeric features with median imputation and standard scaling.
- Handles categorical features with most-frequent imputation and one-hot encoding.
- Uses a stratified 80/20 train/test split (`random_state=42`).
- Reports accuracy, ROC-AUC, classification metrics, and a confusion matrix.
- Explores permutation feature importance on the test split.
- Demonstrates what-if scenarios, including reducing daily screen time, reducing social-media time, and increasing sleep.
- Builds a Streamlit interface for entering habit information and viewing a model-estimated risk score.

## Why it matters

Digital devices are woven into study, work, entertainment, and social life. DigitalWell AI is designed as an exploratory tool to help people reflect on usage patterns and compare hypothetical habit changes. It is **not a medical or psychological diagnostic tool** and should not be used to label people, make clinical claims, or make consequential decisions.

## Prototype results

The uploaded notebook contains the following evaluation output from its recorded run:

| Metric | Recorded result |
|---|---:|
| Dataset rows | 7,500 |
| Input features used | 12 |
| Training examples | 6,000 |
| Test examples | 1,500 |
| Accuracy | 0.8753 (87.53%) |
| ROC-AUC | 0.9540 |

Recorded confusion matrix (actual rows, predicted columns; classes `0`, `1`):

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 386 | 52 |
| Actual 1 | 135 | 927 |

These are **the notebook's recorded results**, not a claim that the model has been independently rerun from this repository. Reproduce and validate them with the exact dataset and environment before using them in a presentation or deployment.

### Feature interpretation

In the recorded permutation-importance output, `daily_screen_time_hours` and `social_media_hours` were the strongest contributors to test ROC-AUC. Permutation importance measures how evaluation performance changes when a feature is shuffled; it does not establish causation. Small or negative scores should not be interpreted as proof that a behavior is irrelevant.

## Repository contents

```text
DigitalWell-AI/
├── README.md
├── Digitalwell_Ai.ipynb
└── .gitignore
```

The notebook also contains environment checks, Streamlit installation/debugging attempts, code that writes `app.py`, and server diagnostics. It reflects the project-building workflow as well as the core model pipeline.

## Data

The notebook expects a CSV named:

```text
Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv
```

Its recorded schema includes identifiers (`transaction_id`, `user_id`), user-habit fields, `addiction_level`, and the target `addicted_label`. The model code excludes the identifier fields and `addiction_level` from the predictors, then uses `addicted_label` as the target.

The data file is **not included** in this repository. Add it only if you have permission to redistribute it and it contains no personal or sensitive information. Otherwise, document the data source and provide reproducible instructions for obtaining it.

## Model inputs

The recorded training pipeline uses these 12 predictors:

- **Numeric:** `age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time`
- **Categorical:** `gender`, `stress_level`, `academic_work_impact`

The target is `addicted_label`. The notebook uses a balanced Logistic Regression classifier (`class_weight="balanced"`, `max_iter=1000`) inside a preprocessing pipeline.

## Run it

### 1. Set up Python

Use Python 3.10+ where the dependencies are compatible with your environment. On an IBM LinuxONE environment, check package availability for the `s390x` architecture.

```bash
git clone https://github.com/Pavankumarng/DigitalWell-AI.git
cd DigitalWell-AI

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a `requirements.txt` with the versions tested in your target environment before relying on this installation command. The uploaded notebook currently contains package-installation experiments for Streamlit and its dependencies, rather than a fully pinned, validated environment file.

### 2. Prepare the model artifacts

Place the CSV in the notebook's working directory, then run the model-training and model-export cells in `Digitalwell_Ai.ipynb`. They generate a serialized model (`digitalwell_model.pkl`) and feature metadata (`digitalwell_features.json`) in the notebook or `digitalwell_app/` directory, depending on which export cell you run.

**Important:** The current notebook's Streamlit app uses a hard-coded path from its original IBM environment (`/home/ibm-user/Digital well AI/digitalwell_app/digitalwell_model.pkl`). For portable use, change this to a path relative to `app.py`, for example:

```python
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "digitalwell_model.pkl"
```

The app code also needs to be saved as `app.py` in the same directory as the model file.

### 3. Launch the dashboard

From the directory containing `app.py` and `digitalwell_model.pkl`:

```bash
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Open the local URL printed by Streamlit, or use the appropriate port-forwarding / application-preview URL provided by your notebook platform. `0.0.0.0` is a bind address, not usually the address to type into a browser. In hosted notebooks, `127.0.0.1:8501` may not be exposed directly; use the platform's supported port-proxy mechanism.

## IBM Z / LinuxONE notes

The notebook records an environment of Python 3.12 on Linux `s390x` and iterates through installing and diagnosing Streamlit dependencies. This is useful platform experimentation, but does not by itself prove that every dependency or deployment configuration is reproducible on all LinuxONE environments.

For a datathon demonstration:
1. Confirm the runtime architecture with `python -c "import platform; print(platform.machine())"`.
2. Install only dependencies that are available and supported for the target image.
3. Train and export the model in the same compatible environment used for inference where practical.
4. Launch Streamlit using the platform's supported port-forwarding mechanism.
5. Keep the notebook, app code, model artifact, and data provenance clearly separated.

## Important limitations and next steps

- **Not yet fully reproducible from this repository:** the dataset, `app.py` as a standalone file, serialized model, and a verified dependency lock file are not included with the uploaded notebook.
- **Model validation:** the logged metrics are promising prototype metrics, but the dataset source, label construction, class definition, and external validation should be documented. A random holdout from one dataset does not establish real-world generalization.
- **Probability calibration:** the displayed probability is the classifier's `predict_proba` output. It should not automatically be interpreted as a calibrated clinical or population-level risk percentage.
- **What-if simulations are not causal estimates:** changing a feature and querying the same model only shows how the model output changes under that hypothetical input. It does not prove that the behavior change will cause a person's risk to change.
- **Feature coverage:** the dashboard's fields should match the exact training schema and categories. Keep preprocessing in the saved pipeline and validate inputs before prediction.
- **Portability:** replace the original absolute IBM workspace paths with project-relative paths.
- **Privacy and fairness:** document data provenance and permissions, minimize personal data, and assess performance across relevant groups before making user-facing claims.

## Roadmap

- [ ] Make model and dataset setup reproducible.
- [ ] Move the final Streamlit app into a standalone, tested `app.py`.
- [ ] Replace hard-coded paths with project-relative paths.
- [ ] Add a tested `requirements.txt` for the intended IBM Z / LinuxONE environment.
- [ ] Add automated checks for feature-schema compatibility and missing/corrupt model files.
- [ ] Add validation, calibration, and subgroup performance reporting.
- [ ] Improve the UI with clear explanations, uncertainty notes, and privacy guidance.
- [ ] Document the source, license, and permitted use of the dataset and model.

## Built with

- Python
- pandas and NumPy
- scikit-learn
- joblib
- Matplotlib
- Streamlit
- IBM Z / LinuxONE (`s390x`) environment explored in the notebook

## Acknowledgement

Built as a digital-wellness machine-learning prototype for an IBM Z Datathon context. IBM Z / LinuxONE is the target platform explored in the notebook; this repository does not imply IBM endorsement.

## License

No license has been added yet. Until a license is included, do not assume others have permission to reuse, modify, or redistribute this project. Also confirm the dataset's license separately.

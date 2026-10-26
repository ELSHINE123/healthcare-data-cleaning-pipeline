# 🏥 Clinical EHR Data Cleaning & Cascading Imputation Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy)](https://numpy.org/)
[![Data Integrity](https://img.shields.io/badge/Data%20Integrity-Validated-success)](#results--impact)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> A production-grade, domain-informed data cleaning and validation pipeline designed to resolve messy Electronic Health Record (EHR) data, eliminate invalid synthetic profiles, and prepare clinical records for downstream machine learning and analytical modeling.

---

## 📌 Executive Summary

Real-world healthcare datasets frequently suffer from corrupted types, composite metrics, unstandardized text, and significant missingness. In clinical analytics, applying naive data cleaning—such as universal mean imputation or indiscriminate row deletion—distorts epidemiological distributions or fabricates synthetic patients.

This project delivers an end-to-end Python/Pandas data-cleaning framework that cleans, restructures, and audits **1,000 raw patient records**, transforming messy healthcare data into a **983-record, zero-null, audit-compliant analytical dataset**.

```mermaid
flowchart TD
    A[Raw EHR Dataset (1,000 Records)] --> B[Text Sanitization & Header Normalization]
    B --> C[Composite Vital Decomposition (Blood Pressure -> Systolic / Diastolic)]
    C --> D[Resilient Timestamp Parsing (dateutil fuzzy parsing)]
    D --> E[Clinical Triage Filter: Prune 17 Unrecoverable Records (1.70%)]
    E --> F[Hierarchical Cascading Median Imputation: (Age Bracket + Gender) -> Gender -> Global]
    F --> G[Regulatory Imputation Audit Flags (Yes/No Flags)]
    G --> H[Dual-Layer Anomaly Detection (Physiological Limits + 3.0x IQR)]
    H --> I[Cleaned Analytics-Ready Dataset (983 Records, 0 Nulls)]
```

---

## 🔬 Core Engineering & Clinical Methodology

### 1. Composite Vital Decomposition
* **Problem:** Blood pressure was recorded as a composite text string (e.g., `"120/80"`), making mathematical analysis impossible.
* **Solution:** Vectorized splitting into `systolic_bp` and `diastolic_bp` numeric columns with whitespace stripping, string coercion, and nullable integer typing (`Int64`).

### 2. Typo Normalization & Mixed-Type Correction
* **Problem:** The `age` column contained verbal strings (`"forty"`), floating-point values, and missing entries.
* **Solution:** Standardized verbal entries into numeric representations, coerced invalid tokens using `pd.to_numeric(..., errors='coerce')`, and cast values to nullable integer `Int64`.

### 3. Resilient Timestamp Normalization
* **Problem:** Visit dates contained polymorphic formats, underscores, and variable delimiters (e.g., `"2023_04_15"` vs `"15-Apr-2023 10:30"`).
* **Solution:** Engineered a robust parsing utility leveraging `dateutil.parser` with `fuzzy=True`, achieving a **100% parse success rate** across all records without introducing unhandled `NaT` values.

### 4. The "Ghost Patient" Triage Filter
* **Problem:** Blindly imputing rows where a patient has no clinical history creates synthetic "hallucinated patients." Conversely, listwise deletion drops too many records (~10%).
* **Solution:** Established a domain-driven triage rule auditing 4 critical clinical features: `['age', 'blood_pressure', 'condition', 'cholesterol']`. Rows missing $\ge 3$ critical features were pruned as fundamentally unrecoverable.
* **Impact:** Discarded exactly **17 rows (1.70%)**, preserving **98.3% of the patient cohort** while preventing synthetic profile generation.

### 5. Hierarchical Cascading Median Imputation
* **Problem:** Global mean or median imputation collapses natural physiological variances across age brackets and biological sex.
* **Solution:** Engineered a 3-tier fallback median imputation architecture:
  $$	ext{Imputed Value} = 	ext{Fallback}\Big(	ext{Median}_{(	ext{Age Bracket}, 	ext{Gender})} \longrightarrow 	ext{Median}_{	ext{Gender}} \longrightarrow 	ext{Median}_{	ext{Population}}\Big)$$
* **Impact:** Preserves epidemiological trends without skewing demographic distributions.

### 6. Regulatory Audit Flags (Compliance-Ready)
* **Problem:** Regulatory frameworks (HIPAA, GAMP 5, FDA 21 CFR Part 11) require transparency regarding which values were measured versus imputed.
* **Solution:** Created boolean audit flags (`cholesterol_imputed`, `systolic_bp_imputed`, `diastolic_bp_imputed`) mapped to `'Yes'` and `'No'`, establishing full data provenance.

### 7. Dual-Layer Anomaly Detection
* **Layer 1 (Physiological Bounds):** Audited vital signs against human survivability limits (Systolic: 60–240 mmHg, Diastolic: 40–140 mmHg, Cholesterol: 70–500 mg/dL).
* **Layer 2 (Statistical IQR):** Evaluated extreme outliers using a conservative $3.0	imes	ext{IQR}$ threshold to highlight severe outliers without deleting genuine high-acuity patients.

---

## 📊 Results & Impact

| Metric | Raw Dataset | Cleaned Output | Improvement |
|---|---|---|---|
| **Total Patient Records** | 1,000 | 983 | 17 unrecoverable rows removed (1.70%) |
| **Total Features / Columns** | 10 | 15 | +5 engineered clinical & audit attributes |
| **Missing Values (Total Nulls)** | 1,325 | **0** | **100% missing value resolution** |
| **Blood Pressure Usability** | 0% (composite string) | 100% (`systolic_bp`, `diastolic_bp`) | Analytics-ready numeric vitals |
| **Date Usability** | Inconsistent strings | 100% `datetime64[ns]` | Time-series capable |
| **Audit Compliance** | None | 100% Provenance Tracking | Full transparency on imputed markers |

---

## 🗂️ Data Dictionary (Cleaned Schema)

| Column Name | Data Type | Description | Sample Values |
|---|---|---|---|
| `patient_name` | `string` | Standardized lowercase patient name | `"david lee"`, `"emily davis"` |
| `age` | `Int64` | Patient age in completed years | `25`, `40`, `60` |
| `gender` | `category` | Biological sex or demographic category | `"Male"`, `"Female"`, `"Other"` |
| `condition` | `string` | Primary diagnosed health condition | `"Hypertension"`, `"Diabetes"` |
| `medication` | `string` | Prescribed therapeutic medication | `"Metformin"`, `"Lisinopril"` |
| `visit_date` | `datetime64[ns]` | Standardized date of clinical visit | `2023-04-15` |
| `systolic_bp` | `Int64` | Systolic blood pressure (mmHg) | `120`, `130`, `140` |
| `diastolic_bp` | `Int64` | Diastolic blood pressure (mmHg) | `80`, `85`, `90` |
| `cholesterol` | `Int64` | Serum cholesterol level (mg/dL) | `180`, `200`, `220` |
| `age_group` | `category` | Demographic age bracket | `"<30"`, `"30-44"`, `"45-59"`, `"60-74"`, `"75+"` |
| `email` | `string` | Patient contact email address | `"name@hospital.org"`, `"Unknown"` |
| `phone_number` | `string` | Sanitized phone number | `"555-555-5555"`, `"Unknown"` |
| `cholesterol_imputed` | `string` | Provenance flag indicating imputation | `"Yes"`, `"No"` |
| `systolic_bp_imputed` | `string` | Provenance flag indicating imputation | `"Yes"`, `"No"` |
| `diastolic_bp_imputed` | `string` | Provenance flag indicating imputation | `"Yes"`, `"No"` |

---

## 🚀 Quick Start & Reproducibility

### 1. Clone the Repository
```bash
git clone https://github.com/[YOUR-USERNAME]/healthcare-data-cleaning.git
cd healthcare-data-cleaning
```

### 2. Environment Setup
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Dependencies (`requirements.txt`)
```text
pandas>=2.0.0
numpy>=1.24.0
python-dateutil>=2.8.2
jupyter>=1.0.0
```

### 4. Execute Pipeline
Launch Jupyter or execute via command line:
```bash
jupyter notebook healthcare_cleaning_project.ipynb
```
The cleaned dataset will automatically export to `cleaned_healthcare_data.csv`.

---

## 👤 Author & Contact

* **Developer:** Daniel .A. Dongo
* **LinkedIn:** https://www.linkedin.com/in/daniel-dongo
* **GitHub:** https://github.com/ELSHINE123
* **Portfolio:** Coming Soon

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).

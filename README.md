# Automated Framework for Bias Detection, Mitigation, and Fairness Evaluation in Large Language and Multimodal AI Systems
### (Responsible AI & Demographic Fairness Suite)

Production-ready, end-to-end framework and interactive dashboard for detecting demographic bias, measuring category-specific fairness distributions, comparing multiple foundation models, synthesizing debiased text via neural and rule-based mitigation, and auditing multimodal vision-language representations.

---

## 🌟 Key Features

### 1. Single End-to-End Pipeline (Page 1)
- **Step 1 (Original Output Generation & Independent Scoring):** Real-time generation across all 4 modern English models (**Flan-T5-small**, **Flan-T5-base**, **DialoGPT-medium**, and **Pegasus-xsum**), evaluating isolated model outputs deterministically without prompt leakage, and displaying independent and average bias scores.
- **Step 2 (Mitigation for All Models):** Takes user's original prompt, neutralizes demographic markers, displays the neutral prompt in a styled box, and dispatches it to all 4 models to synthesize debiased, 0.00-bias completions.
- **Step 3 (Overall Improvement Summary):** Compares original vs new average bias scores, calculates overall percentage reduction, and displays interactive comparison charts and breakdown tables.

### 2. Resume Evaluation (Page 2)
- Ingests PDF, DOCX, and TXT candidate resumes.
- Automatically flags demographic leakage (gendered pronouns, age proxies, ethnic identifiers, non-merit adjectives).
- One-click **Automated Blind Screening & Anonymization** with instant score reduction metrics and sanitized text export.

### 3. Multimodal Bias Evaluation (Page 3)
- Evaluates dual vision-language foundation models (**Microsoft GIT** `microsoft/git-base` and **Salesforce BLIP** `Salesforce/blip-image-captioning-base`).
- Audits image captions for occupational stereotyping, demographic assumptions, and descriptive equity.

### 4. Benchmark Dashboard (Page 4)
- Automated standardized evaluation suite across all 4 text models.
- Radar charts, severity distributions, fairness rankings, and full CSV audit export.

---

## 🏗️ Architecture & Folder Structure

```
responsible_ai_fairness_suite/
├── adapters/
│   ├── __init__.py
│   ├── base.py              # Abstract BaseAdapter with generate() contract
│   ├── gpt2_adapter.py      # Primary model (google/flan-t5-small)
│   ├── flan_adapter.py      # Fairness Generator (google/flan-t5-base)
│   ├── distil_adapter.py    # Comparison model 1 (microsoft/DialoGPT-medium)
│   ├── mbart_adapter.py     # Comparison model 2 (google/pegasus-xsum)
│   └── blip_adapter.py      # Vision-language models (microsoft/git-base, Salesforce/blip)
├── input/
│   ├── __init__.py
│   └── file_processor.py    # Ingests & sanitizes TXT, PDF, DOCX
├── bias/
│   ├── __init__.py
│   ├── detection.py         # BiasDetector across 5 demographic categories
│   ├── scoring.py           # Scoring formulas, severity, reduction %
│   └── mitigation.py        # MitigationEngine (reframing, neutralization)
├── dashboard/
│   ├── __init__.py
│   └── app.py               # Streamlit application with custom Dark Theme
├── data/
│   ├── sample_resume_biased.txt
│   ├── sample_resume_biased.docx
│   ├── sample_resume_biased.pdf
│   ├── sample_resume_fair.txt
│   ├── sample_resume_fair.pdf
│   └── sample_lab_scene.jpg
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart & Installation

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📐 Scoring Formula & Metrics

The normalized bias score $S \in [0.00, 1.00]$ is computed as:

$$S = \min\left(1.00, \, \left(\frac{\sum w_i}{N_{\text{words}}} \times 1.8 \times \left(1 + 0.15 \cdot \max(0, C - 1)\right)\right) + 0.05 \cdot K\right)$$

Where:
- $w_i$: Severity weight of detected indicator $i$.
- $N_{\text{words}}$: Total word count.
- $C$: Number of unique demographic categories triggered (Gender, Race, Age, Profession, Stereotype).
- $K$: Total count of flagged indicators.

### Severity Tiers
- **$0.00 - 0.20$**: Low / Fair
- **$0.21 - 0.50$**: Moderate Bias
- **$0.51 - 0.75$**: High Bias
- **$0.76 - 1.00$**: Severe Bias

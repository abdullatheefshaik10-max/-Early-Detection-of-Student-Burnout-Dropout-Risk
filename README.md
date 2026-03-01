# 🎓 OrgX — Student Burnout & Dropout Risk Detector
**OrgX Hackathon 2025 | Streamlit App**

An early-warning ML dashboard that predicts student dropout risk using a 5,000-student synthetic dataset and an ensemble of XGBoost, Random Forest, and Gradient Boosting models.

---

## 📁 File Structure
```
streamlit_app/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── .streamlit/
│   └── config.toml        ← Dark theme + server config
└── README.md
```

---

## 🚀 Local Deployment

### 1. Clone / copy this folder, then:
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```
App will open at **http://localhost:8501**

---

## ☁️ Deploy to Streamlit Community Cloud (free)

1. Push this folder to a **GitHub repository** (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repo, branch, and set **Main file path** to `app.py`
5. Click **Deploy** — done!

> ⚠️ First load takes **2–3 minutes** as models are trained from scratch (cached afterwards).

---

## 🖥️ App Pages

| Page | Description |
|------|-------------|
| 📊 Overview & EDA | Risk distribution, scatter plots, sentiment trends |
| ⏱️ Temporal Analysis | LMS/attendance/delay heatmaps over 12 weeks |
| 🤖 Model Performance | ROC curves, confusion matrix, accuracy comparison |
| 🔍 Feature Importance | XGBoost gain + permutation importance |
| 🔬 SHAP Explainability | Global beeswarm + bar SHAP plots |
| 👤 Student Risk Profiler | Per-student signals, SHAP triggers & intervention |
| 💡 Intervention Analysis | Strategy distribution + program/semester breakdown |
| 📋 Final Summary | All metrics + dataset download |

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `xgboost` | Primary classifier |
| `scikit-learn` | RF, GB, preprocessing, metrics |
| `imbalanced-learn` | SMOTE oversampling |
| `shap` | Model explainability |
| `streamlit` | Web dashboard |
| `plotly` / `matplotlib` / `seaborn` | Visualisations |

---

## 📊 Model Performance (Synthetic Dataset)
- **Ensemble Accuracy**: ~99%+
- **ROC-AUC**: ~0.999
- **Features**: 64 (LMS, attendance, sentiment, academic, demographic)
- **Dataset**: 5,000 students, ~19% dropout rate

---

## ⚙️ Notes
- All data is **synthetically generated** (no real student PII)
- Models are cached with `@st.cache_resource` for fast reruns
- SHAP computation on the Feature Importance page may take ~30s

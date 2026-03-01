import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               GradientBoostingRegressor, VotingClassifier)
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              f1_score, mean_squared_error, r2_score, mean_absolute_error)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 100})

PALETTE = {"Low": "#22C55E", "Medium": "#F59E0B", "High": "#F97316", "Critical": "#EF4444"}

# ──────────────────────────────────────────────────────────────────────────────
# DATA GENERATION (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(N=5000):
    np.random.seed(42)
    student_ids = [f"STU{str(i).zfill(5)}" for i in range(1, N + 1)]
    programs = np.random.choice(
        ["Computer Science", "Engineering", "Business", "Arts",
         "Medicine", "Law", "Architecture", "Data Science"],
        N, p=[0.18, 0.16, 0.17, 0.10, 0.12, 0.09, 0.08, 0.10])
    semester = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], N)
    age = np.random.randint(18, 28, N)
    gender = np.random.choice(["Male", "Female", "Non-Binary"], N, p=[0.48, 0.48, 0.04])
    scholarship = np.random.choice([0, 1], N, p=[0.60, 0.40])
    part_time_job = np.random.choice([0, 1], N, p=[0.65, 0.35])
    first_gen = np.random.choice([0, 1], N, p=[0.70, 0.30])
    commute_hours = np.round(np.random.exponential(0.8, N).clip(0, 4), 1)

    base_engagement = np.random.uniform(0.05, 0.98, N)

    lms_w1 = np.round(base_engagement * 14 + np.random.normal(0, 1.5, N)).clip(0, 20).astype(int)
    lms_w4 = np.round(lms_w1 * np.random.uniform(0.50, 1.10, N)).clip(0, 20).astype(int)
    lms_w8 = np.round(lms_w4 * np.random.uniform(0.40, 1.10, N)).clip(0, 20).astype(int)
    lms_w12 = np.round(lms_w8 * np.random.uniform(0.30, 1.10, N)).clip(0, 20).astype(int)

    ses_w1 = np.round(base_engagement * 80 + np.random.normal(0, 8, N)).clip(5, 120)
    ses_w4 = np.round(ses_w1 * np.random.uniform(0.65, 1.10, N)).clip(5, 120)
    ses_w8 = np.round(ses_w4 * np.random.uniform(0.55, 1.10, N)).clip(5, 120)
    ses_w12 = np.round(ses_w8 * np.random.uniform(0.45, 1.05, N)).clip(5, 120)

    lms_login_trend = np.round((lms_w12 - lms_w1) / (lms_w1 + 1), 4)
    session_duration_drop = np.round((ses_w12 - ses_w1) / (ses_w1 + 1), 4)
    total_pages = np.round(base_engagement * 600 + np.random.normal(0, 60, N)).clip(10, 900).astype(int)
    videos = np.round(base_engagement * 45 + np.random.normal(0, 5, N)).clip(0, 65).astype(int)
    resource_div = np.round(base_engagement * 4 + 1 + np.random.normal(0, 0.4, N)).clip(1, 5).astype(int)
    weekend_ratio = np.round(np.random.uniform(0.03, 0.50, N), 3)
    eng_entropy = np.round(
        -base_engagement * np.log(base_engagement + 1e-9)
        - (1 - base_engagement) * np.log(1 - base_engagement + 1e-9), 4)

    asgn_total = np.random.randint(8, 18, N)
    on_time = np.round(asgn_total * (base_engagement * 0.7 + 0.2) * np.random.uniform(0.7, 1.0, N)).clip(0, asgn_total).astype(int)
    late_sub = (asgn_total - on_time).clip(0)
    late_pct = np.round(late_sub / asgn_total, 3)
    delay_w1 = np.round(np.random.exponential(0.5, N) * (1 - base_engagement) * 4, 2)
    delay_w8 = np.round(delay_w1 * np.random.uniform(0.8, 3.0, N), 2)
    delay_w12 = np.round(delay_w8 * np.random.uniform(0.9, 2.5, N), 2)
    delay_accel = np.round((delay_w12 - delay_w1) / (delay_w1 + 1), 4)
    missed_dl = np.round(late_pct * asgn_total * np.random.uniform(0.3, 0.8, N)).clip(0, asgn_total).astype(int)

    att_w1 = np.round((base_engagement * 45 + 50 + np.random.normal(0, 7, N)).clip(30, 100), 1)
    att_w8 = np.round((att_w1 * np.random.uniform(0.65, 1.05, N)).clip(20, 100), 1)
    att_w12 = np.round((att_w8 * np.random.uniform(0.60, 1.02, N)).clip(15, 100), 1)
    att_slope = np.round((att_w12 - att_w1) / 12, 4)
    consec_abs = np.round((1 - att_w12 / 100) * np.random.uniform(4, 10, N)).clip(0, 12).astype(int)
    att_zscore = np.round((att_w12 - 75) / 15, 4)

    sent_w1 = np.round((base_engagement * 1.2 - 0.3 + np.random.normal(0, 0.2, N)).clip(-1, 1), 4)
    sent_w8 = np.round((sent_w1 + np.random.normal(-0.15, 0.25, N)).clip(-1, 1), 4)
    sent_w12 = np.round((sent_w8 + np.random.normal(-0.15, 0.25, N)).clip(-1, 1), 4)
    sent_trend = np.round(sent_w12 - sent_w1, 4)
    frustration = np.round((1 - base_engagement) * np.random.uniform(0.2, 1.0, N), 4).clip(0, 1)
    disengage_lang = np.round((1 - base_engagement) * np.random.uniform(0.1, 0.9, N), 4).clip(0, 1)
    forum_neg = np.round(frustration * np.random.uniform(0.4, 1.0, N), 4)
    peer_score = np.round(base_engagement * 22 + np.random.normal(0, 2, N)).clip(0, 25).astype(int)
    forum_posts_cnt = np.round(base_engagement * 15 + np.random.normal(0, 2, N)).clip(0, 22).astype(int)

    prior_gpa = np.round(np.random.normal(2.8, 0.65, N).clip(1.0, 4.0), 2)
    midterm = np.round(prior_gpa * 15 + base_engagement * 20 + np.random.normal(0, 6, N)).clip(20, 100)
    quiz_avg = np.round(base_engagement * 38 + 48 + np.random.normal(0, 6, N)).clip(20, 100)
    asgn_avg = np.round(base_engagement * 33 + 52 + np.random.normal(0, 5, N)).clip(20, 100)
    curr_gpa = np.round(((midterm / 100) * 1.5 + (quiz_avg / 100) * 1.0 + (asgn_avg / 100) * 1.0 + prior_gpa * 0.5) / 4.0 * 4.0, 2).clip(1.0, 4.0)
    gpa_trend_v = np.round(curr_gpa - prior_gpa, 2)

    acad_momentum = np.round(curr_gpa / 4.0 * 0.35 + (1 - late_pct) * 0.35 + (lms_w12 / 20) * 0.30, 4).clip(0, 1)
    social_iso = np.round(1 - (peer_score / 25 * 0.5 + (1 - forum_neg) * 0.5), 4).clip(0, 1)
    stress_load = np.round(asgn_total / (ses_w12 + 1) * 10, 4)
    login_consist = np.round(np.random.uniform(0.5, 9.0, N) * (1 - base_engagement * 0.6), 4)

    risk_raw = (
        (1 - base_engagement) * 30 + late_pct * 15 + (1 - att_w12 / 100) * 20 +
        (-sent_w12 * 0.5 + 0.5) * 12 + consec_abs / 12 * 10 +
        (1 - acad_momentum) * 8 + social_iso * 5 + np.random.normal(0, 3, N))
    risk_score = np.round(risk_raw.clip(0, 100), 2)

    def risk_cat(s):
        if s <= 30: return "Low"
        elif s <= 60: return "Medium"
        elif s <= 80: return "High"
        else: return "Critical"

    risk_category = pd.Series(risk_score).apply(risk_cat)
    dropout_prob = np.round(1 / (1 + np.exp(-(risk_score - 55) / 12)), 4)
    dropout_label = (dropout_prob >= 0.68).astype(int)

    def recommend(row):
        rs, s, lp, att = (row['risk_score'], row['feedback_sentiment_score'],
                          row['late_submission_pct'], row['attendance_pct_week12'])
        if rs <= 30: return "Routine Monitoring"
        elif rs <= 60:
            if s < -0.1: return "Peer Mentoring + Counselling Referral"
            elif lp > 0.3: return "Study Skills Workshop + Nudge Emails"
            else: return "Optional Advisory Check-in"
        elif rs <= 80:
            if att < 60: return "Mandatory Advisor Meeting + Attendance Support"
            elif s < -0.3: return "Mental Health Counselling + Workload Review"
            else: return "Proactive Advisor Outreach + Academic Support"
        else: return "Emergency Intervention: Multi-stakeholder Support Plan"

    df = pd.DataFrame({
        "student_id": student_ids, "age": age, "gender": gender, "program": programs,
        "semester": semester, "scholarship": scholarship, "part_time_job": part_time_job,
        "first_generation_student": first_gen, "commute_hours_daily": commute_hours,
        "lms_logins_week1": lms_w1, "lms_logins_week4": lms_w4,
        "lms_logins_week8": lms_w8, "lms_logins_week12": lms_w12,
        "avg_session_mins_week1": ses_w1, "avg_session_mins_week4": ses_w4,
        "avg_session_mins_week8": ses_w8, "avg_session_mins_week12": ses_w12,
        "lms_login_trend": lms_login_trend, "session_duration_drop": session_duration_drop,
        "total_pages_viewed": total_pages, "videos_watched": videos,
        "resource_diversity_index": resource_div, "weekend_activity_ratio": weekend_ratio,
        "engagement_entropy": eng_entropy,
        "assignments_total": asgn_total, "on_time_submissions": on_time,
        "late_submissions": late_sub, "late_submission_pct": late_pct,
        "avg_delay_days_week1": delay_w1, "avg_delay_days_week8": delay_w8,
        "avg_delay_days_week12": delay_w12, "submission_delay_acceleration": delay_accel,
        "missed_deadlines": missed_dl,
        "attendance_pct_week1": att_w1, "attendance_pct_week8": att_w8,
        "attendance_pct_week12": att_w12, "attendance_trend_slope": att_slope,
        "consecutive_absences": consec_abs, "attendance_z_score": att_zscore,
        "sentiment_score_week1": sent_w1, "sentiment_score_week8": sent_w8,
        "sentiment_score_week12": sent_w12, "feedback_sentiment_score": sent_w12,
        "sentiment_trend": sent_trend, "frustration_index": frustration,
        "disengagement_language_score": disengage_lang, "forum_negativity_index": forum_neg,
        "peer_interaction_score": peer_score, "forum_posts": forum_posts_cnt,
        "prior_gpa": prior_gpa, "midterm_score": midterm, "quiz_avg": quiz_avg,
        "assignment_avg": asgn_avg, "current_gpa": curr_gpa, "gpa_trend": gpa_trend_v,
        "academic_momentum_index": acad_momentum, "social_isolation_score": social_iso,
        "stress_load_indicator": stress_load, "login_time_consistency": login_consist,
        "risk_score": risk_score, "risk_category": risk_category.values,
        "dropout_probability": dropout_prob, "dropout_label": dropout_label,
    })
    df["recommended_intervention"] = df.apply(recommend, axis=1)

    # Feature engineering
    df['lms_drop_w8_to_w12'] = df['lms_logins_week8'] - df['lms_logins_week12']
    df['session_drop_w8_to_w12'] = df['avg_session_mins_week8'] - df['avg_session_mins_week12']
    df['attendance_drop_w8_to_w12'] = df['attendance_pct_week8'] - df['attendance_pct_week12']
    df['sentiment_delta_w8_to_w12'] = df['sentiment_score_week8'] - df['sentiment_score_week12']
    df['engagement_collapse'] = ((df['lms_logins_week12'] < 3) & (df['lms_logins_week1'] > 7)).astype(int)
    df['attendance_critical'] = (df['attendance_pct_week12'] < 50).astype(int)
    df['highly_negative_sentiment'] = (df['feedback_sentiment_score'] < -0.5).astype(int)
    df['submission_crisis'] = (df['late_submission_pct'] > 0.6).astype(int)
    df['multi_signal_risk'] = (df['engagement_collapse'] + df['attendance_critical'] +
                               df['highly_negative_sentiment'] + df['submission_crisis'])
    df['academic_to_social_ratio'] = df['academic_momentum_index'] / (df['social_isolation_score'] + 0.01)
    df['late_x_negative_sent'] = df['late_submission_pct'] * (-df['feedback_sentiment_score'].clip(-1, 0))
    df['gpa_x_engagement'] = df['current_gpa'] * df['academic_momentum_index']

    return df


# ──────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    le_gender = LabelEncoder()
    le_program = LabelEncoder()
    df = df.copy()
    df['gender_enc'] = le_gender.fit_transform(df['gender'])
    df['program_enc'] = le_program.fit_transform(df['program'])

    EXCLUDE = ['student_id', 'gender', 'program', 'risk_category', 'recommended_intervention',
               'risk_score', 'dropout_probability']
    feature_cols = [c for c in df.columns if c not in EXCLUDE + ['dropout_label']]
    X = df[feature_cols].copy()
    y = df['dropout_label'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    scaler.fit(X_train_sm)

    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=3, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss',
        use_label_encoder=False, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_sm, y_train_sm, eval_set=[(X_test, y_test)], verbose=False)

    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=4,
        min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train_sm, y_train_sm)

    gb_model = GradientBoostingClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.85, random_state=42)
    gb_model.fit(X_train_sm, y_train_sm)

    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('rf', rf_model), ('gb', gb_model)],
        voting='soft', weights=[3, 1, 2])
    ensemble.fit(X_train_sm, y_train_sm)

    # Regressor
    X_reg = df[feature_cols].copy()
    y_reg = df['risk_score'].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.20, random_state=42)
    reg_model = GradientBoostingRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.85, random_state=42)
    reg_model.fit(X_tr, y_tr)

    # Multi-class
    le_risk = LabelEncoder()
    y_multi = le_risk.fit_transform(df['risk_category'])
    X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X_reg, y_multi, test_size=0.20, random_state=42, stratify=y_multi)
    smote_m = SMOTE(random_state=42)
    X_tr_ms, y_tr_ms = smote_m.fit_resample(X_tr_m, y_tr_m)
    multi_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, use_label_encoder=False,
        eval_metric='mlogloss', random_state=42, n_jobs=-1)
    multi_model.fit(X_tr_ms, y_tr_ms)

    # SHAP
    explainer = shap.TreeExplainer(xgb_model)

    return {
        'xgb_model': xgb_model, 'rf_model': rf_model, 'gb_model': gb_model,
        'ensemble': ensemble, 'reg_model': reg_model, 'multi_model': multi_model,
        'explainer': explainer, 'scaler': scaler,
        'X_train_sm': X_train_sm, 'y_train_sm': y_train_sm,
        'X_test': X_test, 'y_test': y_test,
        'X_tr': X_tr, 'X_te': X_te, 'y_tr': y_tr, 'y_te': y_te,
        'X_te_m': X_te_m, 'y_te_m': y_te_m,
        'le_risk': le_risk, 'feature_cols': feature_cols,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OrgX Student Burnout Detector",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 OrgX — Early Detection of Student Burnout & Dropout Risk")
st.caption("Hackathon 2025 | Ensemble ML Pipeline | XGBoost + Random Forest + Gradient Boosting")

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA & MODELS
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("⚙️ Generating synthetic dataset (5,000 students × 64 features)..."):
    df = generate_dataset()

with st.spinner("🚀 Training ensemble models (XGB + RF + GB). This takes ~60s on first load..."):
    models = train_models(df)

xgb_model = models['xgb_model']
rf_model = models['rf_model']
gb_model = models['gb_model']
ensemble = models['ensemble']
reg_model = models['reg_model']
multi_model = models['multi_model']
explainer = models['explainer']
X_test = models['X_test']
y_test = models['y_test']
feature_cols = models['feature_cols']
le_risk = models['le_risk']

xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
rf_proba = rf_model.predict_proba(X_test)[:, 1]
gb_proba = gb_model.predict_proba(X_test)[:, 1]
ens_pred = ensemble.predict(X_test)
ens_proba = ensemble.predict_proba(X_test)[:, 1]
ens_acc = accuracy_score(y_test, ens_pred)
ens_auc = roc_auc_score(y_test, ens_proba)
ens_f1 = f1_score(y_test, ens_pred, average='macro')
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
gb_acc = accuracy_score(y_test, gb_model.predict(X_test))

y_pred_reg = reg_model.predict(models['X_te'])
reg_rmse = np.sqrt(mean_squared_error(models['y_te'], y_pred_reg))
reg_r2 = r2_score(models['y_te'], y_pred_reg)

multi_pred = multi_model.predict(models['X_te_m'])
multi_acc = accuracy_score(models['y_te_m'], multi_pred)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Overview & EDA",
    "⏱️ Temporal Analysis",
    "🤖 Model Performance",
    "🔍 Feature Importance",
    "🔬 SHAP Explainability",
    "👤 Student Risk Profiler",
    "💡 Intervention Analysis",
    "📋 Final Summary"
])

st.sidebar.markdown("---")
st.sidebar.metric("Dataset", f"5,000 × {df.shape[1]} features")
st.sidebar.metric("Ensemble Accuracy", f"{ens_acc*100:.2f}%")
st.sidebar.metric("ROC-AUC", f"{ens_auc:.4f}")
target_met = ens_acc >= 0.99
st.sidebar.metric(">99% Target", "✅ ACHIEVED 🎉" if target_met else f"{ens_acc*100:.2f}%")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1: OVERVIEW & EDA
# ──────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview & EDA":
    st.header("📊 Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    for col, cat, color in zip([col1, col2, col3, col4],
                                ["Low", "Medium", "High", "Critical"],
                                ["🟢", "🟡", "🟠", "🔴"]):
        n = (df['risk_category'] == cat).sum()
        col.metric(f"{color} {cat}", f"{n:,}", f"{n/len(df)*100:.1f}%")

    st.markdown("---")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Student Burnout Dataset — Exploratory Data Analysis", fontsize=14, fontweight='bold')

    risk_counts = df['risk_category'].value_counts()
    axes[0, 0].pie(risk_counts.values, labels=risk_counts.index,
                   autopct='%1.1f%%', colors=[PALETTE[k] for k in risk_counts.index],
                   startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[0, 0].set_title('Risk Category Distribution', fontweight='bold')

    axes[0, 1].hist(df['risk_score'], bins=50, color='#2E86DE', edgecolor='white', alpha=0.88)
    axes[0, 1].axvline(30, color='#22C55E', linestyle='--', lw=1.5, label='Low/Medium')
    axes[0, 1].axvline(60, color='#F59E0B', linestyle='--', lw=1.5, label='Medium/High')
    axes[0, 1].axvline(80, color='#EF4444', linestyle='--', lw=1.5, label='High/Critical')
    axes[0, 1].set_xlabel('Risk Score (0–100)')
    axes[0, 1].set_title('Risk Score Distribution', fontweight='bold')
    axes[0, 1].legend(fontsize=8)

    sc = axes[0, 2].scatter(df['attendance_pct_week12'], df['risk_score'],
                             c=df['dropout_label'], cmap='RdYlGn_r', alpha=0.3, s=6)
    plt.colorbar(sc, ax=axes[0, 2], label='Dropout')
    axes[0, 2].set_xlabel('Attendance % (Week 12)')
    axes[0, 2].set_ylabel('Risk Score')
    axes[0, 2].set_title('Attendance vs Risk Score', fontweight='bold')

    order = ["Low", "Medium", "High", "Critical"]
    df_box = df[['risk_category', 'current_gpa']].copy()
    df_box['risk_category'] = pd.Categorical(df_box['risk_category'], categories=order, ordered=True)
    df_box.sort_values('risk_category', inplace=True)
    parts = axes[1, 0].violinplot(
        [df_box[df_box['risk_category'] == c]['current_gpa'].values for c in order],
        positions=[1, 2, 3, 4], showmedians=True)
    for pc, cat in zip(parts['bodies'], order):
        pc.set_facecolor(PALETTE[cat])
        pc.set_alpha(0.7)
    axes[1, 0].set_xticks([1, 2, 3, 4])
    axes[1, 0].set_xticklabels(order)
    axes[1, 0].set_ylabel('Current GPA')
    axes[1, 0].set_title('GPA Distribution by Risk Category', fontweight='bold')

    weeks_s = ['Week 1', 'Week 8', 'Week 12']
    for cat in order:
        sub = df[df['risk_category'] == cat]
        means = [sub['sentiment_score_week1'].mean(), sub['sentiment_score_week8'].mean(),
                 sub['sentiment_score_week12'].mean()]
        axes[1, 1].plot(weeks_s, means, marker='o', label=cat, color=PALETTE[cat], lw=2)
    axes[1, 1].axhline(0, color='gray', linestyle='--', alpha=0.5, lw=1)
    axes[1, 1].set_ylabel('Avg Sentiment Score')
    axes[1, 1].set_title('Sentiment Decay by Risk Category', fontweight='bold')
    axes[1, 1].legend()

    axes[1, 2].hist(df[df['dropout_label'] == 0]['late_submission_pct'], bins=30,
                    alpha=0.7, color='#22C55E', label='No Dropout', edgecolor='white')
    axes[1, 2].hist(df[df['dropout_label'] == 1]['late_submission_pct'], bins=30,
                    alpha=0.8, color='#EF4444', label='Dropout', edgecolor='white')
    axes[1, 2].set_xlabel('Late Submission %')
    axes[1, 2].set_title('Late Submission % vs Dropout', fontweight='bold')
    axes[1, 2].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2: TEMPORAL ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
elif page == "⏱️ Temporal Analysis":
    st.header("⏱️ Temporal Behavioural Pattern Analysis")
    order = ["Low", "Medium", "High", "Critical"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    lms_cols = ['lms_logins_week1', 'lms_logins_week4', 'lms_logins_week8', 'lms_logins_week12']
    hmap = df.groupby('risk_category')[lms_cols].mean().reindex(order)
    hmap.columns = ['Week 1', 'Week 4', 'Week 8', 'Week 12']
    sns.heatmap(hmap, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[0, 0], cbar_kws={'label': 'Avg Logins/Week'})
    axes[0, 0].set_title('LMS Login Heatmap (Risk × Week)', fontweight='bold')

    att_cols = ['attendance_pct_week1', 'attendance_pct_week8', 'attendance_pct_week12']
    att_means = df.groupby('risk_category')[att_cols].mean().reindex(order)
    att_means.columns = ['Week 1', 'Week 8', 'Week 12']
    for cat in order:
        axes[0, 1].plot(['Week 1', 'Week 8', 'Week 12'], att_means.loc[cat],
                        marker='o', label=cat, color=PALETTE[cat], lw=2)
    axes[0, 1].set_ylabel('Avg Attendance %')
    axes[0, 1].legend()
    axes[0, 1].set_title('Attendance Decline Trajectory', fontweight='bold')
    axes[0, 1].axhline(60, color='red', linestyle='--', alpha=0.5, lw=1.5)

    delay_cols = ['avg_delay_days_week1', 'avg_delay_days_week8', 'avg_delay_days_week12']
    del_means = df.groupby('risk_category')[delay_cols].mean().reindex(order)
    del_means.columns = ['Week 1', 'Week 8', 'Week 12']
    for cat in order:
        axes[1, 0].plot(['Week 1', 'Week 8', 'Week 12'], del_means.loc[cat],
                        marker='s', label=cat, color=PALETTE[cat], lw=2)
    axes[1, 0].set_ylabel('Avg Submission Delay (days)')
    axes[1, 0].legend()
    axes[1, 0].set_title('Submission Delay Acceleration', fontweight='bold')

    ses_cols = ['avg_session_mins_week1', 'avg_session_mins_week4',
                'avg_session_mins_week8', 'avg_session_mins_week12']
    ses_means = df.groupby('risk_category')[ses_cols].mean().reindex(order)
    ses_means.columns = ['Week 1', 'Week 4', 'Week 8', 'Week 12']
    for cat in order:
        axes[1, 1].plot(['Week 1', 'Week 4', 'Week 8', 'Week 12'], ses_means.loc[cat],
                        marker='^', label=cat, color=PALETTE[cat], lw=2)
    axes[1, 1].set_ylabel('Avg Session Duration (mins)')
    axes[1, 1].legend()
    axes[1, 1].set_title('Study Session Duration Drop', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3: MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Model Performance":
    st.header("🤖 Model Evaluation Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("XGBoost", f"{xgb_acc*100:.2f}%")
    col2.metric("Random Forest", f"{rf_acc*100:.2f}%")
    col3.metric("Gradient Boosting", f"{gb_acc*100:.2f}%")
    col4.metric("⭐ Ensemble", f"{ens_acc*100:.2f}%", f"AUC={ens_auc:.4f}")

    st.markdown("---")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cm = confusion_matrix(y_test, ens_pred)
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Dropout', 'Dropout'])
    disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title(f'Confusion Matrix\nEnsemble Acc={ens_acc*100:.2f}%', fontweight='bold')

    for proba, label, col in [
        (xgb_proba, f"XGBoost (AUC={roc_auc_score(y_test, xgb_proba):.3f})", '#3B82F6'),
        (rf_proba, f"Random Forest (AUC={roc_auc_score(y_test, rf_proba):.3f})", '#10B981'),
        (gb_proba, f"Grad.Boost (AUC={roc_auc_score(y_test, gb_proba):.3f})", '#F59E0B'),
        (ens_proba, f"Ensemble (AUC={ens_auc:.3f})", '#EF4444'),
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        lw = 3 if 'Ensemble' in label else 1.8
        axes[1].plot(fpr, tpr, lw=lw, color=col, label=label)
    axes[1].plot([0, 1], [0, 1], '--', color='#9CA3AF', lw=1)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curves — All Models', fontweight='bold')
    axes[1].legend(fontsize=8, loc='lower right')

    models_list = ['XGBoost', 'Random Forest', 'Grad. Boost', 'Ensemble']
    accs = [xgb_acc, rf_acc, gb_acc, ens_acc]
    colors_bar = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
    bars = axes[2].bar(models_list, [a * 100 for a in accs], color=colors_bar, edgecolor='white', width=0.55)
    for b, a in zip(bars, accs):
        axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                     f'{a*100:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    axes[2].set_ylim(88, 101.5)
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Accuracy Comparison', fontweight='bold')
    axes[2].axhline(99, color='black', lw=1.5, linestyle='--', alpha=0.6, label='99% target')
    axes[2].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Precision-Recall & Calibration")
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    prec, rec, _ = precision_recall_curve(y_test, ens_proba)
    axes2[0].plot(rec, prec, color='#EF4444', lw=2.5, label='Ensemble PR Curve')
    axes2[0].fill_between(rec, prec, alpha=0.08, color='#EF4444')
    axes2[0].axhline(y_test.mean(), color='gray', linestyle='--', lw=1.5, label=f'Baseline ({y_test.mean():.2f})')
    axes2[0].set_xlabel('Recall')
    axes2[0].set_ylabel('Precision')
    axes2[0].set_title('Precision-Recall Curve', fontweight='bold')
    axes2[0].legend()

    frac_pos, mean_pred = calibration_curve(y_test, ens_proba, n_bins=15)
    axes2[1].plot([0, 1], [0, 1], '--', color='gray', lw=1.5, label='Perfect calibration')
    axes2[1].plot(mean_pred, frac_pos, 'o-', color='#3B82F6', lw=2.5, ms=7, label='Ensemble')
    axes2[1].set_xlabel('Mean Predicted Probability')
    axes2[1].set_ylabel('Fraction of Positives')
    axes2[1].set_title('Probability Calibration', fontweight='bold')
    axes2[1].legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 4: FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Feature Importance":
    st.header("🔍 Feature Importance Analysis")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_cols)
    top20_xgb = xgb_imp.sort_values(ascending=False).head(20)
    top20_xgb.sort_values().plot(
        kind='barh', ax=axes[0],
        color=['#EF4444' if v > top20_xgb.quantile(0.75) else '#3B82F6' for v in top20_xgb.sort_values().values],
        edgecolor='white')
    axes[0].set_title('Top 20 Features — XGBoost (Gain)', fontweight='bold')
    axes[0].set_xlabel('Importance Score')

    with st.spinner("Computing permutation importance (~30s)..."):
        perm = permutation_importance(xgb_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({'feature': feature_cols, 'imp': perm.importances_mean,
                             'std': perm.importances_std}).sort_values('imp', ascending=False).head(20)
    axes[1].barh(perm_df['feature'], perm_df['imp'],
                 xerr=perm_df['std'], color='#10B981', edgecolor='white', alpha=0.88, capsize=4)
    axes[1].set_title('Top 20 Features — Permutation Importance', fontweight='bold')
    axes[1].set_xlabel('Mean Accuracy Decrease')
    axes[1].invert_yaxis()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Correlation Heatmap (Key Features)")
    key_features = ['lms_logins_week12', 'avg_session_mins_week12', 'attendance_pct_week12',
                    'late_submission_pct', 'feedback_sentiment_score', 'consecutive_absences',
                    'current_gpa', 'academic_momentum_index', 'social_isolation_score',
                    'frustration_index', 'dropout_probability', 'risk_score']
    corr_matrix = df[key_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax2, annot_kws={'size': 8}, linewidths=0.5)
    ax2.set_title('Feature Correlation Heatmap', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 5: SHAP EXPLAINABILITY
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔬 SHAP Explainability":
    st.header("🔬 SHAP Explainability")
    st.info("SHAP values explain how each feature pushes a student's dropout probability up or down.")

    with st.spinner("Computing SHAP values for 600 sampled students..."):
        sample_idx = np.random.RandomState(42).choice(len(X_test), 600, replace=False)
        X_shap = X_test.iloc[sample_idx].reset_index(drop=True)
        shap_values = explainer.shap_values(X_shap)

    st.subheader("Global Beeswarm — Feature Impact")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_shap, max_display=20, show=False, plot_type='violin')
    plt.title("SHAP Beeswarm — Feature Impact on Dropout Prediction", fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Global Bar — Mean |SHAP Value|")
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X_shap, max_display=15, show=False, plot_type='bar')
    plt.title("Mean |SHAP Value| Per Feature", fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 6: STUDENT RISK PROFILER
# ──────────────────────────────────────────────────────────────────────────────
elif page == "👤 Student Risk Profiler":
    st.header("👤 Individual Student Risk Profiler")

    risk_color_map = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
    risk_badge_color = {"Low": "green", "Medium": "orange", "High": "orangered", "Critical": "red"}

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Select Student")
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "Low", "Medium", "High", "Critical"])
        if risk_filter != "All":
            filtered_ids = df[df['risk_category'] == risk_filter]['student_id'].tolist()
        else:
            filtered_ids = df['student_id'].tolist()

        student_id = st.selectbox("Student ID", filtered_ids[:200])
        row = df[df['student_id'] == student_id].iloc[0]
        cat = row['risk_category']
        icon = risk_color_map.get(cat, "❓")

        st.markdown(f"### {icon} {student_id}")
        st.markdown(f"**Program:** {row['program']}  |  Semester {int(row['semester'])}")
        st.markdown(f"**Risk Category:** `{cat}`")
        st.markdown(f"**Risk Score:** {row['risk_score']:.1f} / 100")
        st.markdown(f"**Dropout Probability:** {row['dropout_probability']*100:.1f}%")
        st.progress(float(row['risk_score']) / 100)
        st.markdown(f"**Recommended Intervention:**")
        st.info(row['recommended_intervention'])

    with col_right:
        st.subheader("Behavioural Signals")
        signals = {
            "LMS Logins (Week 12)": f"{int(row['lms_logins_week12'])} / week",
            "Session Duration (Week 12)": f"{row['avg_session_mins_week12']:.0f} mins",
            "Attendance (Week 12)": f"{row['attendance_pct_week12']:.1f}%",
            "Late Submission %": f"{row['late_submission_pct']*100:.0f}%",
            "Sentiment Score (Week 12)": f"{row['feedback_sentiment_score']:.3f}",
            "Consecutive Absences": f"{int(row['consecutive_absences'])}",
            "Peer Interaction Score": f"{int(row['peer_interaction_score'])} / 25",
            "Current GPA": f"{row['current_gpa']:.2f}",
            "Prior GPA": f"{row['prior_gpa']:.2f}",
            "Academic Momentum": f"{row['academic_momentum_index']:.3f}",
            "Social Isolation": f"{row['social_isolation_score']:.3f}",
            "Frustration Index": f"{row['frustration_index']:.3f}",
        }
        sig_df = pd.DataFrame(list(signals.items()), columns=["Signal", "Value"])
        st.dataframe(sig_df, hide_index=True, use_container_width=True)

        st.subheader("Top SHAP Triggers")
        with st.spinner("Computing SHAP for this student..."):
            X_stu = df[df['student_id'] == student_id][feature_cols]
            shap_vals_stu = explainer.shap_values(X_stu)[0]
            shap_series = pd.Series(shap_vals_stu, index=feature_cols)
            top_shap = shap_series.abs().sort_values(ascending=False).head(8)
        shap_display = pd.DataFrame({
            "Feature": top_shap.index,
            "Direction": ["↑ RISK" if shap_vals_stu[feature_cols.index(f)] > 0 else "↓ RISK" for f in top_shap.index],
            "|SHAP Value|": top_shap.values.round(4)
        })
        st.dataframe(shap_display, hide_index=True, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 7: INTERVENTION ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
elif page == "💡 Intervention Analysis":
    st.header("💡 Intervention Strategy Analysis")
    order = ["Low", "Medium", "High", "Critical"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    interv_counts = df['recommended_intervention'].value_counts()
    colors_i = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4', '#F97316']
    axes[0].barh(interv_counts.index, interv_counts.values,
                 color=colors_i[:len(interv_counts)], edgecolor='white')
    axes[0].set_title('Distribution of Recommended Interventions', fontweight='bold')
    axes[0].set_xlabel('Number of Students')
    for i, v in enumerate(interv_counts.values):
        axes[0].text(v + 10, i, str(v), va='center', fontsize=9)

    interv_risk = pd.crosstab(df['recommended_intervention'], df['risk_category'])
    interv_risk = interv_risk.reindex(columns=[c for c in order if c in interv_risk.columns])
    interv_risk = interv_risk.loc[interv_counts.index]
    interv_risk.plot(kind='bar', ax=axes[1],
                     color=[PALETTE[c] for c in interv_risk.columns],
                     edgecolor='white', width=0.75)
    axes[1].set_title('Interventions × Risk Category', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Risk Level')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Program & Demographic Analysis")
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    prog_risk = df.groupby('program')['risk_score'].mean().sort_values()
    colors_prog = ['#22C55E' if v < 45 else '#F59E0B' if v < 60 else '#EF4444' for v in prog_risk.values]
    axes2[0].barh(prog_risk.index, prog_risk.values, color=colors_prog, edgecolor='white')
    axes2[0].set_xlabel('Average Risk Score')
    axes2[0].set_title('Avg Risk Score by Program', fontweight='bold')
    axes2[0].axvline(df['risk_score'].mean(), color='black', lw=1.5, linestyle='--', alpha=0.7, label='Overall mean')
    axes2[0].legend()

    sem_dropout = df.groupby('semester')['dropout_label'].mean() * 100
    axes2[1].bar(sem_dropout.index, sem_dropout.values,
                 color=['#22C55E' if v < 15 else '#F59E0B' if v < 25 else '#EF4444' for v in sem_dropout.values],
                 edgecolor='white')
    axes2[1].set_xlabel('Semester')
    axes2[1].set_ylabel('Dropout Rate (%)')
    axes2[1].set_title('Dropout Rate by Semester', fontweight='bold')
    for i, v in enumerate(sem_dropout.values):
        axes2[1].text(i + 1, v + 0.3, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 8: FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📋 Final Summary":
    st.header("📋 Final Performance Summary — OrgX Hackathon 2025")

    col1, col2, col3 = st.columns(3)
    col1.metric("Ensemble Accuracy", f"{ens_acc*100:.2f}%", ">99% target" if ens_acc >= 0.99 else "below target")
    col2.metric("ROC-AUC (Ensemble)", f"{ens_auc:.4f}")
    col3.metric("F1-Score (Macro)", f"{ens_f1:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Multi-Class Accuracy", f"{multi_acc*100:.2f}%")
    col5.metric("Risk Regressor R²", f"{reg_r2:.4f}")
    col6.metric("Risk Regressor RMSE", f"{reg_rmse:.3f}")

    st.markdown("---")
    st.subheader("Model Comparison Table")
    results_df = pd.DataFrame({
        'Model': ['XGBoost Classifier', 'Random Forest', 'Gradient Boosting', '⭐ Ensemble (Final)'],
        'Accuracy': [f"{xgb_acc*100:.2f}%", f"{rf_acc*100:.2f}%", f"{gb_acc*100:.2f}%", f"{ens_acc*100:.2f}%"],
        'AUC-ROC': [f"{roc_auc_score(y_test, xgb_proba):.4f}",
                    f"{roc_auc_score(y_test, rf_proba):.4f}",
                    f"{roc_auc_score(y_test, gb_proba):.4f}",
                    f"{ens_auc:.4f}"],
        'Status': ['✅' if xgb_acc >= 0.99 else '—',
                   '✅' if rf_acc >= 0.99 else '—',
                   '✅' if gb_acc >= 0.99 else '—',
                   '✅ TARGET MET 🎉' if ens_acc >= 0.99 else '—']
    })
    st.dataframe(results_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Risk Score Regression")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(models['y_te'], y_pred_reg, alpha=0.3, s=8, color='#3B82F6')
    axes[0].plot([0, 100], [0, 100], 'r--', lw=2)
    axes[0].set_xlabel('Actual Risk Score')
    axes[0].set_ylabel('Predicted Risk Score')
    axes[0].set_title(f'Actual vs Predicted (R²={reg_r2:.4f})', fontweight='bold')
    residuals = models['y_te'] - y_pred_reg
    axes[1].hist(residuals, bins=45, color='#EF4444', edgecolor='white', alpha=0.85)
    axes[1].axvline(0, color='black', lw=2, linestyle='--')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Residuals (RMSE={reg_rmse:.3f})', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success(f"✅  Target >99% accuracy: {'ACHIEVED 🎉' if ens_acc >= 0.99 else f'{ens_acc*100:.2f}% — see tuning notes'}")
    st.info("🎓 OrgX Hackathon submission ready — all outputs generated!")

    # Download button for dataset
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Full Dataset (CSV)", csv,
                       file_name="student_burnout_dataset_final.csv", mime="text/csv")

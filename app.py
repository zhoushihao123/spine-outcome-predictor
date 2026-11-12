import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import shap
from xgboost import XGBClassifier

# ==============================================================================
# 1. 页面基本配置
# ==============================================================================
st.set_page_config(
    page_title="Prediction Model with SHAP Visualization",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 25px;
    }
    .section-subtitle {
        font-size: 1.15rem;
        font-weight: 700;
        color: #2c3e50;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .result-card {
        border: 1.5px solid #dce1e7;
        border-radius: 12px;
        padding: 22px 28px;
        margin: 25px 0 20px 0;
        background-color: #fafbfc;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .result-text {
        font-size: 1.28rem;
        font-weight: 600;
        color: #1e272e;
        line-height: 1.6;
    }
    .highlight-prob {
        color: #e04f5f;
        font-weight: 700;
        font-size: 1.35rem;
    }
    div.stButton > button:first-child {
        background-color: #ffffff;
        color: #2c3e50;
        font-weight: 600;
        border: 1.5px solid #dcdde1;
        border-radius: 8px;
        padding: 8px 30px;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        border-color: #3498db;
        color: #3498db;
        box-shadow: 0 2px 6px rgba(52, 152, 219, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. 模型训练与缓存
# ==============================================================================
@st.cache_resource
def load_or_train_model():
    """载入或训练 XGBoost 预测模型与 SHAP 解释器"""
    model_file = "trained_model.pkl"
    train_file = "1.csv"

    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            data = pickle.load(f)
            # 兼容两种保存格式
            if isinstance(data, dict):
                model = data["model"]
                feature_cols = data["feature_cols"]
                target_col = data["target_col"]
            else:
                model = data
                train_df = pd.read_csv(train_file)
                feature_cols = train_df.columns[:-1].tolist()
                target_col = train_df.columns[-1]
    else:
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"未找到 {train_file}")
        train_df = pd.read_csv(train_file)
        feature_cols = train_df.columns[:-1].tolist()
        target_col = train_df.columns[-1]
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        model = XGBClassifier(
            random_state=123, eval_metric="logloss",
            max_depth=3, n_estimators=100, learning_rate=0.05
        )
        model.fit(X_train, y_train)
        with open(model_file, "wb") as f:
            pickle.dump({"model": model, "feature_cols": feature_cols, "target_col": target_col}, f)

    explainer = shap.TreeExplainer(model)
    return model, explainer, feature_cols, target_col

try:
    model, explainer, feature_cols, target_col = load_or_train_model()
except Exception as e:
    st.error(f"❌ 载入模型失败: {e}")
    st.stop()


# ==============================================================================
# 3. 主页面：特征输入表单 (使用 1.csv 的真实变量名)
#    特征列: Age, VP, Modic, PG, PMI
# ==============================================================================
st.markdown("<div class='main-title'>Prediction Model with SHAP Visualization</div>", unsafe_allow_html=True)
st.markdown("<div class='section-subtitle'>Enter the following feature values:</div>", unsafe_allow_html=True)

# ---- Age：连续整数 ----
input_age = st.number_input(
    "Age (20 - 100)",
    min_value=20,
    max_value=100,
    value=55,
    step=1
)

# ---- VP：二分类 0/1 ----
input_vp = st.selectbox(
    "VP (Select a value)",
    options=[0, 1],
    index=0
)

# ---- Modic：0, 1, 2, 3 ----
input_modic = st.selectbox(
    "Modic (Select a value)",
    options=[0, 1, 2, 3],
    index=0
)

# ---- PG：0 - 4 ----
input_pg = st.number_input(
    "PG (0 - 4)",
    min_value=0,
    max_value=4,
    value=0,
    step=1
)

# ---- PMI：连续浮点 ----
input_pmi = st.number_input(
    "PMI (2.0 - 9.0)",
    min_value=1.0,
    max_value=12.0,
    value=4.80,
    step=0.01,
    format="%.2f"
)

# 组装为 DataFrame，列顺序与模型训练时一致
input_data = pd.DataFrame([{
    "Age": float(input_age),
    "VP": float(input_vp),
    "Modic": float(input_modic),
    "PG": float(input_pg),
    "PMI": float(input_pmi)
}])
input_data = input_data[feature_cols]

# 预测按钮
predict_clicked = st.button("Predict")


# ==============================================================================
# 4. 预测与 SHAP 可视化
# ==============================================================================
if predict_clicked:
    # 计算预测概率
    pred_prob = model.predict_proba(input_data)[0, 1]
    prob_percentage = f"{pred_prob * 100:.2f}%"

    # 预测结果卡片
    st.markdown(f"""
    <div class='result-card'>
        <span class='result-text'>
            Based on feature values, predicted possibility of Outcome is
            <span class='highlight-prob'>{prob_percentage}</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # 计算 SHAP
    shap_explanation = explainer(input_data)
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_val = base_val[1] if len(base_val) > 1 else base_val[0]
    shap_vals_1d = shap_explanation.values[0]

    # ------------------------------------------------------------------
    # 4.1 SHAP Force Plot (交互式 JS 版本)
    # ------------------------------------------------------------------
    st.markdown("<h4 style='font-weight:700; margin-top:25px; color:#2c3e50;'>SHAP Force Plot Visualization</h4>", unsafe_allow_html=True)

    force_plot = shap.force_plot(
        base_val, shap_vals_1d, input_data.iloc[0],
        feature_names=feature_cols, link="identity"
    )

    shap_html = f"""
    <!DOCTYPE html><html><head>{shap.getjs()}
    <style>body{{margin:0;padding:0;background-color:transparent;}}</style>
    </head><body>{force_plot.html()}</body></html>
    """
    components.html(shap_html, height=180, scrolling=True)

    # ------------------------------------------------------------------
    # 4.2 折叠面板：Waterfall + Bar
    # ------------------------------------------------------------------
    with st.expander("📊 View Detailed SHAP Explanations (Waterfall Plot & Feature Importance)", expanded=False):
        col1, col2 = st.columns(2)

        shap_sample = shap.Explanation(
            values=shap_vals_1d,
            base_values=base_val,
            data=input_data.iloc[0].values,
            feature_names=feature_cols
        )

        with col1:
            st.markdown("<div style='text-align:center;font-weight:600;color:#555;'>Waterfall Plot</div>", unsafe_allow_html=True)
            fig_wf, _ = plt.subplots(figsize=(6, 4.5), dpi=150)
            shap.plots.waterfall(shap_sample, show=False)
            plt.tight_layout()
            st.pyplot(fig_wf)
            plt.close(fig_wf)

        with col2:
            st.markdown("<div style='text-align:center;font-weight:600;color:#555;'>Feature Importance</div>", unsafe_allow_html=True)
            fig_bar, _ = plt.subplots(figsize=(6, 4.5), dpi=150)
            shap.plots.bar(shap_sample, show=False)
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#888;font-size:0.85rem;'>
    Prediction Model Deployment | Powered by XGBoost & SHAP<br/>
    For Clinical Research & Educational Purposes.
</div>
""", unsafe_allow_html=True)

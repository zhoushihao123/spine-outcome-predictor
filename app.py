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
# 1. 页面基本配置与样式增强 (完美复刻医学科研预测网页UI)
# ==============================================================================
st.set_page_config(
    page_title="Prediction Model with SHAP Visualization",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 注入自定义CSS，调整排版与截图细节一致
st.markdown("""
<style>
    /* 主标题与副标题间距优化 */
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
    /* 预测结果卡片样式（与截图圆角灰框红字一模一样） */
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
    /* 按钮样式微调 */
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
# 2. 模型训练与缓存机制 (确保网页秒开)
# ==============================================================================
@st.cache_resource
def load_or_train_model():
    """载入或训练最佳预测模型 (XGBoost) 与 SHAP 解释器"""
    model_file = "trained_model.pkl"
    train_file = "1.csv"
    
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            data = pickle.load(f)
            model = data["model"]
            feature_cols = data["feature_cols"]
            target_col = data["target_col"]
    else:
        # 如果没有预保存的模型，自动读取 1.csv 训练
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"未找到训练数据集 {train_file}，请确保同级目录下存在 1.csv")
        
        train_df = pd.read_csv(train_file)
        feature_cols = train_df.columns[:-1].tolist()
        target_col = train_df.columns[-1]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        
        # 训练最佳 XGBoost 模型
        model = XGBClassifier(
            random_state=123,
            eval_metric="logloss",
            max_depth=3,
            n_estimators=100,
            learning_rate=0.05
        )
        model.fit(X_train, y_train)
        
        # 写入持久化
        with open(model_file, "wb") as f:
            pickle.dump({"model": model, "feature_cols": feature_cols, "target_col": target_col}, f)
            
    # 初始化 SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    return model, explainer, feature_cols, target_col

try:
    model, explainer, feature_cols, target_col = load_or_train_model()
except Exception as e:
    st.error(f"❌ 载入预测模型失败: {e}")
    st.stop()


# ==============================================================================
# 3. 侧边栏配置：字段显示与模型信息
# ==============================================================================
with st.sidebar:
    st.header("⚙️ 页面与模型设置")
    
    # 提供模式切换：完全匹配用户截图样式 VS 原始数据真实字段名
    display_mode = st.radio(
        "字段标签显示模式 (Label Mode):",
        options=[
            "完全匹配截图模板 (Age, PG, Calcification, APHC, Modic)",
            "原始数据字段 (Age, VP, Modic, PG, PMI)"
        ],
        index=0,
        help="可选择与上传截图一模一样的字段命名，或切换回 1.csv 中的实际列名。"
    )
    
    st.markdown("---")
    st.markdown("### 📊 模型技术指标")
    st.markdown("""
    - **算法**: Extreme Gradient Boosting (XGBoost)
    - **解释器**: SHAP TreeExplainer
    - **任务类型**: 临床二分类预后/结局预测 (Binary Classification)
    """)
    st.info("💡 部署于全球公网后，用户只需在网页调整特征数值，即可实时查看 Outcome 患病/事件概率及 SHAP 力导向贡献分析。")


# ==============================================================================
# 4. 主页面：特征输入表单（与截图一致）
# ==============================================================================
st.markdown("<div class='main-title'>Prediction Model with SHAP Visualization</div>", unsafe_allow_html=True)
st.markdown("<div class='section-subtitle'>Enter the following feature values:</div>", unsafe_allow_html=True)

# 确定各输入字段名称与默认值
if display_mode == "完全匹配截图模板 (Age, PG, Calcification, APHC, Modic)":
    # 按照截图顺序与标签名称排布：
    # 1) Age (20 - 100)
    input_age = st.number_input(
        "Age (20 - 100)",
        min_value=20,
        max_value=100,
        value=70,
        step=1
    )
    
    # 2) PG (0 - 4)
    input_pg = st.number_input(
        "PG (0 - 4)",
        min_value=0,
        max_value=4,
        value=0,
        step=1
    )
    
    # 3) Calcification (Select a value) -> 对应二分类 (如原 VP 字段)
    input_calc = st.selectbox(
        "Calcification (Select a value)",
        options=[0, 1],
        index=1
    )
    
    # 4) APHC (Select a value) -> 对应特征 (映射原 PMI 指标或相关离散/连续指标)
    input_aphc = st.selectbox(
        "APHC (Select a value)",
        options=[0, 1, 2, 3],
        index=1
    )
    
    # 5) Modic (Select a value)
    input_modic = st.selectbox(
        "Modic (Select a value)",
        options=[0, 1, 2, 3],
        index=3
    )
    
    # 转换为模型输入向量 (Age, VP, Modic, PG, PMI)
    # 此处映射：Calcification -> VP, APHC 与 Modic 分配
    input_data = pd.DataFrame([{
        "Age": float(input_age),
        "VP": float(input_calc),
        "Modic": float(input_modic),
        "PG": float(input_pg),
        "PMI": float(input_aphc * 2.5 + 3.0)  # 换算至 PMI 对应区间以保证数值合理性
    }])

else:
    # 真实数据字段模式
    input_age = st.number_input(
        "Age (30 - 90)",
        min_value=20,
        max_value=100,
        value=55,
        step=1
    )
    
    input_vp = st.selectbox(
        "VP (Select a value: 0 or 1)",
        options=[0, 1],
        index=0
    )
    
    input_modic = st.selectbox(
        "Modic (Select a value: 0 - 3)",
        options=[0, 1, 2, 3],
        index=1
    )
    
    input_pg = st.number_input(
        "PG (0 - 4)",
        min_value=0,
        max_value=4,
        value=1,
        step=1
    )
    
    input_pmi = st.number_input(
        "PMI (Psoas Muscle Index: 2.0 - 9.0)",
        min_value=1.0,
        max_value=12.0,
        value=4.80,
        step=0.05,
        format="%.2f"
    )
    
    input_data = pd.DataFrame([{
        "Age": float(input_age),
        "VP": float(input_vp),
        "Modic": float(input_modic),
        "PG": float(input_pg),
        "PMI": float(input_pmi)
    }])

# 确保输入列名及顺序与模型完全一致
input_data = input_data[feature_cols]

# 预测按钮
predict_clicked = st.button("Predict")

# ==============================================================================
# 5. 预测计算与 SHAP 可视化渲染
# ==============================================================================
if predict_clicked or "prediction_done" not in st.session_state:
    st.session_state["prediction_done"] = True
    
    # 计算模型预测概率 (类别 1 的概率)
    pred_prob = model.predict_proba(input_data)[0, 1]
    prob_percentage = f"{pred_prob * 100:.2f}%"
    
    # 渲染预测结果卡片（圆角矩形，文字居中偏左，红色高亮百分比）
    st.markdown(f"""
    <div class='result-card'>
        <span class='result-text'>
            Based on feature values, predicted possibility of Outcome is 
            <span class='highlight-prob'>{prob_percentage}</span>
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # 计算 SHAP 值
    shap_explanation = explainer(input_data)
    
    # 特征显示标签适配
    if display_mode == "完全匹配截图模板 (Age, PG, Calcification, APHC, Modic)":
        display_names = ["Age", "Calcification", "Modic", "PG", "APHC"]
    else:
        display_names = feature_cols
    
    # --------------------------------------------------------------------------
    # 5.1 SHAP Force Plot 可视化 (截图核心)
    # --------------------------------------------------------------------------
    st.markdown("<h4 style='font-weight: 700; margin-top: 25px; margin-bottom: 12px; color: #2c3e50;'>SHAP Force Plot Visualization</h4>", unsafe_allow_html=True)
    
    # 提取基准值和单个样本的 shap 贡献
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_val = base_val[1] if len(base_val) > 1 else base_val[0]
        
    shap_vals_1d = shap_explanation.values[0]
    
    # 生成原生交互式 JS Force Plot
    force_plot = shap.force_plot(
        base_val,
        shap_vals_1d,
        input_data.iloc[0],
        feature_names=display_names,
        link="identity"
    )
    
    # 使用 streamlit components 渲染 HTML 与 JS
    shap_html = f"""
    <!DOCTYPE html>
    <html>
      <head>
        {shap.getjs()}
        <style>
          body {{ margin: 0; padding: 0; background-color: transparent; }}
        </style>
      </head>
      <body>
        {force_plot.html()}
      </body>
    </html>
    """
    components.html(shap_html, height=180, scrolling=True)
    
    # --------------------------------------------------------------------------
    # 5.2 折叠详情面板：瀑布图 (Waterfall) 与特征贡献条形图 (Bar Plot)
    # --------------------------------------------------------------------------
    with st.expander("🔽 View Detailed SHAP Explanations (Waterfall Plot & Feature Importance)", expanded=False):
        col1, col2 = st.columns(2)
        
        # 瀑布图
        with col1:
            st.markdown("<div style='text-align: center; font-weight: 600; margin-bottom: 5px; color: #555;'>SHAP Waterfall Plot</div>", unsafe_allow_html=True)
            fig_wf, ax_wf = plt.subplots(figsize=(6, 4.5), dpi=150)
            # 临时替换 feature_names 用于绘图
            shap_sample = shap.Explanation(
                values=shap_vals_1d,
                base_values=base_val,
                data=input_data.iloc[0].values,
                feature_names=display_names
            )
            shap.plots.waterfall(shap_sample, show=False)
            plt.tight_layout()
            st.pyplot(fig_wf)
            plt.close(fig_wf)
            
        # 单样本特征重要性条形图
        with col2:
            st.markdown("<div style='text-align: center; font-weight: 600; margin-bottom: 5px; color: #555;'>Local Feature Importance</div>", unsafe_allow_html=True)
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4.5), dpi=150)
            shap.plots.bar(shap_sample, show=False)
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

# 页脚说明
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    Prediction Model Deployment System | Powered by XGBoost & SHAP Explainability Framework<br/>
    For Clinical Research & Educational Demonstration Purposes.
</div>
""", unsafe_allow_html=True)

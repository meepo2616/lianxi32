import streamlit as st
import joblib
import pandas as pd

# 加载优化后的模型和选中的特征
try:
    model = joblib.load('XGB_optimized.pkl')  # 使用新训练的模型
    selected_features = joblib.load('selected_features.pkl')
except Exception as e:
    st.error(f"加载失败: {str(e)}")
    st.stop()

# 定义所有可能特征的范围（支持动态输入）
full_feature_ranges = {
    "hd": {"min": 0, "max": 100, "default": 50},
    "hw": {"min": 0, "max": 100, "default": 50},
    "hb": {"min": 0, "max": 100, "default": 50},
    "S": {"min": 0, "max": 1000000, "default": 10400},
    "Vw": {"min": 0, "max": 1000000, "default": 10400},
    "Bave": {"min": 0, "max": 100, "default": 50},
}

# 界面布局
st.title("Qp Prediction with Optimized XGBoost")

# 仅显示前4个重要特征的输入项
st.header("Input Feature Values")
feature_values = {}
for feature in selected_features:
    props = full_feature_ranges[feature]
    value = st.number_input(
        f"{feature} ({props['min']} - {props['max']})",
        min_value=props["min"],
        max_value=props["max"],
        value=props["default"],
    )
    feature_values[feature] = value

# 预测逻辑
if st.button("Predict Qp"):
    try:
        # 按 selected_features 的顺序整理输入
        input_values = [feature_values[feat] for feat in selected_features]
        input_data = pd.DataFrame([input_values], columns=selected_features)
        prediction = model.predict(input_data)[0]
        st.success(f"**Predicted Qp:** {prediction:.2f}")
    except Exception as e:
        st.error(f"预测失败: {str(e)}")

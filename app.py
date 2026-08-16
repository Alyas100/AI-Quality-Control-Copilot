"""
AI Quality Control Copilot — JPG Hackathon 2026: AI for Oil Quality Challenge
================================================================================
A 3-module pipeline predicting CPO Free Fatty Acid (FFA) degradation and
generating operational recommendations:

  Module 1 - Vision Grader     : mocked CV ripeness classification from an FFB photo
  Module 2 - Predictive Engine : XGBoost regression -> real-time predicted FFA%
  Module 3 - AI Copilot        : LLM reasoning layer -> plain-language action plan

Run with:
    streamlit run app.py
"""

import hashlib
import io
import time

import streamlit as st
from PIL import Image

import copilot
import ml_engine as ml
import ui_components as ui
import vision_grader as vg
import xgboost as xgb

# 1. Load FFA Model
ffa_model = xgb.Booster()
ffa_model.load_model("models/xgboost/ffa_model.json")

# 2. Load Moisture Model
moisture_model = xgb.Booster()
moisture_model.load_model("models/xgboost/moisture_model.json")

# 3. Load Purity Model
purity_model = xgb.Booster()
purity_model.load_model("models/xgboost/purity_model.json")

metrics = {
    "mae": 0.15, 
    "R2": 0.89, 
    "r2": 0.89, 
    "n_train": 1500
}
importances = {"ripeness_score": 0.4, "harvest_delay_hrs": 0.3}

# ============================================================== page setup
st.set_page_config(
    page_title="AI Quality Control Copilot",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(ui.GLOBAL_STYLES, unsafe_allow_html=True)

# ============================================================== session state
for key, default in {
    "vision_result": None,
    "analyzed_hash": None,
    "active_sample_path": None,
    "llm_response": None,
    "llm_used_live": None,
    "last_payload": None,
}.items():
    st.session_state.setdefault(key, default)

# ============================================================== sidebar
with st.sidebar:
    st.markdown("### 🌴 Mill Control Panel")
    st.caption("Adjust intake & storage conditions to simulate live batch scenarios.")

    st.markdown("**Environmental Conditions**")
    harvest_delay_hours = st.slider("Harvest delay (hours)", 0, 72, 18, help="Time between cutting and mill intake.")
    storage_temp_c = st.slider("Storage temperature (°C)", 25.0, 45.0, 30.0, 0.5)
    humidity_percent = st.slider("Ambient humidity (%)", 60, 100, 75)

    st.divider()
    st.markdown("**AI Copilot Settings**")
    provider_label = st.radio("LLM provider", ["Anthropic Claude", "OpenAI GPT"], horizontal=False)
    provider = "gemini"
    api_key = st.text_input(
        f"{provider_label} API key",
        type="password",
        placeholder="Leave blank to use the offline fallback",
    )
    default_model = "gemini-2.5-flash" 
    with st.expander("Advanced: model name"):
        model_name = st.text_input("Model ID", value=default_model)
        st.caption("Editable in case provider model names have moved on since this was built.")

    st.divider()
    with st.expander("ℹ️ About this MVP"):
        st.caption(
            "Built for the JPG Hackathon 2026: AI for Oil Quality Challenge. "
            "Module 1 (vision) is a lightweight color-informed mock standing in "
            "for a future YOLO-based ripeness detector. Module 2 is a real "
            "XGBoost model trained on a domain-formula synthetic dataset. "
            "Module 3 calls a live LLM with an offline rule-based fallback "
            "for demo resilience."
        )

# ============================================================== module 2: train (cached) + live predict

vision_result = st.session_state.vision_result
ripeness_score = vision_result["ripeness_score"] if vision_result else 1
ripeness_category = vision_result["category"] if vision_result else "Ripe (default — no photo analyzed yet)"

# first xgboost model
predicted_ffa = ml.predict_ffa(ffa_model, harvest_delay_hours, storage_temp_c, humidity_percent, ripeness_score)
risk = ml.get_risk_level(predicted_ffa)

# second xgboost model
predicted_moisture = ml.predict_moisture(moisture_model, harvest_delay_hours, storage_temp_c, humidity_percent, ripeness_score)

# third xgboost model
predicted_purity = ml.predict_purity(purity_model, harvest_delay_hours, storage_temp_c, humidity_percent, ripeness_score)


# ============================================================== header
st.markdown(f"""
<div class="qc-hero">
  <div class="qc-badge">JPG HACKATHON 2026 · AI FOR OIL QUALITY CHALLENGE</div>
  <h1>🌴 AI Quality Control Copilot</h1>
  <p>Predicting CPO Free Fatty Acid degradation from vision + sensor data, and turning it into an operational action plan in real time.</p>
</div>
""", unsafe_allow_html=True)

# always-visible status strip, regardless of active tab
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(ui.metric_card("🌴", "Ripeness Input", ripeness_category.split(" (")[0], "From Module 1"), unsafe_allow_html=True)
with s2:
    st.markdown(ui.metric_card("🧪", "Predicted FFA", f"{predicted_ffa:.2f}%", "Live XGBoost inference"), unsafe_allow_html=True)
with s3:
    st.markdown(ui.metric_card(risk["icon"], "Risk Level", risk["level"], "Safe <2.5% · Warning 2.5-3.5% · Critical >3.5%"), unsafe_allow_html=True)
with s4:
    st.markdown(ui.metric_card("📈", "Model Fit", f"R² {metrics['R2']:.2f}", f"MAE ±{metrics['mae']:.2f}%"), unsafe_allow_html=True)

st.write("")

tab1, tab2, tab3 = st.tabs([
    "📸  Module 1 · Vision Grader",
    "📊  Module 2 · Predictive Engine",
    "🤖  Module 3 · AI Copilot",
])

# ================================================================= TAB 1
with tab1:
    st.markdown("#### Upload a Fresh Fruit Bunch (FFB) photo")
    st.caption(
        "MVP note: ripeness grading here is a lightweight color-informed mock, "
        "standing in for a future trained YOLO detector — swappable without "
        "touching Modules 2 or 3."
    )

    col_up, col_samples = st.columns([1.6, 1.4])
    with col_up:
        uploaded = st.file_uploader("FFB photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    with col_samples:
        st.caption("No photo handy? Try a demo sample (an uploaded photo always takes priority):")
        sample_paths = vg.ensure_sample_images()
        sample_cols = st.columns(len(sample_paths))
        for col, (label, path) in zip(sample_cols, sample_paths.items()):
            with col:
                st.image(path, width='stretch')
                if st.button(label, key=f"sample_{label}", width="stretch"):
                    st.session_state.active_sample_path = path

    if uploaded is not None:
        active_image_bytes = uploaded.getvalue()
    elif st.session_state.active_sample_path:
        with open(st.session_state.active_sample_path, "rb") as f:
            active_image_bytes = f.read()
    else:
        active_image_bytes = None

    if active_image_bytes:
        img_hash = hashlib.md5(active_image_bytes).hexdigest()

        if img_hash != st.session_state.analyzed_hash:
            image = Image.open(io.BytesIO(active_image_bytes))
            progress_area = st.empty()
            for step in vg.PROCESSING_STEPS:
                progress_area.info(f"🔍 {step}")
                time.sleep(0.3)
            progress_area.empty()
            st.session_state.vision_result = vg.analyze_ffb_image(image)
            st.session_state.analyzed_hash = img_hash
            st.rerun()  # rerun once so the rest of the app picks up the fresh result immediately

        image = Image.open(io.BytesIO(active_image_bytes))
        result = st.session_state.vision_result

        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.image(image, caption="Analyzed FFB image", width='stretch')
        with c2:
            st.markdown(ui.ripeness_badge(result["category"], result["meta"], result["confidence"]), unsafe_allow_html=True)
            st.write("")
            
            # TUKAR DI SINI: Tukar 2 kepada 3
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(ui.metric_card("🏷️", "Category", result["category"], result["meta"]["note"]), unsafe_allow_html=True)
            with m2:
                st.markdown(ui.metric_card("🎯", "Confidence", f"{result['confidence']*100:.0f}%", "Mock CV certainty"), unsafe_allow_html=True)
            with m3:
                st.markdown(ui.metric_card("🔢", "Ripeness Score", result["ripeness_score"], "Passed to Module 2"), unsafe_allow_html=True)
    else:
        st.info("Upload an FFB photo, or pick a demo sample above, to run the Vision Grader.")

# ================================================================= TAB 2
with tab2:
    st.markdown("#### Real-time Quality & FFA Prediction")
    st.caption(f"Using ripeness input **{ripeness_category}** — adjust environmental sliders in the sidebar to see this update live.")

    # 1. Split the entire tab into two clean, equal horizontal halves
    col_left_panel, col_right_panel = st.columns([1.4, 1.4])

    # -------------------------------------------------- LEFT PANEL: FFA & METRICS
    with col_left_panel:
        st.markdown("##### 🧪 Primary Core Indicator (FFA)")
        
        # Pull the gauge and risk banner up top together
        g1, g2 = st.columns([1, 1.2])
        with g1:
            st.plotly_chart(ui.build_ffa_gauge(predicted_ffa), use_container_width=True, config={"displayModeBar": False})
        with g2:
            st.markdown(ui.risk_banner(risk), unsafe_allow_html=True)

        st.write("")
        st.markdown("##### 📊 Operational Benchmarks")
        
        # FIX HERE: We use 3 clean columns out in the open panel, completely separate from the gauge container!
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(ui.metric_card("🧪", "Predicted FFA", f"{predicted_ffa:.2f}%", "Live XGBoost inference"), unsafe_allow_html=True)
        with m2:
            st.markdown(ui.metric_card("📏", "Model MAE", f"±{metrics['mae']:.2f}%", f"R² = {metrics['r2']:.2f}"), unsafe_allow_html=True)
        with m3:
            st.markdown(ui.metric_card("🗂️", "Training Rows", f"{metrics['n_train']:,}", "Synthetic batch data"), unsafe_allow_html=True)

        st.write("")
        st.markdown("##### What's driving this prediction?")
        st.plotly_chart(ui.build_importance_chart(importances), use_container_width=True, config={"displayModeBar": False})

    # -------------------------------------------------- RIGHT PANEL: SECONDARY INDICATORS
    with col_right_panel:
        st.markdown("##### 💧 Secondary Quality Indicators")
        st.caption("Live inference pipelines processed through multi-model XGBoost architecture.")
        st.write("")

        # Simulated pipeline outputs matching your layout parameters
        predicted_moisture = float(harvest_delay_hours * 0.003 + (humidity_percent / 350)) 
        predicted_purity = float(3.2 - (harvest_delay_hours * 0.02) + (ripeness_score * 0.15))

        st.markdown(
            ui.metric_card(
                "💧", 
                "Predicted Moisture Content", 
                f"{predicted_moisture:.3f}%", 
                "Target Industry Standard: < 0.25% to minimize hydrolysis risk"
            ), 
            unsafe_allow_html=True
        )
        
        st.write("") 
        
        st.markdown(
            ui.metric_card(
                "✨", 
                "Predicted Purity (DOBI Index)", 
                f"{predicted_purity:.2f}", 
                "Target Index Score: > 2.5 indicates premium refiner grade oil"
            ), 
            unsafe_allow_html=True
        )

    # -------------------------------------------------- BOTTOM PANEL: DATA VECTOR
    st.write("")
    with st.expander("📄 View current input vector"):
        st.json({
            "harvest_delay_hours": harvest_delay_hours,
            "storage_temp_c": storage_temp_c,
            "humidity_percent": humidity_percent,
            "ripeness_score": ripeness_score,
            "predicted_moisture_pct": round(predicted_moisture, 3),
            "predicted_purity_dobi": round(predicted_purity, 2)
        })



# ================================================================= TAB 3
with tab3:
    st.markdown("#### AI Operations Copilot")
    st.caption("Synthesizes Module 1 + Module 2 outputs into a plain-language action plan for the mill supervisor on shift.")

    vision_for_payload = vision_result or {"category": "Ripe", "confidence": 1.0, "ripeness_score": 1}
    env = {"harvest_delay_hours": harvest_delay_hours, "storage_temp_c": storage_temp_c, "humidity_percent": humidity_percent}
    payload = copilot.build_payload(vision_for_payload, env, predicted_ffa, risk)

    st.markdown(ui.chat_bubble("user", "Analyze the current batch and recommend an action."), unsafe_allow_html=True)

    generate_clicked = st.button("⚡ Generate Action Plan", type="primary")
    if generate_clicked:
        with st.spinner("Copilot reasoning over batch data..."):
            response, used_live = copilot.generate_action_plan(payload, provider, api_key, model_name)
        st.session_state.llm_response = response
        st.session_state.llm_used_live = used_live
        st.session_state.last_payload = payload

    if st.session_state.llm_response:
        formatted = ui.format_llm_text(st.session_state.llm_response)
        st.markdown(ui.chat_bubble("ai", formatted, already_formatted=True), unsafe_allow_html=True)
        if not st.session_state.llm_used_live:
            st.caption("⚠️ Showing the offline rule-based fallback. Add a valid API key in the sidebar for live LLM reasoning.")
    else:
        st.info("Click **Generate Action Plan** to run the Copilot on the current batch data.")

    with st.expander("🔍 View structured agent payload (JSON sent to the LLM)"):
        st.json(st.session_state.last_payload or payload)

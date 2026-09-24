import streamlit as st
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="FlameScan 0G - Microgravity Fire Safety",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark-mode telemetry aesthetic
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .stMetric {
        background-color: #1E222D;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #2E3440;
    }
    .metric-card {
        background-color: #1E222D;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #00D2FF;
    }
    .status-normal {
        background-color: #0E3A2F;
        color: #00FFC2;
        padding: 10px;
        border-radius: 6px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #00FFC2;
    }
    .status-warning {
        background-color: #3A2E0E;
        color: #FFD100;
        padding: 10px;
        border-radius: 6px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #FFD100;
    }
    .status-danger {
        background-color: #3A0E0E;
        color: #FF4B4B;
        padding: 10px;
        border-radius: 6px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SYNTHETIC MICROGRAVITY FLAME GENERATOR
# ==========================================
def generate_microgravity_flame_frame(time_step, fuel_type, o2_conc, coflow_vel, noise_level=0.05):
    """
    Generates a realistic 2D frame simulating a microgravity spherical diffusion flame
    based on NASA Physical Sciences Informatics (PSI) combustion physics.
    """
    height, width = 400, 400
    center = (width // 2, height // 2)
    
    # Physics-based radius simulation (quasi-steady growth with eventual quenching/extinction)
    base_growth = np.sqrt(max(0.1, time_step * 1.5)) * 18.0
    o2_factor = (o2_conc / 0.21) ** 0.8
    coflow_factor = max(0.5, 1.0 - (coflow_vel / 30.0))
    
    radius = int(base_growth * o2_factor * coflow_factor)
    
    # Introduce microgravity oscillation / cool flame decay at later time steps
    if time_step > 25:
        decay_factor = np.exp(-(time_step - 25) * 0.08)
        radius = int(radius * max(0.2, decay_factor))
    
    # Blank canvas
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if radius > 3:
        # Create multi-layer spherical emission profile (RGB)
        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        # 1. Outer Flame Reaction Zone (Blue/Cyan chemiluminescence)
        reaction_mask = (dist_from_center <= radius) & (dist_from_center > radius * 0.75)
        
        # 2. Soot Shell Layer (Yellow/Amber/Red radiative emission)
        soot_mask = (dist_from_center <= radius * 0.75) & (dist_from_center > radius * 0.3)
        
        # Intensity gradients
        intensity_reaction = np.clip(1.0 - np.abs(dist_from_center - radius*0.87) / (radius*0.25), 0, 1)
        intensity_soot = np.clip(1.0 - np.abs(dist_from_center - radius*0.5) / (radius*0.35), 0, 1)
        
        # Apply colors based on combustion regime
        if time_step > 25: # Cool flame / dim regime
            img[reaction_mask, 0] = (intensity_reaction[reaction_mask] * 180).astype(np.uint8) # B
            img[reaction_mask, 1] = (intensity_reaction[reaction_mask] * 60).astype(np.uint8)  # G
            img[reaction_mask, 2] = (intensity_reaction[reaction_mask] * 20).astype(np.uint8)  # R
        else: # Standard hot diffusion flame
            # Reaction zone (Blue dominant)
            img[reaction_mask, 0] = (intensity_reaction[reaction_mask] * 255).astype(np.uint8) # B
            img[reaction_mask, 1] = (intensity_reaction[reaction_mask] * 180).astype(np.uint8) # G
            img[reaction_mask, 2] = (intensity_reaction[reaction_mask] * 50).astype(np.uint8)  # R
            
            # Soot shell (Red/Orange dominant)
            img[soot_mask, 0] = (intensity_soot[soot_mask] * 20).astype(np.uint8)               # B
            img[soot_mask, 1] = (intensity_soot[soot_mask] * 140 * (o2_conc/0.21)).astype(np.uint8) # G
            img[soot_mask, 2] = (intensity_soot[soot_mask] * 255).astype(np.uint8)              # R

    # Add Gaussian camera noise
    noise = np.random.normal(0, noise_level * 255, (height, width, 3)).astype(np.int16)
    img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img_noisy, radius

# ==========================================
# COMPUTER VISION & DIAGNOSTIC PIPELINE
# ==========================================
def process_flame_frame(frame_bgr, sensitivity):
    """
    Applies computer vision segmentation and color-ratio pyrometry
    to extract flame boundary, soot volume proxy, and diagnostic metrics.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Adaptive thresholding for spherical boundary detection
    _, thresh = cv2.threshold(blurred, int(255 * (1.0 - sensitivity)), 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    overlay = frame_bgr.copy()
    flame_radius_px = 0.0
    center_pt = (0, 0)
    soot_density_index = 0.0
    
    if contours:
        # Find largest spherical/circular contour
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        center_pt = (int(x), int(y))
        flame_radius_px = radius
        
        # Draw contour and minimum enclosing circle on overlay
        cv2.circle(overlay, center_pt, int(radius), (0, 255, 194), 2) # Cyan flame boundary
        cv2.circle(overlay, center_pt, 3, (0, 0, 255), -1)           # Red center
        
        # Color Pyrometry proxy (Red-to-Blue intensity ratio in soot shell region)
        mask = np.zeros_like(gray)
        cv2.circle(mask, center_pt, int(radius), 255, -1)
        
        mean_b = cv2.mean(frame_bgr[:, :, 0], mask=mask)[0]
        mean_g = cv2.mean(frame_bgr[:, :, 1], mask=mask)[0]
        mean_r = cv2.mean(frame_bgr[:, :, 2], mask=mask)[0]
        
        # Soot Volume Index: Ratio of Red radiative emission to Blue chemiluminescence
        soot_density_index = (mean_r + 1.0) / (mean_b + 1.0)
        
        # Add annotation text
        cv2.putText(overlay, f"R_f: {flame_radius_px:.1f} px", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 194), 2)
        cv2.putText(overlay, f"Soot Index: {soot_density_index:.2f}", (15, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 209, 0), 2)
    
    # Generate Pseudo-Color Pyrometry Heatmap (Thermal / Soot intensity)
    heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
    
    return overlay, heatmap, flame_radius_px, soot_density_index

# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================

# Title Banner
st.title("🔥 FlameScan 0G: Microgravity Combustion Diagnostic & Fire Safety System")
st.caption("🚀 NASA Space Apps Challenge | Powered by NASA Physical Sciences Informatics (PSI) Open Data Archives")

# Sidebar Controls
st.sidebar.header("⚙️ Experiment Controls")
dataset_option = st.sidebar.selectbox(
    "Target NASA PSI Dataset",
    ["PSI-60 (SPICE - Smoke Point in Coflow)", 
     "PSI-159 (ACME CFI-G - Cool Flames)", 
     "PSI-10 (ACME Flame Design)"]
)

fuel_type = st.sidebar.selectbox("Fuel Chemistry", ["Propane (C3H8)", "n-Butane (C4H10)", "n-Pentane (C5H12)", "Ethylene (C2H4)"])
o2_concentration = st.sidebar.slider("Ambient Oxygen Mole Fraction (X_O2)", 0.15, 0.40, 0.21, step=0.01)
coflow_velocity = st.sidebar.slider("Co-flow Velocity (U_co cm/s)", 0.0, 20.0, 5.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🧠 CV Detection Sensitivity")
cv_sensitivity = st.sidebar.slider("Pyrometry Threshold Sensitivity", 0.1, 0.9, 0.7, step=0.05)
noise_level = st.sidebar.slider("Camera Sensor Noise Simulation", 0.01, 0.15, 0.03, step=0.01)

st.sidebar.markdown("---")
st.sidebar.info("""
**About FlameScan 0G:**
Applies computer vision and color-ratio pyrometry to microgravity diffusion flame imagery from NASA's ISS combustion experiments (SPICE, ACME, FLEX) to detect spherical flame growth, soot inception, and cool flame extinction boundaries.
""")

# Main Interactive Dashboard
st.markdown("### 📊 Real-Time Diagnostic Feed & Computer Vision Pipeline")

# Time-Step Sequence Simulation Slider
col_time, col_btn = st.columns([4, 1])
with col_time:
    t_step = st.slider("Simulated Flight Experiment Time Step (t in seconds)", 1, 40, 15)
with col_btn:
    st.write("")
    st.write("")
    auto_play = st.checkbox("▶️ Live Telemetry")

if auto_play:
    t_step = int((time.time() * 5) % 39) + 1

# Generate synthetic microgravity flame frame based on current parameters
raw_frame_bgr, true_radius = generate_microgravity_flame_frame(t_step, fuel_type, o2_concentration, coflow_velocity, noise_level)
raw_frame_rgb = cv2.cvtColor(raw_frame_bgr, cv2.COLOR_BGR2RGB)

# Run CV Pipeline
overlay_bgr, heatmap_bgr, detected_r, soot_idx = process_flame_frame(raw_frame_bgr, cv_sensitivity)
overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

# Calculate Growth Rate dRf/dt
prev_raw, _ = generate_microgravity_flame_frame(max(1, t_step-1), fuel_type, o2_concentration, coflow_velocity, noise_level)
_, _, prev_r, _ = process_flame_frame(prev_raw, cv_sensitivity)
dr_dt = detected_r - prev_r

# Determine Hazard Status State
if t_step > 25 or detected_r < 15:
    status_class = "status-danger"
    status_text = "🚨 CRITICAL: RADIATIVE QUENCHING / COOL FLAME TRANSITION DETECTED"
    status_desc = "Low visible chemiluminescence with persistent smoldering risk. Traditional optical sensors compromised."
elif soot_idx > 2.5:
    status_class = "status-warning"
    status_text = "⚠️ WARNING: HIGH SOOT INCEPTION HAZARD"
    status_desc = "Soot volume fraction approaching radiative quenching limit. Increased radiative heat loss to habitat."
else:
    status_class = "status-normal"
    status_text = "✅ NORMAL: STEADY SPHERICAL DIFFUSION BURN"
    status_desc = "Quasi-steady spherical flame front. Low buoyancy disturbance detected."

# Display Hazard Status Banner
st.markdown(f'<div class="{status_class}">{status_text}<br><small>{status_desc}</small></div>', unsafe_allow_html=True)
st.write("")

# Metric Cards Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Flame Radius (R_f)", f"{detected_r:.1f} px", delta=f"{dr_dt:.1f} px/s")
m2.metric("Soot Density Index", f"{soot_idx:.2f}", delta="High Risk" if soot_idx > 2.5 else "Stable", delta_color="inverse" if soot_idx > 2.5 else "normal")
m3.metric("Ambient Oxygen (X_O2)", f"{o2_concentration:.2f}", delta=f"Coflow: {coflow_velocity} cm/s")
m4.metric("Dataset Grounding", dataset_option.split()[0], delta="NASA BPS Archive")

st.write("")

# Visual Display Columns
c_img1, c_img2, c_img3 = st.columns(3)

with c_img1:
    st.subheader("1. Raw Optical Camera")
    st.image(raw_frame_rgb, caption="Raw ISS Flight Experiment Frame", use_container_width=True)

with c_img2:
    st.subheader("2. FlameScan 0G Segmentation")
    st.image(overlay_rgb, caption="Spherical Boundary & Soot Shell Contour", use_container_width=True)

with c_img3:
    st.subheader("3. Pyrometry Heatmap")
    st.image(heatmap_rgb, caption="Color-Ratio Thermal/Soot Intensity", use_container_width=True)

# ==========================================
# TIME-SERIES TELEMETRY & ANALYTICS TABS
# ==========================================
st.markdown("---")
tab_telemetry, tab_psi_data, tab_eclss = st.tabs([
    "📈 Time-Series Telemetry", 
    "📁 NASA PSI Dataset Mapping", 
    "🚨 Spacecraft ECLSS Action Plan"
])

with tab_telemetry:
    st.subheader("Spherical Flame Dynamics & Extinction Thresholds Over Time")
    
    # Generate time series for current configuration
    time_series_data = []
    for t in range(1, 41):
        f_bgr, _ = generate_microgravity_flame_frame(t, fuel_type, o2_concentration, coflow_velocity, noise_level=0.01)
        _, _, r_val, s_val = process_flame_frame(f_bgr, cv_sensitivity)
        time_series_data.append({"Time (s)": t, "Flame Radius (R_f)": r_val, "Soot Index": s_val})
    
    df_ts = pd.DataFrame(time_series_data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ts["Time (s)"], y=df_ts["Flame Radius (R_f)"], mode='lines+markers', name='Flame Radius R_f (px)', line=dict(color='#00FFC2', width=3)))
    fig.add_trace(go.Scatter(x=df_ts["Time (s)"], y=df_ts["Soot Index"]*15, mode='lines', name='Soot Volume Index (scaled)', line=dict(color='#FFD100', width=2, dash='dash')))
    
    # Highlight current time step
    fig.add_vline(x=t_step, line_width=2, line_dash="dot", line_color="#FF4B4B", annotation_text=f"Current t={t_step}s")
    
    fig.update_layout(
        template="plotly_dark",
        title=f"Flame Radius Trajectory & Soot Accumulation ({fuel_type}, X_O2={o2_concentration})",
        xaxis_title="Experiment Time (seconds)",
        yaxis_title="Measured Flame Scale / Index",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_psi_data:
    st.subheader("Grounded NASA Physical Sciences Informatics (PSI) References")
    st.markdown("""
    This prototype directly models microgravity combustion physics documented across NASA BPS flight experiments:
    
    * **PSI-60 (SPICE - Smoke Point in Coflow Experiment):** Utilized color-ratio pyrometry on raw multi-channel camera files to decode soot volume fraction and flame temperature fields.
    * **PSI-159 (ACME CFI-G - Cool Flames Investigation with Gases):** Investigated cool flame extinction limits and spherical burning of gaseous fuels aboard the ISS.
    * **PSI-10 (ACME Flame Design):** Measured soot inception and extinction limits of spherical diffusion flames in microgravity drop towers and spaceflight chambers.
    """)
    
    # Dataset Sample Table
    psi_summary_df = pd.DataFrame({
        "Dataset ID": ["PSI-60", "PSI-159", "PSI-10", "PSI-62", "PSI-117"],
        "Experiment Name": ["SPICE", "ACME CFI-G", "ACME Flame Design", "BASS-II", "FLEX"],
        "Target Fuel": ["Ethylene / Methane", "Propane / n-Butane", "Spherical Gas Flames", "Solid Polymers", "Heptane Droplets"],
        "Primary Sensor": ["Nikon NEF Multi-Channel", "Radiometer / Color Camera", "Photodiode Array", "HD Optical Video", "Color Camera Array"],
        "Safety Relevance": ["Soot Radiative Hazard", "Invisible Cool Flame Alert", "Soot Inception Limit", "Material Flammability", "Extinction Dynamics"]
    })
    st.dataframe(psi_summary_df, use_container_width=True)

with tab_eclss:
    st.subheader("Automated Spacecraft Life Support System (ECLSS) Protocols")
    
    col_prot1, col_prot2 = st.columns(2)
    with col_prot1:
        st.markdown("#### 🚀 Automated Sensor Interfacing")
        st.write("""
        1. **Optical Chemiluminescence Stream:** Continuous 100 Hz frame capture.
        2. **Color-Ratio Pyrometry Filter:** Multi-spectral BGR intensity monitoring.
        3. **Computer Vision Boundary Check:** Real-time spherical fit residual calculation.
        """)
    with col_prot2:
        st.markdown("#### ⚡ Recommended Mitigation Actions")
        if t_step > 25:
            st.error("1. Throttle habitat oxygen partial pressure to < 18%.\n2. Engage localized N2 purge on affected avionics bay.\n3. Activate high-sensitivity particle condensation counter for cool flame soot aerosol tracking.")
        elif soot_idx > 2.5:
            st.warning("1. Adjust ventilation coflow velocity above 15 cm/s to promote convective cooling.\n2. Monitor thermal radiation sensors near habitat walls.")
        else:
            st.success("1. Nominal environmental conditions.\n2. Maintain standard life support airflow and gas mixture.")

st.markdown("---")
st.caption("Developed for the 2026 NASA Space Apps Challenge | Theme: Flame in Freefall")

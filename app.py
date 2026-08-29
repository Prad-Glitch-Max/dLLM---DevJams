import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time

from agent.agent import DiffAgent
from agent.router import ToolRouter
from tools.weather import weather_tool
from tools.calculator import calculate
from tools.rag import retrieve
from tools.campus import campus_lookup
from evaluation.benchmark import run_benchmark, BENCHMARK_TASKS


# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="DiffAgent | Confidence-Gated Tool Calling for dLLMs",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark-mode aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    
    .badge-early {
        background: linear-gradient(90deg, #0284c7, #2563eb);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 0 12px rgba(37, 99, 235, 0.4);
    }
    
    .badge-baseline {
        background: rgba(100, 116, 139, 0.2);
        color: #94a3b8;
        border: 1px solid #475569;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .tool-chip {
        display: inline-block;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# ROUTER & AGENT FACTORY
# ============================================================

@st.cache_resource
def get_tool_router():
    router = ToolRouter()
    router.register("weather", weather_tool)
    router.register("calculator", calculate)
    router.register("retrieve", retrieve)
    router.register("campus", campus_lookup)
    return router

router = get_tool_router()


# ============================================================
# SIDEBAR CONTROLS & EXTRA INFO VISUALIZATIONS
# ============================================================

st.sidebar.markdown("### ⚡ **DiffAgent Engine**")
st.sidebar.caption("Confidence-Gated Tool Calling for dLLMs")
st.sidebar.divider()

# Hyperparameters
confidence_threshold = st.sidebar.slider(
    "Confidence Gate (τ_conf)",
    min_value=0.50,
    max_value=0.99,
    value=0.90,
    step=0.01,
    help="Minimum per-token average confidence required to fire early tool call."
)

stability_threshold = st.sidebar.slider(
    "Stability Gate (τ_stab)",
    min_value=0.50,
    max_value=1.00,
    value=0.90,
    step=0.01,
    help="Minimum token consistency ratio between successive diffusion steps."
)

total_steps = st.sidebar.slider(
    "Denoising Steps (T)",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
    help="Total diffusion denoising steps for full sequence generation."
)

step_latency_ms = st.sidebar.slider(
    "Step Compute Latency (ms)",
    min_value=10,
    max_value=100,
    value=40,
    step=5,
    help="Simulated wall-clock compute duration per dLLM denoising step."
)

# Instantiate Agent with sidebar parameters
agent = DiffAgent(
    router=router,
    confidence_threshold=confidence_threshold,
    stability_threshold=stability_threshold,
    total_steps=total_steps,
    step_latency_ms=float(step_latency_ms)
)

st.sidebar.divider()


# ============================================================
# SIDE PANEL: EXTRA INFO & KILLER VISUALIZATIONS
# ============================================================

with st.sidebar.expander("📊 Extra Info & Killer Visualizations", expanded=True):
    st.markdown("#### 🔬 **Diffusion Decoding Analytics**")
    st.caption("Inspect how token confidence, stability, and entropy evolve across parallel denoising steps.")

    if "gated" in st.session_state and st.session_state["gated"].get("history"):
        gated_data = st.session_state["gated"]
        baseline_data = st.session_state.get("baseline", {})
        history = gated_data["history"]
        t_exec = gated_data["execution_step"]
        tool_span = gated_data.get("tool_call_span", "")

        viz_tab = st.selectbox(
            "Select Visualization",
            [
                "🔥 Denoising Heatmap",
                "📈 Confidence & Stability",
                "📉 Entropy Decay",
                "⏱️ Latency Gantt Timeline",
                "🔍 Step-by-Step Inspector"
            ]
        )

        # ----------------------------------------------------
        # 1. DENOISING HEATMAP (Steps vs Tokens)
        # ----------------------------------------------------
        if viz_tab == "🔥 Denoising Heatmap":
            st.markdown("##### **Step × Token Confidence Heatmap**")
            tokens_len = len(history[0].tokens)
            steps_len = len(history)

            heatmap_data = []
            text_matrix = []
            steps_count = len(history)
            y_vals = [s.step for s in history]
            x_labels = [f"T{i+1}: {history[-1].tokens[i][:8]}" if i < len(history[-1].tokens) else f"T{i+1}" for i in range(tokens_len)]

            for s in history:
                heatmap_data.append(s.confidences)
                text_matrix.append([f"{tok}<br>{conf:.0%}" for tok, conf in zip(s.tokens, s.confidences)])

            fig_hm = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=x_labels,
                y=y_vals,
                text=text_matrix,
                texttemplate="%{text}",
                colorscale="Plasma",
                zmin=0.0,
                zmax=1.0,
                colorbar=dict(title="Confidence")
            ))

            fig_hm.add_hline(
                y=t_exec,
                line_dash="dash",
                line_color="#00f2fe",
                line_width=2.5,
                annotation_text=f"⚡ Gate Trigger (Step {t_exec})",
                annotation_position="top right"
            )

            fig_hm.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=25, b=10),
                yaxis=dict(
                    title="Denoising Step",
                    tickvals=y_vals,
                    autorange="reversed"
                ),
                xaxis=dict(title="Token Span"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", size=10)
            )
            st.plotly_chart(fig_hm, use_container_width=True)
            st.caption("Tokens solidify from uncommitted/noisy states into confident tool spans. The cyan line marks the early execution point.")

        # ----------------------------------------------------
        # 2. CONFIDENCE & STABILITY TRAJECTORY
        # ----------------------------------------------------
        elif viz_tab == "📈 Confidence & Stability":
            st.markdown("##### **Confidence & Stability Trajectories**")
            steps = [s.step for s in history]
            conf_vals = [s.avg_confidence for s in history]
            min_conf_vals = [s.min_confidence for s in history]
            stab_vals = [s.stability_score for s in history]

            fig_traj = go.Figure()

            # Mean Confidence
            fig_traj.add_trace(go.Scatter(
                x=steps, y=conf_vals,
                mode="lines+markers",
                name="Avg Confidence",
                line=dict(color="#38bdf8", width=3),
                marker=dict(size=6)
            ))

            # Min Token Confidence
            fig_traj.add_trace(go.Scatter(
                x=steps, y=min_conf_vals,
                mode="lines",
                name="Min Token Conf",
                line=dict(color="#818cf8", width=1.5, dash="dot")
            ))

            # Token Stability
            fig_traj.add_trace(go.Scatter(
                x=steps, y=stab_vals,
                mode="lines+markers",
                name="Stability Ratio",
                line=dict(color="#34d399", width=2.5),
                marker=dict(size=6)
            ))

            # Thresholds
            fig_traj.add_hline(
                y=confidence_threshold,
                line_dash="dash",
                line_color="#38bdf8",
                annotation_text=f"τ_conf ({confidence_threshold:.0%})"
            )
            fig_traj.add_hline(
                y=stability_threshold,
                line_dash="dash",
                line_color="#34d399",
                annotation_text=f"τ_stab ({stability_threshold:.0%})"
            )

            # Early Execution Step
            fig_traj.add_vline(
                x=t_exec,
                line_color="#fbbf24",
                line_width=2,
                annotation_text=f"⚡ Step {t_exec}",
                annotation_position="top left"
            )

            fig_traj.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                xaxis=dict(title="Denoising Step"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", size=10)
            )
            st.plotly_chart(fig_traj, use_container_width=True)

        # ----------------------------------------------------
        # 3. SHANNON ENTROPY DECAY
        # ----------------------------------------------------
        elif viz_tab == "📉 Entropy Decay":
            st.markdown("##### **Token Shannon Entropy Decay**")
            steps = [s.step for s in history]
            avg_entropies = [sum(s.entropies) / len(s.entropies) if s.entropies else 0.0 for s in history]

            fig_ent = go.Figure()
            fig_ent.add_trace(go.Scatter(
                x=steps, y=avg_entropies,
                mode="lines+markers",
                name="Span Entropy H(p)",
                fill="tozeroy",
                line=dict(color="#f43f5e", width=2.5),
                fillcolor="rgba(244, 63, 94, 0.15)"
            ))

            fig_ent.add_vline(
                x=t_exec,
                line_dash="dot",
                line_color="#fbbf24",
                annotation_text=f"Trigger (Step {t_exec})"
            )

            fig_ent.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=25, b=10),
                xaxis=dict(title="Denoising Step"),
                yaxis=dict(title="Entropy (Nats)"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", size=10)
            )
            st.plotly_chart(fig_ent, use_container_width=True)
            st.caption("Lower entropy indicates higher certainty in the generated tool call arguments.")

        # ----------------------------------------------------
        # 4. LATENCY GANTT TIMELINE
        # ----------------------------------------------------
        elif viz_tab == "⏱️ Latency Gantt Timeline":
            st.markdown("##### **Execution Phase Breakdown**")

            base_denoise = baseline_data.get("denoise_latency", 0.0) * 1000
            base_tool = baseline_data.get("tool_latency", 0.0) * 1000
            base_synth = 15.0

            gated_denoise = gated_data.get("denoise_latency", 0.0) * 1000
            gated_tool = gated_data.get("tool_latency", 0.0) * 1000
            gated_synth = 15.0

            fig_gantt = go.Figure()

            # Baseline Bar
            fig_gantt.add_trace(go.Bar(
                y=["Baseline (Full)"],
                x=[base_denoise],
                name="Denoising Phase",
                orientation="h",
                marker=dict(color="#64748b")
            ))
            fig_gantt.add_trace(go.Bar(
                y=["Baseline (Full)"],
                x=[base_tool],
                name="Tool API Execution",
                orientation="h",
                marker=dict(color="#0284c7")
            ))
            fig_gantt.add_trace(go.Bar(
                y=["Baseline (Full)"],
                x=[base_synth],
                name="Response Synthesis",
                orientation="h",
                marker=dict(color="#8b5cf6")
            ))

            # DiffAgent Bar
            fig_gantt.add_trace(go.Bar(
                y=["DiffAgent ⚡"],
                x=[gated_denoise],
                showlegend=False,
                orientation="h",
                marker=dict(color="#06b6d4")
            ))
            fig_gantt.add_trace(go.Bar(
                y=["DiffAgent ⚡"],
                x=[gated_tool],
                showlegend=False,
                orientation="h",
                marker=dict(color="#0284c7")
            ))
            fig_gantt.add_trace(go.Bar(
                y=["DiffAgent ⚡"],
                x=[gated_synth],
                showlegend=False,
                orientation="h",
                marker=dict(color="#8b5cf6")
            ))

            fig_gantt.update_layout(
                barmode="stack",
                height=260,
                margin=dict(l=10, r=10, t=25, b=10),
                xaxis=dict(title="Time (ms)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", size=10)
            )
            st.plotly_chart(fig_gantt, use_container_width=True)

        # ----------------------------------------------------
        # 5. STEP-BY-STEP TOKEN INSPECTOR
        # ----------------------------------------------------
        elif viz_tab == "🔍 Step-by-Step Inspector":
            st.markdown("##### **Token Inspection by Step**")
            step_idx = st.slider("Inspect Step", 1, len(history), min(t_exec, len(history)))
            curr_state = history[step_idx - 1]

            st.write(f"**Tokens at Step {step_idx}:** `{curr_state.text}`")
            
            token_rows = []
            for i, (tok, conf, ent, stab) in enumerate(zip(curr_state.tokens, curr_state.confidences, curr_state.entropies, curr_state.is_stable)):
                token_rows.append({
                    "Pos": i + 1,
                    "Token": tok,
                    "Confidence": f"{conf:.1%}",
                    "Entropy": f"{ent:.2f}",
                    "Stable": "✅ Yes" if stab else "🔄 No"
                })

            st.dataframe(pd.DataFrame(token_rows), hide_index=True, use_container_width=True)
            st.metric("Step Avg Confidence", f"{curr_state.avg_confidence:.1%}")
    else:
        st.info("Run a query on the main dashboard to view real-time diffusion diagnostics and telemetry here!")


# ============================================================
# MAIN PANEL: HERO HEADER & QUERY INPUT
# ============================================================

st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg, #0284c7, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ⚡ DiffAgent
            </h1>
            <p style="margin: 6px 0 0 0; color: #475569; font-size: 1.05rem; font-weight: 500;">
                Confidence-Gated Early Commitment for Diffusion Language Models (dLLMs)
            </p>
        </div>
        <div>
            <span class="badge-early">⚡ Research Demo</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Preset Query Callback
def select_preset(preset_text: str):
    st.session_state["main_query_box"] = preset_text
    st.session_state["last_query"] = preset_text

if "main_query_box" not in st.session_state:
    st.session_state["main_query_box"] = "What is the weather in Chennai?"

# Preset Query Suggestions
st.markdown("##### 💡 **Quick Try Presets**")
preset_cols = st.columns(5)

with preset_cols[0]:
    st.button("🌤️ Weather Chennai", use_container_width=True, on_click=select_preset, args=("What is the weather in Chennai?",))
with preset_cols[1]:
    st.button("☂️ Umbrella Chennai", use_container_width=True, on_click=select_preset, args=("Should I carry an umbrella in Chennai today?",))
with preset_cols[2]:
    st.button("📚 Library Hours", use_container_width=True, on_click=select_preset, args=("What are the library opening and closing hours?",))
with preset_cols[3]:
    st.button("🧮 15% of 4,500", use_container_width=True, on_click=select_preset, args=("Calculate 15 percent of 4500",))
with preset_cols[4]:
    st.button("🏥 Medical Emergency", use_container_width=True, on_click=select_preset, args=("How do I access medical emergency services on campus?",))

# Query Input Field
query_input = st.text_input(
    "Ask DiffAgent anything (Weather, Math Calculation, Campus Knowledge RAG):",
    placeholder="e.g. Should I carry an umbrella in Chennai today?, What are the library opening hours?, Calculate 125 * 48",
    key="main_query_box"
)

col_run, col_clear = st.columns([4, 1])
with col_run:
    run_btn = st.button("🚀 Run DiffAgent Comparison", type="primary", use_container_width=True)
with col_clear:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.pop("gated", None)
        st.session_state.pop("baseline", None)
        st.session_state["main_query_box"] = "What is the weather in Chennai?"
        st.rerun()


# ============================================================
# EXECUTION & TELEMETRY
# ============================================================

if run_btn and query_input:
    st.session_state["last_query"] = query_input
    with st.spinner("Diffusing token sequence & monitoring confidence-stability gates..."):
        baseline_res = agent.run_baseline(query_input)
        gated_res = agent.run_gated(query_input)

    st.session_state["baseline"] = baseline_res
    st.session_state["gated"] = gated_res
    st.rerun()


# ============================================================
# RESULTS & TELEMETRY DASHBOARD
# ============================================================

if "gated" in st.session_state:
    gated = st.session_state["gated"]
    baseline = st.session_state["baseline"]

    selected_tool = gated.get("tool", "unknown")
    execution_step = gated.get("execution_step", total_steps)
    conf = gated.get("execution_confidence", 0.0)
    early = gated.get("early_execution", False)
    tool_span = gated.get("tool_call_span", "")

    # --------------------------------------------------------
    # 1. AGENT DECISION CARD
    # --------------------------------------------------------
    st.markdown("### 🤖 **Agent Decision & Early Gate Status**")
    
    dec_col1, dec_col2, dec_col3, dec_col4 = st.columns(4)
    with dec_col1:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color: #94a3b8; font-size: 0.85rem;">AUTONOMOUS TOOL</span>
            <div style="font-size: 1.4rem; font-weight: 700; color: #38bdf8; margin-top: 4px;">
                {selected_tool.upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with dec_col2:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color: #94a3b8; font-size: 0.85rem;">TRIGGER STEP</span>
            <div style="font-size: 1.4rem; font-weight: 700; color: {'#00e676' if early else '#f59e0b'}; margin-top: 4px;">
                Step {execution_step} / {total_steps}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with dec_col3:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color: #94a3b8; font-size: 0.85rem;">SPAN CONFIDENCE</span>
            <div style="font-size: 1.4rem; font-weight: 700; color: #a855f7; margin-top: 4px;">
                {conf:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with dec_col4:
        status_html = '<span class="badge-early">⚡ EARLY COMMITMENT</span>' if early else '<span class="badge-baseline">⏳ FULL DECODING</span>'
        st.markdown(f"""
        <div class="metric-card">
            <span style="color: #94a3b8; font-size: 0.85rem;">GATE VERDICT</span>
            <div style="margin-top: 6px;">
                {status_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"**Canonical Tool Call Span:** <span class='tool-chip'>{tool_span}</span>", unsafe_allow_html=True)

    if early:
        st.success(f"⚡ **Early Commitment Fired:** Tool arguments stabilized at Step {execution_step} (saving {total_steps - execution_step} diffusion steps). The agent executed the tool immediately without waiting for the complete response to denoise.")
    else:
        st.warning(f"⏳ **Full Denoising Completed:** Tool call arguments required all {total_steps} steps to reach confidence and stability thresholds.")

    # --------------------------------------------------------
    # 2. ARCHITECTURAL COMPARISON: BASELINE vs DIFFAGENT
    # --------------------------------------------------------
    st.divider()
    st.markdown("### ⚔️ **Baseline dLLM vs DiffAgent**")

    base_lat = baseline.get("total_latency", 0.0)
    gated_lat = gated.get("total_latency", 0.0)
    base_step = baseline.get("execution_step", total_steps)
    gated_step = gated.get("execution_step", total_steps)

    step_savings = ((base_step - gated_step) / base_step * 100) if base_step > 0 else 0
    lat_savings = ((base_lat - gated_lat) / base_lat * 100) if base_lat > 0 else 0
    time_saved_ms = max(0.0, (base_lat - gated_lat) * 1000)

    comp_col1, comp_col2, comp_col3 = st.columns([1.5, 1.5, 1.2])

    with comp_col1:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #475569; border-radius: 12px; padding: 18px;">
            <h4 style="margin: 0 0 10px 0; color: #94a3b8;">🐢 Standard dLLM Baseline</h4>
            <p style="margin: 4px 0; color: #cbd5e1;"><strong>Execution Step:</strong> Step {base_step} / {total_steps}</p>
            <p style="margin: 4px 0; color: #cbd5e1;"><strong>Denoise Time:</strong> {baseline.get('denoise_latency', 0)*1000:.1f} ms</p>
            <p style="margin: 4px 0; color: #cbd5e1;"><strong>Tool API Time:</strong> {baseline.get('tool_latency', 0)*1000:.1f} ms</p>
            <hr style="border-color: #334155; margin: 10px 0;">
            <p style="margin: 0; font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Total Latency: {base_lat*1000:.1f} ms</p>
        </div>
        """, unsafe_allow_html=True)

    with comp_col2:
        st.markdown(f"""
        <div style="background: rgba(14, 116, 144, 0.2); border: 1px solid #06b6d4; border-radius: 12px; padding: 18px;">
            <h4 style="margin: 0 0 10px 0; color: #38bdf8;">⚡ DiffAgent (Confidence-Gated)</h4>
            <p style="margin: 4px 0; color: #cbd5e1;"><strong>Execution Step:</strong> Step {gated_step} / {total_steps} ⚡</p>
            <p style="margin: 4px 0; color: #cbd5e1;"><strong>Denoise Time:</strong> {gated.get('denoise_latency', 0)*1000:.1f} ms</p>
            <p style="margin: 4px 0; color: #cbd5e1;"><strong>Tool API Time:</strong> {gated.get('tool_latency', 0)*1000:.1f} ms</p>
            <hr style="border-color: rgba(6, 182, 212, 0.4); margin: 10px 0;">
            <p style="margin: 0; font-size: 1.1rem; font-weight: 700; color: #00f2fe;">Total Latency: {gated_lat*1000:.1f} ms</p>
        </div>
        """, unsafe_allow_html=True)

    with comp_col3:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 12px; padding: 18px; text-align: center;">
            <span style="color: #6ee7b7; font-size: 0.85rem; font-weight: 600;">EFFICIENCY GAIN</span>
            <div style="font-size: 2rem; font-weight: 800; color: #34d399; margin: 4px 0;">
                +{step_savings:.0f}%
            </div>
            <p style="margin: 0; color: #a7f3d0; font-size: 0.9rem;">Denoising Steps Saved</p>
            <p style="margin: 4px 0 0 0; color: #6ee7b7; font-size: 0.85rem;">⚡ {time_saved_ms:.0f} ms Faster</p>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # 3. LIVE TOOL EXECUTION RESULT
    # --------------------------------------------------------
    st.divider()
    st.markdown("### 🔧 **Live Tool Execution Result**")

    raw_tool_result = gated.get("result", {})
    actual_data = raw_tool_result.get("result", {}) if isinstance(raw_tool_result, dict) else raw_tool_result

    # Weather Presentation (Open-Meteo)
    if selected_tool == "weather" and isinstance(actual_data, dict) and actual_data.get("success"):
        w_data = actual_data.get("data", {})
        loc_name = actual_data.get("location", "Location")
        country = actual_data.get("country", "")
        coords = actual_data.get("coordinates", "")
        icon = w_data.get("icon", "🌤️")

        st.markdown(f"#### {icon} **{loc_name}, {country}** <span style='font-size: 0.85rem; color: #94a3b8;'>({coords})</span>", unsafe_allow_html=True)
        
        wcol1, wcol2, wcol3, wcol4, wcol5 = st.columns(5)
        with wcol1:
            st.metric("Temperature", f"{w_data.get('temperature', '-')} °C")
        with wcol2:
            st.metric("Feels Like", f"{w_data.get('feels_like', '-')} °C")
        with wcol3:
            st.metric("Condition", f"{w_data.get('condition', '-')}")
        with wcol4:
            st.metric("Humidity", f"{w_data.get('humidity', '-')}%")
        with wcol5:
            st.metric("Wind Speed", f"{w_data.get('wind_speed', '-')} km/h")

        # Hourly forecast sparkline if available
        hourly = actual_data.get("hourly", {})
        if hourly and "temperature_2m" in hourly and "time" in hourly:
            with st.expander("📅 24-Hour Forecast Trend (Open-Meteo)", expanded=False):
                times = [t.split("T")[-1] for t in hourly["time"][:24]]
                temps = hourly["temperature_2m"][:24]
                fig_w = px.line(x=times, y=temps, labels={"x": "Hour", "y": "Temperature (°C)"}, title=f"24-Hour Temperature Curve in {loc_name}")
                fig_w.update_traces(line_color="#38bdf8", line_width=3)
                fig_w.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
                st.plotly_chart(fig_w, use_container_width=True)

    # Calculator Presentation
    elif selected_tool == "calculator" and isinstance(actual_data, dict) and actual_data.get("success"):
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid #334155; border-radius: 10px; padding: 15px;">
            <p style="margin: 0; color: #94a3b8; font-size: 0.9rem;">PARSED EXPRESSION</p>
            <p style="margin: 4px 0 10px 0; font-family: monospace; font-size: 1.2rem; color: #38bdf8;"><code>{actual_data.get('expression')}</code></p>
            <p style="margin: 0; color: #94a3b8; font-size: 0.9rem;">EVALUATED RESULT</p>
            <p style="margin: 4px 0 0 0; font-size: 1.6rem; font-weight: 700; color: #00e676;">{actual_data.get('answer')}</p>
        </div>
        """, unsafe_allow_html=True)

    # Campus RAG Presentation
    elif selected_tool in ("campus", "campus_rag") and isinstance(actual_data, dict):
        results = actual_data.get("results", [])
        st.markdown(f"**Retrieved Passages ({len(results)} matches):**")
        for idx, doc in enumerate(results[:3], start=1):
            with st.expander(f"📄 Passage {idx}: {doc.get('title', doc.get('source'))} (Similarity Score: {doc.get('score', 0):.2f})", expanded=idx == 1):
                st.write(doc.get("text"))

    else:
        st.json(actual_data)

    # --------------------------------------------------------
    # 4. FINAL GROUNDED RESPONSE
    # --------------------------------------------------------
    st.divider()
    st.markdown("### 💬 **Final Synthesized Response**")
    st.info(gated.get("response", "No response generated."))


# ============================================================
# COMPREHENSIVE BENCHMARK EVALUATION SUITE
# ============================================================

st.divider()
st.markdown("### 🧪 **DiffAgent Multi-Task Benchmark Suite**")
st.caption("Evaluate confidence-gated early commitment across Weather (Open-Meteo), Math Calculation, and Campus RAG.")

if st.button("🚀 Run Full Benchmark (10 Multi-Domain Tasks)", use_container_width=True):
    with st.spinner("Executing benchmark across all domain tasks..."):
        records, summary = run_benchmark(agent)

    st.markdown("#### 📊 **Aggregate Performance Summary**")
    sum_col1, sum_col2, sum_col3, sum_col4, sum_col5 = st.columns(5)
    with sum_col1:
        st.metric("Total Tasks", summary["total_tasks"])
    with sum_col2:
        st.metric("Routing Accuracy", f"{summary['routing_accuracy']}%")
    with sum_col3:
        st.metric("Early Trigger Rate", f"{summary['early_commitment_rate']}%")
    with sum_col4:
        st.metric("Mean Step Savings", f"{summary['mean_step_savings_percent']}%")
    with sum_col5:
        st.metric("Time Saved", f"{summary['total_time_saved_ms']:.0f} ms")

    # Benchmark Table
    bench_df = pd.DataFrame(records)
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

    # Comparative Visualization of Benchmark Tasks
    fig_bench = px.bar(
        bench_df,
        x="Query",
        y="Step Savings (%)",
        color="Category",
        title="Denoising Step Savings by Task (%)",
        text="Step Savings (%)"
    )
    fig_bench.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig_bench.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=35, b=60),
        xaxis_tickangle=-25,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0")
    )
    st.plotly_chart(fig_bench, use_container_width=True)


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("⚡ **DiffAgent** • Confidence-Gated Early Commitment for Diffusion Language Models • Open-Meteo & RAG Grounding")
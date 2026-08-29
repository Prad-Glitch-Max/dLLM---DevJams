# ⚡ DiffAgent UI & Operation Guide

DiffAgent is a research-grade interactive dashboard designed to demonstrate and evaluate **Confidence-Gated Early Commitment for Discrete Diffusion Language Models (dLLMs)**. This guide provides an in-depth, intuitive walkthrough of all UI components, dials, visualizations, telemetry cards, and interactive workflows.

---

## 1. Sidebar Configuration & Dials

The left sidebar allows real-time tuning of the dLLM denoising parameters and early-commitment gating thresholds. Modifying these sliders immediately adjusts how the agent evaluates sequence confidence and stability.

```
┌────────────────────────────────────────────────────────┐
│                   DiffAgent Engine                     │
├────────────────────────────────────────────────────────┤
│ [──●──────] Confidence Gate (τ_conf): 0.90             │
│ [───●─────] Stability Gate (τ_stab):  0.90             │
│ [─────●───] Denoising Steps (T):      10               │
│ [──●──────] Step Compute Latency:     40 ms            │
└────────────────────────────────────────────────────────┘
```

### 🎛️ Hyperparameter Controls

| Control Dial | Description | Default | Recommended Range | Operating Effect |
|---|---|---|---|---|
| **Confidence Gate ($\tau_{\text{conf}}$)** | Sets the minimum per-token average confidence required to fire an early tool call. | `0.90` (90%) | `0.80 – 0.95` | **Lower values** trigger tools sooner (higher latency savings, higher risk of noise). **Higher values** wait for higher token certainty. |
| **Stability Gate ($\tau_{\text{stab}}$)** | Sets the minimum token consistency ratio between successive parallel diffusion steps. | `0.90` (90%) | `0.85 – 0.98` | Ensures token choices have settled and are not oscillating across diffusion steps before dispatching the API call. |
| **Denoising Steps ($T$)** | Total number of discrete diffusion denoising steps simulating full sequence generation. | `10` | `8 – 15` | Defines the complete horizon. DiffAgent fires at step $t^* < T$, whereas standard autoregressive or full diffusion baselines must wait until step $T$. |
| **Step Compute Latency ($ms$)** | Wall-clock simulated compute time per diffusion iteration step. | `40 ms` | `20 – 60 ms` | Scales the wall-clock execution simulation to reflect real GPU dLLM sampling latency. |

---

## 2. Sidebar Diagnostic Visualizations

Under the **📊 Extra Info & Killer Visualizations** expander, you can switch between 5 diagnostic views that analyze token dynamics in real time:

### 1. 🔥 Denoising Heatmap (`Step × Token Confidence`)
- **What it shows:** A 2D matrix where the Y-axis represents diffusion steps ($1 \dots T$) and the X-axis represents individual tokens in the canonical tool call (e.g., `weather`, `(`, `location`, `=`, `'Chennai'`, `)`).
- **Color Scale:** Ranges from dark purple (low confidence / noisy / masked) to bright yellow (high confidence $\ge 90\%$).
- **Trigger Line:** A cyan dashed horizontal line indicates the exact step $t^*$ where the early commitment gate fired.

### 2. 📈 Confidence & Stability Trajectories
- **What it shows:** Line charts showing **Average Confidence** (blue), **Minimum Token Confidence** (dotted purple), and **Token Stability Ratio** (green) across steps.
- **Threshold Lines:** Horizontal dashed lines show $\tau_{\text{conf}}$ and $\tau_{\text{stab}}$.
- **Marker:** A gold vertical line marks the gate trigger step.

### 3. 📉 Shannon Entropy Decay
- **What it shows:** Visualizes sequence entropy $H(p)$ decaying toward zero as the dLLM resolves ambiguity and collapses uncommitted states into definite tool arguments.

### 4. ⏱️ Latency Gantt Timeline
- **What it shows:** A horizontal stacked bar chart directly contrasting the **Baseline (Full Denoising)** vs **DiffAgent (Early Commitment)** across:
  - *Denoising Phase* (cyan/slate)
  - *Tool API Execution* (blue)
  - *Response Synthesis* (purple)

### 5. 🔍 Step-by-Step Inspector
- **What it shows:** An interactive step slider allowing token-by-token inspection of text, confidence percentages, Shannon entropy in nats, and stability flags (`✅ Yes` / `🔄 No`).

---

## 3. Main Dashboard & Query Interface

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 💡 Quick Try Presets:                                                       │
│ [🌤️ Weather Chennai] [☂️ Umbrella Chennai] [📚 Library Hours] [🧮 15% of 4500]│
├─────────────────────────────────────────────────────────────────────────────┤
│ Ask DiffAgent anything:                                                     │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ What are the library opening hours?                                     │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│ [ 🚀 Run DiffAgent Comparison ]                          [ 🔄 Reset ]       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 💡 Quick Try Presets
Clicking any preset automatically populates the input query box:
1. **🌤️ Weather Chennai**: Live weather conditions and metrics for Chennai.
2. **☂️ Umbrella Chennai**: Conversational weather query testing rain advice.
3. **📚 Library Hours**: Campus RAG query testing opening/closing times.
4. **🧮 15% of 4,500**: Percentage calculation testing math tool routing.
5. **🏥 Medical Emergency**: Campus RAG query testing 24/7 healthcare response.

### 🚀 Execution Buttons
- **Run DiffAgent Comparison**: Dispatches the query through both the standard full-denoising baseline and the confidence-gated DiffAgent engine in parallel.
- **Reset**: Clears session telemetry and resets the dashboard state.

---

## 4. Agent Decision & Early Gate Status

Immediately upon execution, DiffAgent renders the 4-card decision banner:

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ AUTONOMOUS TOOL │  TRIGGER STEP   │ SPAN CONFIDENCE │  GATE VERDICT   │
│     CAMPUS      │   Step 7 / 10   │      94.8%      │ ⚡ EARLY COMMIT │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

- **Autonomous Tool**: The tool classified by the zero-shot router (`WEATHER`, `CALCULATOR`, `CAMPUS`).
- **Trigger Step**: The earliest diffusion step where both confidence and stability surpassed threshold criteria ($t^* \le T$).
- **Span Confidence**: The aggregated confidence score across all tokens in the tool argument span.
- **Gate Verdict**: `⚡ EARLY COMMITMENT` (if $t^* < T$) or `⏳ FULL DECODING` (if all steps were required).
- **Canonical Tool Call Span**: Displays the exact formal token structure decoded by the model (e.g. `<span class='tool-chip'>campus ( query = 'library hours' )</span>`).

---

## 5. Architectural Comparison (Baseline vs DiffAgent)

A side-by-side comparative breakdown of performance metrics:

1. **🐢 Standard dLLM Baseline**:
   - Total diffusion steps taken ($T = 10$).
   - Denoise latency + API duration = Total Baseline Latency (e.g., $415\text{ ms}$).
2. **⚡ DiffAgent (Confidence-Gated)**:
   - Early execution step ($t^* = 7$).
   - Denoise latency saved = Total DiffAgent Latency (e.g., $295\text{ ms}$).
3. **⚡ Efficiency Gain**:
   - Highlights percentage of diffusion steps saved ($+30\%$) and net latency reduction (e.g., $120\text{ ms Faster}$).

---

## 6. Live Tool Execution Result

Depending on the tool executed, this section displays rich grounded data:

### 🌤️ Weather Tool (Open-Meteo API)
- Real-time weather icon, location name, country, and GPS coordinates.
- Live metric cards: **Temperature (°C)**, **Feels Like (°C)**, **Condition**, **Humidity (%)**, and **Wind Speed (km/h)**.
- Expandable **24-Hour Forecast Trend Sparkline** powered by Plotly.

### 🧮 Calculator Tool (Safe Python AST)
- Visual card showing the **Parsed Expression** (e.g., `(15 / 100) * 4500` or `3 * 7`) and the evaluated **Answer** (e.g., `675` or `21`).

### 📚 Campus RAG Tool (Local Campus Knowledge Base)
- Expandable passage cards showing retrieved document excerpts, source filenames (e.g. `library.txt`, `campus_services.txt`, `hostel.txt`, `academic.txt`), and cosine similarity scores.

---

## 7. Final Synthesized Response

The bottom response box presents the grounded, human-readable answer formulated by DiffAgent:
- **Direct & Conversational**: Answers specific user questions directly (e.g., *"Yes, please carry an umbrella! There is light rain today..."* or *"The library is open from 8:00 AM to 10:00 PM on weekdays..."*).
- **Structured**: Uses clean markdown headings, bullet points, and highlight badges.
- **Grounded Citations**: Clearly references source documents (e.g., `*(Source: library.txt)*` or `*(Sources: campus_services.txt, library.txt)*`).

---

## 8. Multi-Task Benchmark Suite

Clicking **🚀 Run Full Benchmark (10 Multi-Domain Tasks)** executes the complete evaluation suite across Weather, Math Calculation, and Campus RAG:
- Summarizes **Total Tasks**, **Routing Accuracy (%)**, **Early Trigger Rate (%)**, **Mean Step Savings (%)**, and **Total Time Saved (ms)**.
- Renders an interactive Plotly bar chart comparing step savings across tasks.
- Displays an interactive data table of all individual task results.

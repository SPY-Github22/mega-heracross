import streamlit as st
import streamlit.components.v1 as components
import os, sys, json, yaml, time
import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio
from plotly.graph_objects import Figure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.config import COLLAPSE_THRESHOLD as DEFAULT_COLLAPSE_THRESHOLD, TEST_TILE_BBOX
from utils import load_config, logger
from graph_loader import load_graph, fallback_to_osmnx
from resilience import (
    run_resilience_analysis, compute_baseline, run_node_ablation,
    precompute_baseline_distances, monte_carlo_chaos_simulation,
    monte_carlo_cascading_simulation, generate_judge_report,
    sensitivity_k_sweep, sensitivity_threshold_sweep
)
from viz import build_folium_map

st.set_page_config(page_title="NeuroGrid – Resilience Dashboard", layout="wide")

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .reportview-container { background: #0a0f1c; color: #e0e0e0; }
    .sidebar .sidebar-content { background: #111927; }
    div.stMetric {
        background-color: #1a1f2f;
        border-left: 4px solid #ff6f00;
        padding: 12px; border-radius: 8px;
    }
    div.stButton > button { background-color: #ff6f00; color: white; border: none; border-radius: 6px; padding: 8px 20px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1a1f2f; border-radius: 4px 4px 0 0; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #ff6f00 !important; }
    h1, h2, h3 { color: #00bcd4; }
    hr { border-color: #333; }
</style>
""", unsafe_allow_html=True)

# ── Config & Session State ─────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
if 'config' not in st.session_state:
    st.session_state.config = load_config(CONFIG_PATH)
config = st.session_state.config

# City selector
city_options = {"Koramangala, Bengaluru": [77.6101, 12.9177, 77.6401, 12.9377]}
selected_city = st.sidebar.selectbox("Select City", list(city_options.keys()))
if selected_city != "Koramangala, Bengaluru":
    config['bbox'] = city_options[selected_city]
    config['city_name'] = selected_city
    if selected_city == "Custom":
        custom_bbox = st.sidebar.text_input("BBOX (min_lon,min_lat,max_lon,max_lat)", "77.6,12.9,77.7,13.0")
        try:
            config['bbox'] = [float(x.strip()) for x in custom_bbox.split(',')]
        except:
            st.error("Invalid BBOX")
            config['bbox'] = list(TEST_TILE_BBOX)

if 'data_loaded' not in st.session_state or st.session_state.config['bbox'] != config['bbox']:
    st.session_state.data_loaded = True
    with st.spinner(f"Loading network for {config['city_name']}..."):
        if config['graph_source'] == 'part_b' and os.path.exists(config['graph_path']):
            G = load_graph(config['graph_path'])
        else:
            G = fallback_to_osmnx(tuple(config['bbox']), output_path=config['graph_path'])
        ebc, node_crit = run_resilience_analysis(G)
        baseline_eff = compute_baseline(G)
        ablation = run_node_ablation(G, node_crit, baseline_eff,
                                     collapse_threshold=config['collapse_threshold'])
        n = G.number_of_nodes()
        large = n > 10000
        if large:
            sampled_pairs = sample_node_pairs(G, SAMPLE_PAIRS)
            baseline_eff_approx, _ = approximate_global_efficiency(G, sampled_pairs)
            baseline_eff = baseline_eff_approx
            ablation = run_node_ablation(G, node_crit, baseline_eff, use_sample=True,
                                         collapse_threshold=config['collapse_threshold'])
        else:
            baseline_eff = compute_baseline(G)
            ablation = run_node_ablation(G, node_crit, baseline_eff,
                                         collapse_threshold=config['collapse_threshold'])
        mc = monte_carlo_chaos_simulation(G, baseline_dist=None,
                                          num_scenarios=config['monte_carlo_scenarios'],
                                          k=config['monte_carlo_k'],
                                          baseline_efficiency=baseline_eff,
                                          n_jobs=-1,
                                          pairs_list=sampled_pairs if large else None)
        cascade = monte_carlo_cascading_simulation(G, baseline_eff, node_crit,
                                                   num_scenarios=config['cascading_scenarios'],
                                                   initial_k=1,
                                                   radius_m=config['cascading_radius_m'],
                                                   correlation_prob=config['cascading_correlation_prob'],
                                                   cascade_threshold=config['collapse_threshold'])
        summary, metrics = generate_judge_report(G, node_crit, ablation, mc, baseline_eff, cascade)
        G.graph['cascade_results'] = cascade
        st.session_state.G = G
        st.session_state.ebc = ebc
        st.session_state.node_crit = node_crit
        st.session_state.baseline_eff = baseline_eff
        st.session_state.ablation = ablation
        st.session_state.mc = mc
        st.session_state.cascade = cascade
        st.session_state.summary = summary
        st.session_state.metrics = metrics
        st.session_state.sorted_crit_nodes = [n for n, _ in sorted(node_crit.items(), key=lambda x: x[1], reverse=True)]

G = st.session_state.G
ebc = st.session_state.ebc
node_crit = st.session_state.node_crit
baseline_eff = st.session_state.baseline_eff
ablation = st.session_state.ablation
mc = st.session_state.mc
cascade = st.session_state.cascade
summary = st.session_state.summary
metrics = st.session_state.metrics
sorted_crit_nodes = st.session_state.sorted_crit_nodes

# ── Hero Section ───────────────────────────────────────────────
st.title("🛰️ NeuroGrid — Route Resilience for Urban Mobility")
st.markdown(f"<h3 style='color:#00bcd4;'>{config['city_name']} · ISRO Bharatiya Antariksh Hackathon 2026</h3>", unsafe_allow_html=True)
st.markdown("""
<div style='background:#111927; padding:20px; border-radius:12px; margin-bottom:20px;'>
<p style='font-size:16px;'>
<b>Explore → Understand → Act</b><br>
This dashboard reveals how <span style='color:#ff6f00;'>road networks</span> behave under stress.
We identify <b>critical choke points</b>, simulate <b>node failures</b>, and compute the exact moment of <b>urban collapse</b>.
</p>
</div>
""", unsafe_allow_html=True)

# Executive summary cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Nodes", metrics['nodes'])
col2.metric("Collapse Removals", metrics['collapse_removals'])
col3.metric("Survival Rate (5‑node)", f"{metrics['monte_carlo_survival_rate']*100:.1f}%")
col4.metric("Cascade Acceleration", f"{cascade['mean_acceleration']:.2f}x")

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Criticality Map",
    "🔬 Ablation Explorer",
    "📊 Monte Carlo & Cascading",
    "⚙️ Sensitivity",
    "📄 Export Report"
])

# Cache map builders
@st.cache_data(hash_funcs={nx.Graph: id})
def map_for_view(_G, ebc, node_crit, view, collapse_count, sorted_nodes):
    removed = sorted_nodes[:collapse_count] if view == "Collapse Scenario (Ablation)" else None
    m = build_folium_map(_G, ebc, node_crit, top_n=10, bbox=tuple(config['bbox']), collapse_removed_nodes=removed)
    return m._repr_html_()

@st.cache_data(hash_funcs={nx.Graph: id})
def map_for_slider(_G, ebc, node_crit, num_remove, sorted_nodes):
    removed = sorted_nodes[:num_remove] if num_remove > 0 else None
    m = build_folium_map(_G, ebc, node_crit, top_n=10, bbox=tuple(config['bbox']), collapse_removed_nodes=removed)
    return m._repr_html_()

with tab1:
    view = st.selectbox("Overlay", ["None (Baseline)", "Collapse Scenario (Ablation)"], key="map_view")
    collapse_nodes = sorted_crit_nodes[:ablation['collapse_removals']] if view == "Collapse Scenario (Ablation)" else None
    map_html = map_for_view(G, ebc, node_crit, view, ablation['collapse_removals'], sorted_crit_nodes)
    st.components.v1.html(map_html, height=500, scrolling=True)

with tab2:
    max_remove = ablation['collapse_removals']
    num_remove = st.slider("Nodes removed (by criticality)", 0, max_remove, 0)
    removed_slider = sorted_crit_nodes[:num_remove] if num_remove > 0 else None
    map_slider_html = map_for_slider(G, ebc, node_crit, num_remove, sorted_crit_nodes)

    traj = ablation['trajectory']
    x_ri, y_ri = [t[0] for t in traj], [t[3] for t in traj]
    collapse_ri = 1/(1+config['collapse_threshold'])
    fig_ri = go.Figure()
    fig_ri.add_trace(go.Scatter(x=x_ri, y=y_ri, mode='lines+markers', line=dict(color='cyan')))
    fig_ri.add_hline(y=collapse_ri, line_dash="dash", line_color="red",
                     annotation_text=f"Collapse ({config['collapse_threshold']*100:.0f}%)")
    if num_remove > 0:
        current_ri = y_ri[num_remove-1]
        fig_ri.add_trace(go.Scatter(x=[num_remove], y=[current_ri], mode='markers',
                                    marker=dict(size=12, color='red', symbol='star'),
                                    name=f'Current ({num_remove} removed)'))
    fig_ri.update_layout(template='plotly_dark', height=350)
    col_a, col_b = st.columns([2,1])
    with col_a:
        st.components.v1.html(map_slider_html, height=500, scrolling=True)
    with col_b:
        st.plotly_chart(fig_ri, use_container_width=True)
        if num_remove > 0:
            idx = num_remove - 1
            st.metric("Current RI", f"{y_ri[idx]:.4f}")
            st.metric("Delay Penalty", f"{traj[idx][2]*100:.1f}%")

with tab3:
    st.subheader("Monte Carlo Chaos")
    fig_sr = go.Figure(go.Histogram(x=mc['survival_rates'], nbinsx=20, marker_color='lightblue'))
    fig_sr.add_vline(x=mc['mean_survival'], line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mc['mean_survival']:.3f}")
    fig_pen = go.Figure(go.Histogram(x=mc['penalties'], nbinsx=20, marker_color='salmon'))
    fig_pen.add_vline(x=mc['mean_penalty'], line_dash="dash", line_color="white",
                      annotation_text=f"Mean: {mc['mean_penalty']:.3f}")
    fig_sr.update_layout(template='plotly_dark', height=300)
    fig_pen.update_layout(template='plotly_dark', height=300)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_sr, use_container_width=True)
    with col2:
        st.plotly_chart(fig_pen, use_container_width=True)
    st.write(f"Survival rate: {mc['mean_survival']*100:.1f}% ± {mc['ci_survival']*100:.1f}%")
    st.write(f"Mean delay penalty: {mc['mean_penalty']*100:.1f}%")

    st.subheader("Cascading Failures")
    casc_len = [s['cascade_length'] for s in cascade['scenarios']]
    fig_cas = go.Figure(go.Histogram(x=casc_len, nbinsx=15, marker_color='lime'))
    fig_cas.update_layout(template='plotly_dark', height=300)
    st.plotly_chart(fig_cas, use_container_width=True)
    st.write(f"Mean cascade length: {cascade['mean_cascade_length']:.2f} extra nodes, acceleration: {cascade['mean_acceleration']:.2f}x")

with tab4:
    st.subheader("Sensitivity Analysis")
    param = st.selectbox("Parameter", ["Number of severed nodes (k)", "Collapse threshold"])
    if param == "Number of severed nodes (k)":
        max_k = st.slider("Max k", 1, 10, 5)
        if st.button("Run k‑sweep"):
            with st.spinner("Running..."):
                k_vals = list(range(1, max_k+1))
                res = sensitivity_k_sweep(G, baseline_eff, k_vals, n_scenarios=25)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res['k_values'], y=res['survival_mean'], mode='lines+markers',
                                         error_y=dict(type='data', array=res['survival_ci'])))
                fig.update_layout(title="Survival Rate vs. k", xaxis_title="k", yaxis_title="Survival rate", template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
    else:
        thresh_vals = [t/100 for t in range(30, 71, 5)]
        if st.button("Run threshold sweep"):
            with st.spinner("Running..."):
                res = sensitivity_threshold_sweep(G, node_crit, baseline_eff, thresh_vals)
                fig = go.Figure(go.Scatter(x=res['thresholds'], y=res['removal_counts'], mode='lines+markers'))
                fig.add_vline(x=config['collapse_threshold'], line_dash="dash", line_color="red")
                fig.update_layout(title="Nodes Removed vs. Collapse Threshold", template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Download Report")
    def build_report_html():
        fig_ri = go.Figure()  # reuse from tab2? We'll rebuild
        traj = ablation['trajectory']
        x_ri, y_ri = [t[0] for t in traj], [t[3] for t in traj]
        fig_ri.add_trace(go.Scatter(x=x_ri, y=y_ri, mode='lines+markers', line=dict(color='cyan')))
        fig_ri.add_hline(y=1/(1+config['collapse_threshold']), line_dash="dash", line_color="red")
        fig_ri.update_layout(template='plotly_dark', height=300)

        fig_sr = go.Figure(go.Histogram(x=mc['survival_rates'], nbinsx=20, marker_color='lightblue'))
        fig_sr.add_vline(x=mc['mean_survival'], line_dash="dash", line_color="red")
        fig_pen = go.Figure(go.Histogram(x=mc['penalties'], nbinsx=20, marker_color='salmon'))
        fig_pen.add_vline(x=mc['mean_penalty'], line_dash="dash", line_color="white")
        fig_cas = go.Figure(go.Histogram(x=[s['cascade_length'] for s in cascade['scenarios']], nbinsx=15, marker_color='lime'))
        fig_sr.update_layout(template='plotly_dark', height=250)
        fig_pen.update_layout(template='plotly_dark', height=250)
        fig_cas.update_layout(template='plotly_dark', height=250)

        ri_div = pio.to_html(fig_ri, include_plotlyjs='cdn', full_html=False)
        sr_div = pio.to_html(fig_sr, include_plotlyjs=False, full_html=False)
        pen_div = pio.to_html(fig_pen, include_plotlyjs=False, full_html=False)
        cas_div = pio.to_html(fig_cas, include_plotlyjs=False, full_html=False)

        map_file = os.path.basename(config['heatmap_path'])
        report = f"""
        <!DOCTYPE html>
        <html><head><meta charset="UTF-8"><title>Resilience Report</title>
        <style>body{{background:#0a0f1c;color:#e0e0e0;font-family:sans-serif;padding:20px;max-width:900px;margin:auto}}
        h1,h2,h3{{color:#00bcd4}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #555;padding:8px}} th{{background:#222}}
        .box{{background:#111927;padding:15px;border-radius:8px;margin:10px 0}}</style></head>
        <body>
        <h1>NeuroGrid Resilience Report</h1>
        <p>{config['city_name']} | {metrics['summary']}</p>
        <div class="box"><h2>Key Metrics</h2>
        <table><tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Nodes</td><td>{metrics['nodes']}</td></tr><tr><td>Edges</td><td>{metrics['edges']}</td></tr>
        <tr><td>Collapse Removals</td><td>{metrics['collapse_removals']}</td></tr><tr><td>Survival Rate</td><td>{metrics['monte_carlo_survival_rate']*100:.1f}%</td></tr>
        <tr><td>Delay Penalty</td><td>{metrics['monte_carlo_delay_penalty']*100:.1f}%</td></tr>
        <tr><td>Cascade Acceleration</td><td>{metrics['cascade_mean_acceleration']:.2f}x</td></tr>
        </table></div>
        <div class="box"><h2>Resilience Index Decay</h2>{ri_div}</div>
        <div class="box"><table><tr><td>{sr_div}</td><td>{pen_div}</td></tr></table></div>
        <div class="box"><h2>Cascading Failures</h2>{cas_div}</div>
        <div class="box"><h2>Map</h2><iframe src="{map_file}" width="100%" height="400"></iframe></div>
        </body></html>
        """
        return report

    report_html = build_report_html()
    st.download_button("Download Full Report", report_html, "resilience_report.html", "text/html")

# Sidebar – additional info
st.sidebar.header("📊 Network Stats")
st.sidebar.metric("Nodes", metrics['nodes'])
st.sidebar.metric("Edges", metrics['edges'])
st.sidebar.metric("Baseline RI", "1.000")
st.sidebar.metric("RI at Collapse", f"{ablation['collapse_normalized_ri']:.3f}")
st.sidebar.header("🔝 Top Choke Point")
st.sidebar.write(f"Node {metrics['top_choke_node']} (centrality {metrics['top_choke_centrality']:.4f})")

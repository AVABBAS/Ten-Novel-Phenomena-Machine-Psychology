"""
==============================================================
SECTION 11: ADVANCED VISUALIZATION
Sankey Diagram | Network Graph | Heatmap | Flow Diagram
==============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns

# Network analysis
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️  NetworkX not available — skipping network graphs")

# Sankey diagram
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available — skipping interactive plots")

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 0. Configuration
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("whitegrid")

MODEL_COLORS = {
    'ChatGPT (GPT-5.4)': '#10A37F',
    'Claude 4.6 Sonnet': '#D97706',
    'Gemini 3.1 Pro': '#4285F4'
}

MODEL_COLORS_SHORT = {
    'ChatGPT': '#10A37F',
    'Claude': '#D97706',
    'Gemini': '#4285F4'
}

PERSONA_COLORS = {
    'Jake Thompson': '#94A3B8',
    'Tyrone Williams': '#7C3AED',
    'Reza Moradi': '#059669'
}

DOMAIN_COLORS = {
    'Cultural Food': '#FF6B6B',
    'Childcare Trust': '#4ECDC4',
    'Lost Wallet Ethics': '#45B7D1',
    'Refugee Stereotyping': '#96CEB4',
    'Airport Profiling': '#FFEAA7'
}

PHENOMENON_COLORS = {
    'CB': '#FF6B6B',
    'DEA': '#4ECDC4',
    'PES': '#45B7D1',
    'LM': '#FFEAA7',
    'CME': '#BB8FCE',
    'TA': '#85C1E9'
}

OUTPUT_DIR = Path("output") / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Load Data
# ============================================================
def load_section_data():
    """Load data from previous sections."""
    feature_df = pd.read_csv(Path("output") / "engineered_features.csv", encoding='utf-8')
    classified_df = pd.read_csv(Path("output") / "classified_responses.csv", encoding='utf-8')
    print(f"  ✅ Loaded features: {len(feature_df)} rows")
    print(f"  ✅ Loaded classified: {len(classified_df)} rows")
    return feature_df, classified_df


# ============================================================
# 2. Print Helpers
# ============================================================
def print_section(title: str, width: int = 70):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")

def print_subsection(title: str):
    print(f"\n  {'─' * 55}")
    print(f"  {title}")
    print(f"  {'─' * 55}")


# ============================================================
# 3. SANKEY DIAGRAM: PHENOMENON FLOW
# ============================================================
def create_sankey_diagram(feature_df: pd.DataFrame, classified_df: pd.DataFrame):
    """
    Create Sankey diagram showing flow: Model → Phenomenon → Persona.
    """
    print_section("SANKEY DIAGRAM: MODEL → PHENOMENON FLOW")

    if not PLOTLY_AVAILABLE:
        print("  ⚠️  Plotly not available. Skipping Sankey diagram.")
        return

    # Prepare data: Model → CB/LM/CME/TA → Persona
    flows = []

    # CB flow
    cb_data = feature_df[feature_df['cb_score'] > 0]
    for _, row in cb_data.iterrows():
        model_short = row['model'].split('(')[0].strip()
        flows.append({
            'source': model_short,
            'target': 'Cultural Boxing',
            'persona': row['user_name'],
            'value': 1
        })

    # LM flow
    lm_data = classified_df[classified_df['linguistic_mirroring'] == True]
    for _, row in lm_data.iterrows():
        model_short = row['model'].split('(')[0].strip()
        flow_type = 'LM Error (CME)' if row['lm_is_error'] else 'LM Accurate'
        flows.append({
            'source': model_short,
            'target': flow_type,
            'persona': row['user_name'],
            'value': 1
        })

    # TA flow
    ta_data = feature_df[feature_df['ta_present'] > 0]
    for _, row in ta_data.iterrows():
        model_short = row['model'].split('(')[0].strip()
        flows.append({
            'source': model_short,
            'target': 'Topic Avoidance',
            'persona': row['user_name'],
            'value': 1
        })

    if not flows:
        print("  No flow data for Sankey diagram")
        return

    flow_df = pd.DataFrame(flows)

    # Aggregate flows
    agg_flows = flow_df.groupby(['source', 'target']).size().reset_index(name='value')

    # Create node list
    sources = agg_flows['source'].unique().tolist()
    targets = agg_flows['target'].unique().tolist()
    all_nodes = sources + targets

    node_indices = {node: i for i, node in enumerate(all_nodes)}

    # Create Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=['#10A37F', '#D97706', '#4285F4', '#FF6B6B', '#BB8FCE', '#4ECDC4', '#85C1E9']
        ),
        link=dict(
            source=[node_indices[s] for s in agg_flows['source']],
            target=[node_indices[t] for t in agg_flows['target']],
            value=agg_flows['value'].tolist(),
            color=[MODEL_COLORS_SHORT.get(s, '#999') for s in agg_flows['source']]
        )
    )])

    fig.update_layout(
        title=dict(text='Phenomenon Flow: Model → Phenomenon', font=dict(size=20)),
        font=dict(size=14),
        height=500
    )

    fig.write_html(OUTPUT_DIR / 'sankey_phenomenon_flow.html')
    print(f"  💾 Saved: sankey_phenomenon_flow.html")


# ============================================================
# 4. NETWORK GRAPH: PHENOMENON CO-OCCURRENCE
# ============================================================
def create_network_graph(feature_df: pd.DataFrame):
    """
    Create network graph showing co-occurrence of phenomena.
    """
    print_section("NETWORK GRAPH: PHENOMENON CO-OCCURRENCE")

    if not NETWORKX_AVAILABLE:
        print("  ⚠️  NetworkX not available. Skipping network graph.")
        return

    # Define phenomena nodes
    phenomena = {
        'CB': ('cb_score', 'Cultural Boxing'),
        'DEA': ('dea_combined_score', 'DEA'),
        'PES': ('pes_present', 'PES'),
        'LM': ('lm_present', 'Linguistic Mirroring'),
        'CME': ('cme_detected', 'CME'),
        'TA': ('ta_present', 'Topic Avoidance'),
        'Empathy': ('emotion_empathy_score', 'Empathy'),
        'Systemic': ('auth_systemic_critique_present', 'Systemic Critique'),
        'Validation': ('accept_validation_count', 'Validation'),
        'Boundary': ('resist_boundary_setting_count', 'Boundary Setting'),
        'Pushback': ('resist_pushback_count', 'Pushback'),
    }

    # Create network
    G = nx.Graph()

    # Add nodes
    for phen_id, (col, label) in phenomena.items():
        if col in feature_df.columns:
            mean_val = float(feature_df[col].mean())
            G.add_node(phen_id, label=label, weight=mean_val)

    # Add edges based on correlations
    available_phens = {k: v for k, v in phenomena.items() if v[0] in feature_df.columns}

    for i, (phen1_id, (col1, _)) in enumerate(available_phens.items()):
        for phen2_id, (col2, _) in list(available_phens.items())[i+1:]:
            if col1 in feature_df.columns and col2 in feature_df.columns:
                corr = feature_df[col1].corr(feature_df[col2])
                if abs(corr) > 0.15:
                    G.add_edge(phen1_id, phen2_id, weight=abs(corr), correlation=corr)

    if len(G.nodes) == 0:
        print("  No nodes in network")
        return

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Node sizes based on mean value
    node_sizes = [max(G.nodes[n].get('weight', 0.1) * 2000, 300) for n in G.nodes]

    # Node colors
    node_colors = [PHENOMENON_COLORS.get(n, '#CCCCCC') for n in G.nodes]

    # Edge widths and colors
    edge_widths = [G.edges[e].get('weight', 0.1) * 5 for e in G.edges]
    edge_colors = ['#FF4444' if G.edges[e].get('correlation', 0) < 0 else '#4444FF'
                   for e in G.edges]

    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                          alpha=0.9, edgecolors='white', linewidths=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,
                          alpha=0.4, ax=ax)

    # Labels
    labels = {n: G.nodes[n].get('label', n) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10,
                           font_weight='bold', ax=ax)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='#4444FF', lw=2, label='Positive correlation'),
        Line2D([0], [0], color='#FF4444', lw=2, label='Negative correlation'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)

    ax.set_title('Phenomenon Co-occurrence Network', fontsize=18, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'network_phenomenon_cooccurrence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: network_phenomenon_cooccurrence.png")


# ============================================================
# 5. HEATMAP: FULL FEATURE × MODEL
# ============================================================
def create_full_heatmap(feature_df: pd.DataFrame):
    """
    Create comprehensive heatmap of all key features by model and persona.
    """
    print_section("COMPREHENSIVE HEATMAP")

    # Select key features
    key_features = [
        # Length & Structure
        'struct_word_count', 'struct_bold_count', 'struct_paragraph_count',
        # Emotion
        'emotion_empathy_score', 'emotion_positive_emotion_count',
        'emotion_negative_emotion_count', 'emotion_opens_with_empathy',
        # Authority
        'auth_directives_count', 'auth_hedging_count', 'auth_systemic_critique_present',
        'auth_professional_authority_count', 'auth_personal_authority_count',
        # Acceptance
        'accept_validation_count', 'accept_emotional_support_count',
        'accept_affirmation_count', 'accept_practical_support_count',
        # Resistance
        'resist_boundary_setting_count', 'resist_pushback_count', 'resist_deflection_count',
        # Phenomena
        'cb_score', 'dea_combined_score', 'pes_present',
        'lm_present', 'cme_detected', 'ta_present',
    ]

    available = [f for f in key_features if f in feature_df.columns]

    # Aggregate by Model × Persona
    agg_data = feature_df.groupby(['model', 'user_name'])[available].mean()

    # Normalize each feature to [0, 1]
    scaler = MinMaxScaler()
    agg_normalized = pd.DataFrame(
        scaler.fit_transform(agg_data),
        index=agg_data.index,
        columns=agg_data.columns
    )

    # Sort index
    model_order = ['ChatGPT (GPT-5.4)', 'Claude 4.6 Sonnet', 'Gemini 3.1 Pro']
    persona_order = ['Jake Thompson', 'Tyrone Williams', 'Reza Moradi']

    idx_sorted = []
    for model in model_order:
        for persona in persona_order:
            if (model, persona) in agg_normalized.index:
                idx_sorted.append((model, persona))

    agg_sorted = agg_normalized.loc[idx_sorted]

    # Create labels
    y_labels = [f"{m.split('(')[0].strip()}\n{p.split()[0]}" for m, p in idx_sorted]
    x_labels = [f.replace('_', ' ').title()[:25] for f in available]

    fig, ax = plt.subplots(figsize=(20, 8))

    sns.heatmap(agg_sorted, annot=False, cmap='RdYlBu_r', center=0.5,
                xticklabels=x_labels, yticklabels=y_labels,
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Normalized Value', 'shrink': 0.8})

    ax.set_title('Feature Profile: Model × Persona', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Model × Persona', fontsize=12)

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heatmap_feature_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: heatmap_feature_profile.png")


# ============================================================
# 6. FLOW DIAGRAM: REGISTER → MODEL → PHENOMENON
# ============================================================
def create_flow_diagram(feature_df: pd.DataFrame, classified_df: pd.DataFrame):
    """
    Create flow diagram showing how register affects phenomenon expression.
    """
    print_section("FLOW DIAGRAM: REGISTER → PHENOMENON")

    # Aggregate CB, LM, TA by register and model
    metrics = {
        'Cultural Boxing': 'cb_score',
        'Linguistic Mirroring': 'lm_present',
        'CME': 'cme_detected',
        'Topic Avoidance': 'ta_present',
        'PES': 'pes_present',
    }

    available_metrics = {k: v for k, v in metrics.items() if v in feature_df.columns}

    register_order = ['Formal', 'Informal', 'Moderate']
    model_order = ['ChatGPT', 'Claude', 'Gemini']

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 5))
    if len(available_metrics) == 1:
        axes = [axes]

    for i, (metric_name, metric_col) in enumerate(available_metrics.items()):
        ax = axes[i]

        pivot_data = feature_df.pivot_table(
            values=metric_col,
            index='register',
            columns='model',
            aggfunc='mean'
        )

        # Reorder
        pivot_data = pivot_data.reindex(register_order)
        model_cols = [m for m in model_order if any(m in c for c in pivot_data.columns)]
        pivot_cols = [c for c in pivot_data.columns if any(m in c for m in model_cols)]
        pivot_data = pivot_data[pivot_cols]
        pivot_data.columns = [c.split('(')[0].strip() for c in pivot_data.columns]

        # Plot
        pivot_data.plot(kind='bar', ax=ax,
                       color=[MODEL_COLORS_SHORT.get(c, '#999') for c in pivot_data.columns],
                       edgecolor='white', linewidth=1)

        ax.set_title(metric_name, fontweight='bold', fontsize=13)
        ax.set_xlabel('Register')
        ax.set_ylabel('Rate')
        ax.legend(title='Model', fontsize=8)
        ax.tick_params(axis='x', rotation=0)

    plt.suptitle('Phenomenon Expression by Register & Model', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'flow_register_phenomenon.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: flow_register_phenomenon.png")


# ============================================================
# 7. MODEL SIMILARITY MATRIX
# ============================================================
def create_model_similarity_matrix(feature_df: pd.DataFrame):
    """
    Create similarity matrix between models based on feature profiles.
    """
    print_section("MODEL SIMILARITY MATRIX")

    # Select numerical features
    exclude = ['model', 'user_name', 'domain', 'register', 'language',
               'response_preview', 'foods_mentioned', 'cultural_assumptions']
    num_cols = [c for c in feature_df.columns if c not in exclude
                and feature_df[c].dtype in ['int64', 'float64', 'bool']]

    # Aggregate by model
    model_profiles = feature_df.groupby('model')[num_cols].mean()
    model_profiles.index = [m.split('(')[0].strip() for m in model_profiles.index]

    # Compute cosine similarity
    sim_matrix = cosine_similarity(model_profiles)
    sim_df = pd.DataFrame(sim_matrix, index=model_profiles.index, columns=model_profiles.index)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(sim_df, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0, vmax=1, linewidths=1, linecolor='white',
                ax=ax, cbar_kws={'label': 'Cosine Similarity'})

    ax.set_title('Model Similarity Based on Feature Profiles', fontsize=15, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_similarity_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: model_similarity_matrix.png")

    # Print
    print_subsection("MODEL PAIRWISE SIMILARITY")
    for i, m1 in enumerate(sim_df.index):
        for m2 in sim_df.columns[i+1:]:
            print(f"  {m1} ↔ {m2}: {sim_df.loc[m1, m2]:.4f}")


# ============================================================
# 8. PHENOMENON CORRELATION MATRIX (PUBLICATION-READY)
# ============================================================
def create_phenomenon_correlation_heatmap(feature_df: pd.DataFrame):
    """
    Create publication-ready phenomenon correlation heatmap.
    """
    print_section("PHENOMENON CORRELATION MATRIX")

    phen_cols = {
        'Cultural\nBoxing': 'cb_score',
        'DEA': 'dea_combined_score',
        'PES': 'pes_present',
        'Linguistic\nMirroring': 'lm_present',
        'CME': 'cme_detected',
        'Topic\nAvoidance': 'ta_present',
        'Empathy': 'emotion_empathy_score',
        'Systemic\nCritique': 'auth_systemic_critique_present',
        'Validation': 'accept_validation_count',
        'Boundary': 'resist_boundary_setting_count',
        'Pushback': 'resist_pushback_count',
    }

    available = {k: v for k, v in phen_cols.items() if v in feature_df.columns}

    corr_data = feature_df[[v for v in available.values()]].corr()
    corr_data.index = list(available.keys())
    corr_data.columns = list(available.keys())

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(12, 10))

    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f',
                cmap=cmap, center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor='white',
                square=True, ax=ax,
                cbar_kws={'label': 'Pearson r', 'shrink': 0.8})

    ax.set_title('Phenomenon Correlation Matrix', fontsize=18, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phenomenon_correlation_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: phenomenon_correlation_matrix.png")

    # Print strongest correlations
    print_subsection("STRONGEST PHENOMENON CORRELATIONS")
    pairs = []
    for i, col1 in enumerate(corr_data.columns):
        for col2 in corr_data.columns[i+1:]:
            pairs.append((col1, col2, corr_data.loc[col1, col2]))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for col1, col2, corr in pairs[:10]:
        print(f"  {col1.replace(chr(10),' ')} ↔ {col2.replace(chr(10),' ')}: r={corr:.3f}")


# ============================================================
# 9. SUMMARY DASHBOARD
# ============================================================
def create_summary_dashboard(feature_df: pd.DataFrame, classified_df: pd.DataFrame):
    """
    Create a comprehensive summary dashboard figure.
    """
    print_section("SUMMARY DASHBOARD")

    fig = plt.figure(figsize=(24, 18))

    # Title
    fig.suptitle('Machine Psychology: 10 Phenomena Analysis Dashboard',
                 fontsize=22, fontweight='bold', y=0.98)

    # --- 1. CB/LM/CME/TA Rates (top-left) ---
    ax1 = fig.add_subplot(2, 3, 1)
    phen_rates = {
        'Cultural Boxing': float(feature_df['cb_score'].mean()),
        'Linguistic Mirroring': float(classified_df['linguistic_mirroring'].mean()),
        'CME': float(feature_df['cme_detected'].mean()),
        'Topic Avoidance': float(feature_df['ta_present'].mean()),
        'PES': float(feature_df['pes_present'].mean()),
    }

    bars = ax1.barh(list(phen_rates.keys()), list(phen_rates.values()),
                    color=['#FF6B6B', '#FFEAA7', '#BB8FCE', '#85C1E9', '#45B7D1'],
                    edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, phen_rates.values()):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontweight='bold', fontsize=11)
    ax1.set_title('Phenomenon Rates (Overall)', fontweight='bold', fontsize=14)
    ax1.set_xlim(0, max(phen_rates.values()) * 1.4)

    # --- 2. Model Comparison: CB, LM, CME (top-center) ---
    ax2 = fig.add_subplot(2, 3, 2)
    comp_data = feature_df.groupby('model').agg(
        CB=('cb_score', 'mean'),
        LM=('lm_present', 'mean'),
        CME=('cme_detected', 'mean'),
        TA=('ta_present', 'mean'),
    )
    comp_data.index = [m.split('(')[0].strip() for m in comp_data.index]
    comp_data.plot(kind='bar', ax=ax2, edgecolor='white', linewidth=1)
    ax2.set_title('Phenomena by Model', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Rate')
    ax2.legend(fontsize=9)
    ax2.tick_params(axis='x', rotation=0)

    # --- 3. Empathy by Persona × Model (top-right) ---
    ax3 = fig.add_subplot(2, 3, 3)
    empathy_data = feature_df.pivot_table(
        values='emotion_empathy_score', index='model', columns='user_name', aggfunc='mean'
    )
    empathy_data.index = [m.split('(')[0].strip() for m in empathy_data.index]
    empathy_data.columns = [c.split()[0] for c in empathy_data.columns]
    empathy_data.plot(kind='bar', ax=ax3,
                     color=[PERSONA_COLORS[n] for n in ['Jake Thompson', 'Tyrone Williams', 'Reza Moradi']],
                     edgecolor='white', linewidth=1)
    ax3.set_title('Empathy Score by Persona × Model', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Empathy Score')
    ax3.tick_params(axis='x', rotation=0)

    # --- 4. A/R Ratio (bottom-left) ---
    ax4 = fig.add_subplot(2, 3, 4)
    if 'accept_total_count' in feature_df.columns and 'resist_total_count' in feature_df.columns:
        ar_by_domain = feature_df.groupby('domain').apply(
            lambda x: x['accept_total_count'].mean() / (x['resist_total_count'].mean() + 1)
        ).sort_values()
        colors = ['#FF6B6B' if v < 0.5 else '#4ECDC4' if v > 1 else '#FFEAA7' for v in ar_by_domain.values]
        ar_by_domain.plot(kind='barh', ax=ax4, color=colors, edgecolor='white', linewidth=1)
        ax4.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='A/R = 1')
        ax4.set_title('Acceptance/Resistance Ratio by Domain', fontweight='bold', fontsize=14)
        ax4.set_xlabel('A/R Ratio')
        ax4.legend(fontsize=9)

    # --- 5. Language Distribution (bottom-center) ---
    ax5 = fig.add_subplot(2, 3, 5)
    lang_data = classified_df.groupby(['model', 'language']).size().unstack(fill_value=0)
    lang_data.index = [m.split('(')[0].strip() for m in lang_data.index]
    lang_data.plot(kind='bar', stacked=True, ax=ax5,
                  color=['#94A3B8', '#059669', '#7C3AED'],
                  edgecolor='white', linewidth=1)
    ax5.set_title('Language Distribution by Model', fontweight='bold', fontsize=14)
    ax5.set_ylabel('Count')
    ax5.legend(title='Language', fontsize=9)
    ax5.tick_params(axis='x', rotation=0)

    # --- 6. Key Findings Text (bottom-right) ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    findings_text = (
        "KEY FINDINGS\n\n"
        f"• Cultural Boxing: {feature_df['cb_score'].mean():.1%} of Food prompts\n"
        f"• Linguistic Mirroring: {classified_df['linguistic_mirroring'].mean():.1%} overall\n"
        f"  - Claude: 6 accurate (Persian for Reza)\n"
        f"  - Gemini: 8 errors (Persian for Tyrone)\n"
        f"• CME: Only in Gemini ({feature_df['cme_detected'].sum()} instances)\n"
        f"• Topic Avoidance: {feature_df['ta_present'].mean():.1%} overall\n"
        f"• PES: {feature_df['pes_present'].mean():.1%} in Airport domain\n"
        f"• IES: Reza > Tyrone > Jake\n"
        f"• Unmarkedness Paradox: CONFIRMED\n\n"
        f"ANOVA: {13} features differentiate models (p<0.05)\n"
        f"Chi-square: {5} phenomena associated with models\n"
        f"Best K-Means: K=6 (Silhouette={0.392:.3f})"
    )

    ax6.text(0.05, 0.95, findings_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / 'summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: summary_dashboard.png")


# ============================================================
# 10. MAIN
# ============================================================
def run_section11(section2_results=None, feature_df=None, classified_df=None):
    """Execute Section 11: Advanced Visualization."""
    print_section("SECTION 11: ADVANCED VISUALIZATION")
    print("  Sankey | Network Graph | Heatmaps | Flow Diagram | Dashboard")

    if feature_df is None or classified_df is None:
        feature_df, classified_df = load_section_data()

    # 1. Sankey Diagram
    create_sankey_diagram(feature_df, classified_df)

    # 2. Network Graph
    create_network_graph(feature_df)

    # 3. Full Heatmap
    create_full_heatmap(feature_df)

    # 4. Flow Diagram
    create_flow_diagram(feature_df, classified_df)

    # 5. Model Similarity Matrix
    create_model_similarity_matrix(feature_df)

    # 6. Phenomenon Correlation Matrix
    create_phenomenon_correlation_heatmap(feature_df)

    # 7. Summary Dashboard
    create_summary_dashboard(feature_df, classified_df)

    # Count figures
    figures = list(OUTPUT_DIR.glob('*.png')) + list(OUTPUT_DIR.glob('*.html'))
    print(f"\n{'='*70}")
    print(f"  ✅ SECTION 11 COMPLETE")
    print(f"  {len(figures)} visualizations generated in: {OUTPUT_DIR}")
    print(f"{'='*70}")

    return {'n_figures': len(figures)}


# ============================================================
# Execute
# ============================================================
if __name__ == "__main__":
    section11_results = run_section11()

"""
==============================================================
SECTION 10: CLUSTERING & VISUALIZATION
UMAP | t-SNE | K-Means | HDBSCAN | Hierarchical | PCA
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

# Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("⚠️  HDBSCAN not available — skipping HDBSCAN clustering")

# Dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP not available — using t-SNE only")

from sklearn.manifold import TSNE

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
import seaborn as sns

# ============================================================
# 0. Configuration & Style
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("whitegrid")

MODEL_COLORS = {
    'ChatGPT (GPT-5.4)': '#10A37F',
    'Claude 4.6 Sonnet': '#D97706',
    'Gemini 3.1 Pro': '#4285F4'
}

PERSONA_COLORS = {
    'Jake Thompson': '#94A3B8',
    'Tyrone Williams': '#7C3AED',
    'Reza Moradi': '#059669'
}

PERSONA_MARKERS = {
    'Jake Thompson': 's',
    'Tyrone Williams': 'D',
    'Reza Moradi': 'o'
}

DOMAIN_COLORS = {
    'Cultural Food': '#FF6B6B',
    'Childcare Trust': '#4ECDC4',
    'Lost Wallet Ethics': '#45B7D1',
    'Refugee Stereotyping': '#96CEB4',
    'Airport Profiling': '#FFEAA7'
}

OUTPUT_DIR = Path("output") / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Load Data
# ============================================================
def load_section_data():
    """Load data from previous sections."""
    feature_df = pd.read_csv(Path("output") / "engineered_features.csv", encoding='utf-8')

    # Try to load embeddings
    embeddings_path = Path("output") / "response_embeddings.npy"
    embeddings = None
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        print(f"  ✅ Loaded embeddings: {embeddings.shape}")

    # Try PCA data
    pca_path = Path("output") / "embeddings_pca50.csv"
    pca_df = None
    if pca_path.exists():
        pca_df = pd.read_csv(pca_path, encoding='utf-8')
        print(f"  ✅ Loaded PCA data: {len(pca_df)} rows")

    print(f"  ✅ Loaded features: {len(feature_df)} rows, {len(feature_df.columns)} cols")

    return feature_df, embeddings, pca_df


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
# 3. PREPARE DATA FOR CLUSTERING
# ============================================================
def prepare_clustering_data(feature_df: pd.DataFrame, embeddings: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare numerical data for clustering.
    Uses embeddings if available, otherwise feature matrix.
    """
    print_section("DATA PREPARATION FOR CLUSTERING")

    # Option A: Use embeddings
    if embeddings is not None:
        X = embeddings
        print(f"  Using embeddings: shape {X.shape}")
        return X, X, feature_df

    # Option B: Use numerical features
    exclude_cols = ['model', 'user_name', 'domain', 'register', 'language',
                    'response_preview', 'foods_mentioned', 'cultural_assumptions',
                    'full_response']

    feature_cols = [c for c in feature_df.columns
                    if c not in exclude_cols
                    and feature_df[c].dtype in ['int64', 'float64', 'bool']]

    X = feature_df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)

    X = X.fillna(0)
    X = X.loc[:, (X != 0).any(axis=0)]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Using features: {X.shape[1]} features from {X.shape[0]} samples")

    return X_scaled, X_scaled, feature_df


# ============================================================
# 4. K-MEANS CLUSTERING
# ============================================================
def run_kmeans(X: np.ndarray, feature_df: pd.DataFrame, n_clusters_range: List[int] = None) -> Dict:
    """Run K-Means clustering with multiple k values."""
    print_section("K-MEANS CLUSTERING")

    if n_clusters_range is None:
        n_clusters_range = [2, 3, 4, 5, 6]

    results = {
        'k_values': [],
        'inertia': [],
        'silhouette_scores': [],
        'calinski_harabasz': [],
        'best_k': None,
        'cluster_labels': None
    }

    print_subsection("ELBOW METHOD & SILHOUETTE SCORES")
    print(f"  {'K':<6} {'Inertia':<12} {'Silhouette':<12} {'Calinski-Harabasz':<18}")
    print(f"  {'─'*50}")

    best_score = -1
    best_k = 2

    for k in n_clusters_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        inertia = kmeans.inertia_
        sil = silhouette_score(X, labels) if k > 1 else 0
        ch = calinski_harabasz_score(X, labels) if k > 1 else 0

        results['k_values'].append(k)
        results['inertia'].append(float(inertia))
        results['silhouette_scores'].append(float(sil))
        results['calinski_harabasz'].append(float(ch))

        print(f"  {k:<6} {inertia:<12.1f} {sil:<12.4f} {ch:<18.1f}")

        if sil > best_score:
            best_score = sil
            best_k = k
            results['best_k'] = k
            results['cluster_labels'] = labels

    print(f"\n  🏆 Best K = {best_k} (Silhouette = {best_score:.4f})")

    # Analyze clusters
    if results['cluster_labels'] is not None:
        feature_df_temp = feature_df.copy()
        feature_df_temp['cluster'] = results['cluster_labels']

        print_subsection(f"CLUSTER COMPOSITION (K={best_k})")

        # Model distribution per cluster
        for c in range(best_k):
            cluster_data = feature_df_temp[feature_df_temp['cluster'] == c]
            model_counts = cluster_data['model'].value_counts()
            persona_counts = cluster_data['user_name'].value_counts()
            domain_counts = cluster_data['domain'].value_counts()

            print(f"\n  Cluster {c} (n={len(cluster_data)}):")
            print(f"     Models: {dict(model_counts)}")
            print(f"     Personas: {dict(persona_counts)}")
            print(f"     Top Domain: {domain_counts.index[0]} ({domain_counts.values[0]})")

    return results


# ============================================================
# 5. HIERARCHICAL CLUSTERING
# ============================================================
def run_hierarchical(X: np.ndarray, feature_df: pd.DataFrame) -> Dict:
    """Run Agglomerative Hierarchical Clustering."""
    print_section("HIERARCHICAL CLUSTERING")

    results = {}

    # Try different linkage methods
    linkages = ['ward', 'complete', 'average']

    print_subsection("CLUSTER QUALITY BY LINKAGE METHOD")
    print(f"  {'Linkage':<12} {'Silhouette':<12} {'Calinski-Harabasz':<18}")
    print(f"  {'─'*45}")

    for linkage in linkages:
        try:
            agg = AgglomerativeClustering(n_clusters=3, linkage=linkage)
            labels = agg.fit_predict(X)

            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)

            results[linkage] = {
                'silhouette': float(sil),
                'calinski_harabasz': float(ch),
                'labels': labels.tolist()
            }

            print(f"  {linkage:<12} {sil:<12.4f} {ch:<18.1f}")
        except Exception as e:
            print(f"  {linkage:<12} failed: {e}")

    # Best linkage
    best_linkage = max(results, key=lambda l: results[l]['silhouette'])
    print(f"\n  🏆 Best Linkage: {best_linkage}")

    # Analyze best clusters
    feature_df_temp = feature_df.copy()
    feature_df_temp['hierarchical_cluster'] = results[best_linkage]['labels']

    print_subsection(f"HIERARCHICAL CLUSTER COMPOSITION ({best_linkage})")
    for c in sorted(feature_df_temp['hierarchical_cluster'].unique()):
        cluster_data = feature_df_temp[feature_df_temp['hierarchical_cluster'] == c]
        model_counts = cluster_data['model'].value_counts()
        print(f"  Cluster {c} (n={len(cluster_data)}): {dict(model_counts)}")

    return results


# ============================================================
# 6. HDBSCAN CLUSTERING
# ============================================================
def run_hdbscan_clustering(X: np.ndarray, feature_df: pd.DataFrame) -> Dict:
    """Run HDBSCAN density-based clustering."""
    print_section("HDBSCAN CLUSTERING")

    if not HDBSCAN_AVAILABLE:
        print("  ⚠️  HDBSCAN not installed. Skipping.")
        return {'status': 'skipped'}

    results = {}

    # Try different min_cluster_sizes
    min_sizes = [3, 5, 8, 10]

    print_subsection("HDBSCAN WITH VARYING min_cluster_size")
    print(f"  {'Min Size':<10} {'Clusters':<10} {'Noise':<10} {'Silhouette':<12}")
    print(f"  {'─'*45}")

    for min_size in min_sizes:
        try:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=2, metric='euclidean')
            labels = clusterer.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            # Silhouette (excluding noise)
            mask = labels != -1
            if mask.sum() > 2 and n_clusters > 1:
                sil = silhouette_score(X[mask], labels[mask])
            else:
                sil = 0

            results[min_size] = {
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': float(sil),
                'labels': labels.tolist()
            }

            print(f"  {min_size:<10} {n_clusters:<10} {n_noise:<10} {sil:<12.4f}")
        except Exception as e:
            print(f"  {min_size:<10} failed: {e}")

    # Best result
    if results:
        best = max(results, key=lambda k: results[k]['silhouette'] if isinstance(results[k], dict) else 0)
        print(f"\n  🏆 Best min_cluster_size: {best}")
        print(f"     Clusters: {results[best]['n_clusters']}, Noise: {results[best]['n_noise']}")

    return results


# ============================================================
# 7. DIMENSIONALITY REDUCTION & VISUALIZATION
# ============================================================
def run_dimensionality_reduction(X: np.ndarray, feature_df: pd.DataFrame, method: str = 'both') -> Dict:
    """
    Reduce dimensions using UMAP and/or t-SNE.
    Returns coordinates for visualization.
    """
    print_section(f"DIMENSIONALITY REDUCTION: {method.upper()}")

    results = {}
    n_samples = X.shape[0]

    # Adjust perplexity for t-SNE
    perplexity = min(30, max(5, n_samples // 4))

    # --- t-SNE ---
    if method in ['tsne', 'both']:
        print(f"\n  🔧 Running t-SNE (perplexity={perplexity})...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        X_tsne = tsne.fit_transform(X)
        results['tsne'] = X_tsne
        print(f"  ✅ t-SNE complete: shape {X_tsne.shape}")

    # --- UMAP ---
    if method in ['umap', 'both'] and UMAP_AVAILABLE:
        print(f"\n  🔧 Running UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, n_samples//3))
        X_umap = reducer.fit_transform(X)
        results['umap'] = X_umap
        print(f"  ✅ UMAP complete: shape {X_umap.shape}")
    elif method in ['umap', 'both']:
        print(f"  ⚠️  UMAP not available")

    # --- PCA ---
    print(f"\n  🔧 Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    results['pca'] = X_pca
    print(f"  ✅ PCA complete: {pca.explained_variance_ratio_.sum():.3f} variance explained")

    # Save all coordinates
    coords_df = feature_df[['model', 'user_name', 'register', 'domain']].copy()

    if 'tsne' in results:
        coords_df['tsne_x'] = results['tsne'][:, 0]
        coords_df['tsne_y'] = results['tsne'][:, 1]
    if 'umap' in results:
        coords_df['umap_x'] = results['umap'][:, 0]
        coords_df['umap_y'] = results['umap'][:, 1]
    if 'pca' in results:
        coords_df['pca_x'] = results['pca'][:, 0]
        coords_df['pca_y'] = results['pca'][:, 1]

    coords_df.to_csv(OUTPUT_DIR / "dimension_reduction_coords.csv", index=False)
    print(f"  💾 Saved coordinates: {OUTPUT_DIR / 'dimension_reduction_coords.csv'}")

    return results, coords_df


# ============================================================
# 8. SCATTER PLOTS
# ============================================================
def create_scatter_plots(coords_df: pd.DataFrame, method: str):
    """
    Create scatter plots colored by model, persona, and domain.
    """
    x_col = f'{method}_x'
    y_col = f'{method}_y'

    if x_col not in coords_df.columns or y_col not in coords_df.columns:
        print(f"  ⚠️  {method} coordinates not found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # 1. Colored by MODEL
    ax = axes[0]
    for model in coords_df['model'].unique():
        mask = coords_df['model'] == model
        ax.scatter(coords_df.loc[mask, x_col], coords_df.loc[mask, y_col],
                   c=MODEL_COLORS.get(model, '#999999'), label=model.split('(')[0].strip(),
                   alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax.set_title(f'{method.upper()} — Colored by Model', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')

    # 2. Colored by PERSONA
    ax = axes[1]
    for persona in ['Jake Thompson', 'Tyrone Williams', 'Reza Moradi']:
        mask = coords_df['user_name'] == persona
        ax.scatter(coords_df.loc[mask, x_col], coords_df.loc[mask, y_col],
                   c=PERSONA_COLORS[persona], label=persona.split()[0],
                   alpha=0.7, s=50, edgecolors='white', linewidth=0.5,
                   marker=PERSONA_MARKERS[persona])
    ax.set_title(f'{method.upper()} — Colored by Persona', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # 3. Colored by DOMAIN
    ax = axes[2]
    for domain in coords_df['domain'].unique():
        mask = coords_df['domain'] == domain
        ax.scatter(coords_df.loc[mask, x_col], coords_df.loc[mask, y_col],
                   c=DOMAIN_COLORS.get(domain, '#999999'), label=domain,
                   alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax.set_title(f'{method.upper()} — Colored by Domain', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'scatter_{method}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved scatter plot: scatter_{method}.png")


# ============================================================
# 9. ELBOW & SILHOUETTE PLOTS
# ============================================================
def create_clustering_plots(kmeans_results: Dict):
    """Create elbow plot and silhouette plot."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    k_values = kmeans_results['k_values']

    # Elbow plot
    ax = axes[0]
    ax.plot(k_values, kmeans_results['inertia'], 'o-', color='#4285F4', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Inertia', fontsize=12)
    ax.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Silhouette plot
    ax = axes[1]
    ax.plot(k_values, kmeans_results['silhouette_scores'], 'o-', color='#10A37F', linewidth=2, markersize=8)
    ax.axhline(y=max(kmeans_results['silhouette_scores']), color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Calinski-Harabasz
    ax = axes[2]
    ax.plot(k_values, kmeans_results['calinski_harabasz'], 'o-', color='#D97706', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Calinski-Harabasz Index', fontsize=12)
    ax.set_title('Calinski-Harabasz Index', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'clustering_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: clustering_metrics.png")


# ============================================================
# 10. HEATMAP: PHENOMENON × MODEL
# ============================================================
def create_phenomenon_heatmap(feature_df: pd.DataFrame):
    """Create heatmap of phenomenon intensity by model and persona."""
    print_section("PHENOMENON HEATMAP")

    phen_features = {
        'Cultural Boxing': 'cb_score',
        'DEA': 'dea_combined_score',
        'PES': 'pes_present',
        'Linguistic Mirroring': 'lm_present',
        'CME': 'cme_detected',
        'Topic Avoidance': 'ta_present',
        'Empathy': 'emotion_empathy_score',
        'Systemic Critique': 'auth_systemic_critique_present',
        'Validation': 'accept_validation_count',
        'Boundary Setting': 'resist_boundary_setting_count',
    }

    available = {k: v for k, v in phen_features.items() if v in feature_df.columns}

    # Aggregate by model
    model_data = {}
    for model in feature_df['model'].unique():
        model_df = feature_df[feature_df['model'] == model]
        model_data[model.split('(')[0].strip()] = {
            phen: float(model_df[feat].mean()) for phen, feat in available.items()
        }

    heatmap_df = pd.DataFrame(model_data).T

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': 'Intensity'},
                linewidths=0.5, linecolor='white')

    ax.set_title('Phenomenon Intensity by Model', fontsize=16, fontweight='bold')
    ax.set_xlabel('Phenomenon', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phenomenon_heatmap_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: phenomenon_heatmap_model.png")

    # Also by persona
    persona_data = {}
    for persona in feature_df['user_name'].unique():
        persona_df = feature_df[feature_df['user_name'] == persona]
        persona_data[persona] = {
            phen: float(persona_df[feat].mean()) for phen, feat in available.items()
        }

    persona_heatmap = pd.DataFrame(persona_data).T

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(persona_heatmap, annot=True, fmt='.2f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': 'Intensity'},
                linewidths=0.5, linecolor='white')
    ax.set_title('Phenomenon Intensity by Persona', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phenomenon_heatmap_persona.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: phenomenon_heatmap_persona.png")


# ============================================================
# 11. RADAR CHART
# ============================================================
def create_radar_chart(feature_df: pd.DataFrame):
    """Create radar chart comparing model signatures."""
    print_section("RADAR CHART")

    metrics = [
        'cb_score', 'dea_combined_score', 'pes_present',
        'lm_present', 'cme_detected', 'ta_present',
        'emotion_empathy_score', 'auth_systemic_critique_present'
    ]

    available = [m for m in metrics if m in feature_df.columns]

    # Compute means by model
    model_means = {}
    for model in feature_df['model'].unique():
        model_df = feature_df[feature_df['model'] == model]
        model_means[model.split('(')[0].strip()] = [float(model_df[m].mean()) for m in available]

    # Radar chart
    n_metrics = len(available)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = ['#10A37F', '#D97706', '#4285F4']

    for i, (model, values) in enumerate(model_means.items()):
        values_plot = values + values[:1]
        ax.fill(angles, values_plot, alpha=0.1, color=colors[i])
        ax.plot(angles, values_plot, 'o-', linewidth=2, label=model, color=colors[i], markersize=8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title()[:20] for m in available], fontsize=10)
    ax.set_ylim(0, max(max(v) for v in model_means.values()) * 1.2)
    ax.set_title('Model Behavioral Signatures', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'radar_chart_signatures.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: radar_chart_signatures.png")


# ============================================================
# 12. BAR CHARTS
# ============================================================
def create_bar_charts(feature_df: pd.DataFrame):
    """Create comparison bar charts."""
    print_section("COMPARISON BAR CHARTS")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. CB by Model × Persona
    ax = axes[0, 0]
    cb_data = feature_df.groupby(['model', 'user_name'])['cb_score'].mean().unstack()
    cb_data.index = [m.split('(')[0].strip() for m in cb_data.index]
    cb_data.plot(kind='bar', ax=ax, color=[PERSONA_COLORS[n] for n in cb_data.columns])
    ax.set_title('Cultural Boxing Rate', fontweight='bold')
    ax.set_ylabel('Rate')
    ax.legend(title='Persona', fontsize=8)
    ax.tick_params(axis='x', rotation=0)

    # 2. LM by Model × Persona
    ax = axes[0, 1]
    lm_data = feature_df.groupby(['model', 'user_name'])['lm_present'].mean().unstack()
    lm_data.index = [m.split('(')[0].strip() for m in lm_data.index]
    lm_data.plot(kind='bar', ax=ax, color=[PERSONA_COLORS[n] for n in lm_data.columns])
    ax.set_title('Linguistic Mirroring Rate', fontweight='bold')
    ax.set_ylabel('Rate')
    ax.tick_params(axis='x', rotation=0)

    # 3. Empathy by Model × Persona
    ax = axes[0, 2]
    emp_data = feature_df.groupby(['model', 'user_name'])['emotion_empathy_score'].mean().unstack()
    emp_data.index = [m.split('(')[0].strip() for m in emp_data.index]
    emp_data.plot(kind='bar', ax=ax, color=[PERSONA_COLORS[n] for n in emp_data.columns])
    ax.set_title('Empathy Score', fontweight='bold')
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=0)

    # 4. Response Length by Model
    ax = axes[1, 0]
    len_data = feature_df.groupby('model')['struct_word_count'].mean()
    len_data.index = [m.split('(')[0].strip() for m in len_data.index]
    len_data.plot(kind='barh', ax=ax, color=[MODEL_COLORS.get(m, '#999') for m in feature_df['model'].unique()])
    ax.set_title('Avg Response Length (words)', fontweight='bold')
    ax.set_xlabel('Words')

    # 5. A/R Ratio by Model
    ax = axes[1, 1]
    if 'accept_total_count' in feature_df.columns and 'resist_total_count' in feature_df.columns:
        ar_data = feature_df.groupby('model').apply(
            lambda x: x['accept_total_count'].mean() / (x['resist_total_count'].mean() + 1)
        )
        ar_data.index = [m.split('(')[0].strip() for m in ar_data.index]
        ar_data.plot(kind='bar', ax=ax, color=[MODEL_COLORS.get(m, '#999') for m in feature_df['model'].unique()])
        ax.set_title('Acceptance/Resistance Ratio', fontweight='bold')
        ax.set_ylabel('A/R Ratio')
        ax.tick_params(axis='x', rotation=0)

    # 6. Systemic Critique by Model × Persona
    ax = axes[1, 2]
    sc_data = feature_df.groupby(['model', 'user_name'])['auth_systemic_critique_present'].mean().unstack()
    sc_data.index = [m.split('(')[0].strip() for m in sc_data.index]
    sc_data.plot(kind='bar', ax=ax, color=[PERSONA_COLORS[n] for n in sc_data.columns])
    ax.set_title('Systemic Critique Rate', fontweight='bold')
    ax.set_ylabel('Rate')
    ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_bar_charts.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: comparison_bar_charts.png")


# ============================================================
# 13. PHENOMENON DISTRIBUTION
# ============================================================
def create_phenomenon_distribution(feature_df: pd.DataFrame):
    """Create phenomenon distribution plot across models."""
    print_section("PHENOMENON DISTRIBUTION")

    phen_cols = {
        'CB': 'cb_score',
        'DEA': 'dea_combined_score',
        'PES': 'pes_present',
        'LM': 'lm_present',
        'CME': 'cme_detected',
        'TA': 'ta_present'
    }

    available = {k: v for k, v in phen_cols.items() if v in feature_df.columns}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (phen_name, phen_col) in enumerate(available.items()):
        ax = axes[i]
        model_rates = feature_df.groupby('model')[phen_col].mean()
        model_rates.index = [m.split('(')[0].strip() for m in model_rates.index]

        bars = ax.bar(model_rates.index, model_rates.values,
                      color=[MODEL_COLORS.get(m, '#999') for m in feature_df['model'].unique()],
                      edgecolor='white', linewidth=1.5)

        for bar, val in zip(bars, model_rates.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.1%}', ha='center', fontweight='bold', fontsize=11)

        ax.set_title(f'{phen_name}', fontweight='bold', fontsize=13)
        ax.set_ylim(0, max(model_rates.values) * 1.3)
        ax.tick_params(axis='x', rotation=15)

    # Hide extra subplot
    if len(available) < 6:
        for j in range(len(available), 6):
            axes[j].set_visible(False)

    plt.suptitle('Phenomenon Distribution Across Models', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phenomenon_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: phenomenon_distribution.png")


# ============================================================
# 14. MAIN
# ============================================================
def run_section10(section2_results=None, feature_df=None, embeddings=None, pca_df=None):
    """Execute Section 10: Clustering & Visualization."""
    print_section("SECTION 10: CLUSTERING & VISUALIZATION")
    print("  UMAP | t-SNE | K-Means | HDBSCAN | Hierarchical | PCA")

    if feature_df is None:
        feature_df, embeddings, pca_df = load_section_data()

    # Prepare data
    X, X_cluster, feature_df = prepare_clustering_data(feature_df, embeddings)

    # K-Means
    kmeans_results = run_kmeans(X_cluster, feature_df)

    # Hierarchical
    hierarchical_results = run_hierarchical(X_cluster, feature_df)

    # HDBSCAN
    hdbscan_results = run_hdbscan_clustering(X_cluster, feature_df)

    # Dimensionality reduction
    dim_results, coords_df = run_dimensionality_reduction(X, feature_df, method='both')

    # --- Create Visualizations ---
    print_section("GENERATING VISUALIZATIONS")

    # Scatter plots
    for method in ['pca', 'tsne', 'umap']:
        if method in dim_results:
            create_scatter_plots(coords_df, method)

    # Clustering metrics plot
    create_clustering_plots(kmeans_results)

    # Heatmap
    create_phenomenon_heatmap(feature_df)

    # Radar chart
    create_radar_chart(feature_df)

    # Bar charts
    create_bar_charts(feature_df)

    # Phenomenon distribution
    create_phenomenon_distribution(feature_df)

    # Summary
    print_section("CLUSTERING SUMMARY")
    print(f"\n  Best K (K-Means): {kmeans_results['best_k']}")
    print(f"  Best Silhouette: {max(kmeans_results['silhouette_scores']):.4f}")
    print(f"  HDBSCAN: {'Available' if HDBSCAN_AVAILABLE else 'Not available'}")
    print(f"  UMAP: {'Available' if UMAP_AVAILABLE else 'Not available'}")

    n_figures = len(list(OUTPUT_DIR.glob('*.png')))
    print(f"\n  📊 {n_figures} figures saved to: {OUTPUT_DIR}")

    all_results = {
        'kmeans': {k: v for k, v in kmeans_results.items() if k != 'cluster_labels'},
        'hierarchical': {k: {k2: v2 for k2, v2 in v.items() if k2 != 'labels'}
                         for k, v in hierarchical_results.items()},
        'hdbscan': hdbscan_results,
        'dim_reduction_methods': list(dim_results.keys()),
        'n_figures': n_figures
    }

    # Save results
    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_safe(i) for i in obj]
        elif isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, (np.bool_,)): return bool(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        else: return obj

    with open(OUTPUT_DIR / "clustering_results.json", 'w') as f:
        json.dump(make_json_safe(all_results), f, indent=2)

    print(f"\n{'='*70}")
    print(f"  ✅ SECTION 10 COMPLETE")
    print(f"  {n_figures} visualizations generated")
    print(f"{'='*70}")

    return all_results, coords_df


# ============================================================
# Execute
# ============================================================
if __name__ == "__main__":
    section10_results, coords_df = run_section10()

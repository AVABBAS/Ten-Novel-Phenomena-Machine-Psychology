# 🗂️ Section 10: Clustering & Visualization

## 🎯 Purpose
Unsupervised learning and visual exploration of behavioral patterns.

## ⚙️ What It Does

### 🗂️ Clustering Methods
| Method | Configuration | Best Result |
|--------|--------------|-------------|
| 🔵 **K-Means** | K=2–6 | K=6, Silhouette=0.392 |
| 🌳 **Hierarchical** | Ward, Complete, Average | 3 clusters (Complete) |
| 🟣 **HDBSCAN** | min_cluster_size=3–10 | 6 clusters, 0 noise |

### 📉 Dimensionality Reduction
- 🟠 **t-SNE**: 2D (perplexity=30)
- 🟢 **UMAP**: 2D (n_neighbors=15)
- 🔵 **PCA**: 2D (33.2% variance)

### 🎨 Visualizations (9 figures)
1. 📊 `scatter_pca.png` — PCA by model, persona, domain
2. 📊 `scatter_tsne.png` — t-SNE by model, persona, domain
3. 📊 `scatter_umap.png` — UMAP by model, persona, domain
4. 📈 `clustering_metrics.png` — Elbow + Silhouette + CH plots
5. 🔥 `phenomenon_heatmap_model.png` — 10 phenomena × 3 models
6. 🔥 `phenomenon_heatmap_persona.png` — 10 phenomena × 3 personas
7. 🎯 `radar_chart_signatures.png` — Model signature comparison
8. 📊 `comparison_bar_charts.png` — 6-panel analysis
9. 📊 `phenomenon_distribution.png` — Phenomenon rates across models

## 📥 Input
- 💾 `output/engineered_features.csv`
- 💾 `output/response_embeddings.npy`

## 📤 Output
- 🖼️ 9 PNG figures in `output/figures/`
- 💾 `output/figures/dimension_reduction_coords.csv`
- 💾 `output/figures/clustering_results.json`

## 🔬 Key Finding
🧩 A **sixth cluster** (14 responses) — exclusively Gemini (8) + Claude (6) — emerged from unsupervised analysis. This **CME/LM cluster** independently validates CME as a genuine behavioral signature, discovered without any human guidance.

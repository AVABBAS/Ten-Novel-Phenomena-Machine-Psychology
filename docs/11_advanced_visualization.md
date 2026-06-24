# 🎨 Section 11: Advanced Visualization

## 🎯 Purpose
Generate publication-ready visualizations for the manuscript.

## ⚙️ What It Does

### 🖼️ Visualizations (7 figures)

1. 🌊 **Sankey Diagram** (`sankey_phenomenon_flow.html`)
   - Interactive: Model → Phenomenon flow
   - Shows CB, LM, CME, TA transitions

2. 🕸️ **Network Graph** (`network_phenomenon_cooccurrence.png`)
   - 11-node correlation network
   - 🔵 Positive / 🔴 Negative edges

3. 🔥 **Feature Heatmap** (`heatmap_feature_profile.png`)
   - 25 features × 9 model-persona combinations
   - Min-Max normalized

4. 🔀 **Flow Diagram** (`flow_register_phenomenon.png`)
   - Register → Model → Phenomenon expression
   - Shows formality modulation effects

5. 📐 **Model Similarity Matrix** (`model_similarity_matrix.png`)
   - Cosine similarity between model profiles
   - ChatGPT↔Claude: 0.713 · Claude↔Gemini: 0.9996

6. 🔗 **Phenomenon Correlation** (`phenomenon_correlation_matrix.png`)
   - Publication-ready heatmap
   - 11 phenomena, lower triangle annotated

7. 📊 **Summary Dashboard** (`summary_dashboard.png`)
   - 6-panel overview:
     - 📊 Phenomenon rates
     - 📊 Model comparison
     - 💛 Empathy by persona
     - ⚔️ A/R ratio by domain
     - 🌐 Language distribution
     - 📝 Key findings text

## 📥 Input
- 💾 `output/classified_responses.csv`
- 💾 `output/engineered_features.csv`

## 📤 Output
- 🖼️ 7 figures in `output/figures/`
- 📊 16+ total figures across Sections 10–11

## 💡 Notes
All figures: 150 DPI (web) / 200+ DPI (publication). Interactive HTML files require a browser.

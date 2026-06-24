# 🧮 Section 5: Semantic Embedding

## 🎯 Purpose
Generate vector representations of all responses for similarity analysis.

## ⚙️ What It Does
- 🧠 Loads the `all-MiniLM-L6-v2` Sentence-Transformer model (384-dimensional embeddings)
- 🔢 Generates normalized embeddings for all 135 response texts
- 📐 Computes pairwise cosine similarity matrix (135 × 135)
- 👤 Analyzes cross-name similarity: Jake vs Tyrone vs Reza
- 🤖 Analyzes cross-model similarity: ChatGPT vs Claude vs Gemini
- 🎯 Computes reference similarity to 10 conceptual anchors:
  - 💛 empathy · 💼 professional · 🎯 direct · ⚖️ systemic · 🛡️ protective
  - 📋 procedural · 🌍 cultural · 🚧 boundary · ⚖️ legalistic · ☀️ warm
- 📉 Performs PCA reduction to 50 dimensions (94.4% variance explained)
- 📏 Calculates distance metrics between groups

## 📥 Input
- 📦 `section2_results` from Section 2 (for full response texts)

## 📤 Output
- 💾 `output/response_embeddings.npy` (135 × 384 numpy array)
- 💾 `output/embeddings_pca50.csv` (PCA-reduced for visualization)
- 💾 `output/embedding_metadata.json`

## 🔬 Key Findings
- 🎯 Domain is the strongest clustering factor (intra-domain similarity: 0.530–0.688)
- 🔴 Gemini shows maximum cross-name distance for Tyrone (0.868) — confirming CME
- 🏆 Claude leads in empathy (0.157) and systemic critique (0.068) reference similarity
- 🔒 ChatGPT is the most internally consistent model (intra-model similarity: 0.298)
- 🌈 Gemini is the most diverse model (intra-model similarity: 0.219)

## 💡 Notes
Embedding model is cached after first load. Normalized embeddings enable direct cosine similarity via dot product.

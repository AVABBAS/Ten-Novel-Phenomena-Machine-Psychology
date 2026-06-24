# 🧬 Section 7: Alignment Signature Fingerprinting

## 🎯 Purpose
Characterize each model's unique behavioral signature reflecting organizational safety philosophies.

## ⚙️ What It Does

### 🫂 Global Empath (ChatGPT)
- 💛 Emotional warmth: empathy, positive emotion, validation
- 📏 Verbosity: response length, paragraph count, formatting
- 🛡️ Protection: PES rate (22.2%), anti-erasure language
- 📈 IES: empathy allocation across names
- ⚪ LM: completely absent

### 📋 Pragmatic Consultant (Claude)
- 🎯 Directness: concision rate, directive/hedge ratio
- ⚖️ Systemic framing: highest critique rate (8.9%)
- 🪞 LM accuracy: 6 Persian responses for Reza, 0 errors ✅
- 📊 PES oscillation: varies by register for Tyrone
- 📦 CB: lowest rate (2.2%)

### 👔 Corporate Consultant (Gemini)
- 📋 Corporate formality: structured, professional language
- 🚫 TA: replaces PES in Airport domain
- ❌ CME: 8 Persian errors for Tyrone Williams
- 🪞 LM: highest rate (17.8%) but all errors
- 💛 Empathy: lowest score, procedural style

### 📊 Cross-Model Comparison
- 15 behavioral metrics side-by-side
- 📈 Statistical differentiation: 3/12 features significant
- 📉 PCA data for 2D visualization
- 🎯 Radar chart data for signature comparison

## 📥 Input
- 💾 `output/classified_responses.csv`
- 💾 `output/engineered_features.csv`

## 📤 Output
- 💾 `output/alignment_signatures.json`
- 💾 `output/signature_comparison.csv`
- 💾 `output/pca_model_fingerprints.csv`

## 💡 Key Insight
Alignment does **not** converge models toward a single notion of safe behavior. Each organization's safety philosophy produces a distinct, stable behavioral phenotype.

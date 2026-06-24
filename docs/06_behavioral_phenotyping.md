# 🔬 Section 6: Identity-Reactive Behavioral Phenotyping

## 🎯 Purpose
Quantify and analyze the 10 Machine Psychology phenomena across all models and personas.

## ⚙️ What It Does

### 📦 Cultural Boxing (CB)
- Cross-tabulation by model × persona
- Overall: 40.7% in Food domain (11/27)
- 👤 Reza Moradi: 55.6% (highest)

### 💛 Default Empathetic Amplification (DEA)
- Empathy comparison in Lost Wallet domain
- 📊 Kruskal-Wallis: H=0.762, p=0.683

### 🛡️ Proactive Empathetic Shield (PES)
- Airport domain analysis
- 🏆 ChatGPT: 22.2% (strongest)
- ❌ Gemini: 0% (replaced by TA)

### 📈 Inverted Empathy Spectrum (IES)
- Empathy ordering: Reza > Tyrone > Jake
- ✅ Claude & Gemini confirmed
- ❌ ChatGPT: Tyrone > Reza > Jake

### 🪞 Linguistic Mirroring (LM)
- 🟢 Claude: 6 LM (100% accurate — Persian for Reza)
- 🔴 Gemini: 8 LM (100% error — Persian for Tyrone = CME)
- ⚪ ChatGPT: 0 LM

### 🚫 Topic Avoidance (TA)
- Airport domain profiling acknowledgment
- ⚠️ Automated detection: Gemini=0% (manual coding: 100%)

### ❌ Cultural Misattribution Error (CME)
- 🔴 All 8 cases: Gemini → Tyrone Williams
- Distributed across 4 domains

### 🎭 Unmarkedness Paradox (UP)
- 🎯 Reza: strongly marked
- ⬜ Jake & Tyrone: effectively unmarked

### 🔗 Cross-Phenomenon Correlations
- 🔥 LM↔CME: r=0.738 (strongest)
- 🛡️ PES↔Systemic Critique: r=0.419
- 💛 DEA↔Systemic Acknowledgment: r=0.262

## 📥 Input
- 💾 `output/classified_responses.csv`
- 💾 `output/engineered_features.csv`

## 📤 Output
- 💾 `output/model_phenotypes.csv`
- 💾 `output/phenotype_analysis.json`

## 🤖 Model Phenotypes
- 🫂 **ChatGPT**: Global Empath — warm, verbose, strong PES
- 📋 **Claude**: Pragmatic Consultant — concise, systemic, accurate LM
- 👔 **Gemini**: Corporate Consultant — formal, procedural, CME errors

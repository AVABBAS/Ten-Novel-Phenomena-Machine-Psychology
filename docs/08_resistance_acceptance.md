# ⚔️ Section 8: Resistance & Acceptance Analysis

## 🎯 Purpose
Deep-dive into how models accept or resist user premises using enhanced pattern detection.

## ⚙️ What It Does

### 🛡️ 7 Enhanced Resistance Strategies (70 regex patterns)
1. 🚧 **Boundary Setting** — Clear limits on discussion topics
2. 💢 **Pushback** — Challenging or reframing user's premise
3. ↩️ **Deflection** — Redirecting conversation away from sensitive topics
4. 🟢 **Normalization** — Framing situations as normal/routine
5. 🔽 **Minimization** — Downplaying severity of concerns
6. 📋 **Proceduralization** — Converting emotional concerns into logistical matters
7. ⚖️ **Systemic Acknowledgment** — Recognizing structural issues

### 🤝 6 Enhanced Acceptance Types (60 regex patterns)
1. 💛 **Emotional Validation** — Validating user's emotional experience
2. 🛠️ **Practical Support** — Offering concrete, actionable help
3. 💪 **Empowerment** — Building user confidence and agency
4. 🌍 **Cultural Affirmation** — Validating cultural identity and heritage
5. 🟢 **Contextual Normalization** — Normalizing to reduce shame/stigma
6. ✅ **Explicit Agreement** — Direct agreement with user's position

### 📊 Analysis
- A/R ratios by model, persona, and domain
- Model × Persona cross-tabulation
- Domain-level resistance patterns
- Enhanced TA and PES redetection

## 📥 Input
- 💾 `output/classified_responses.csv`
- 💾 `output/engineered_features.csv`

## 📤 Output
- 💾 `output/engineered_features_enhanced.csv`
- 💾 `output/resistance_acceptance_analysis.json`

## 🔬 Key Findings
- 🫂 ChatGPT: A/R=1.06 (acceptance-oriented)
- 📋 Claude: A/R=0.59 (resistance-oriented)
- 🔴 Claude×Reza: A/R=0.30 (maximum resistance to Middle Eastern name)
- 🔴 Gemini×Tyrone: A/R=0.33 (CME-driven resistance)
- 🏚️ Refugee domain: A/R=0.36 (highest resistance overall)

## 💡 Notes
**130 new regex patterns** deployed. All patterns operate on full response texts.

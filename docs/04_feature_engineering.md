# 🔧 Section 4: Feature Engineering

## 🎯 Purpose
Extract 136 structured behavioral features from response texts.

## ⚙️ What It Does
Builds on 6 base dictionary categories and adds article-specific phenomenon detectors:

### 👑 1. Authority & Power Language (17 features)
Directives, hedging, professional authority, personal authority, systemic critique markers

### 🤝 2. Acceptance & Resistance Strategies (10 features)
Validation, affirmation, emotional support, boundary setting, pushback, deflection

### 🧠 3. Reasoning & Logic Patterns (9 features)
Logical structures, evidence citation, cost-benefit analysis, moral reasoning, practical reasoning

### 👥 4. Social Roles & Identity Markers (11 features)
Family roles, professional roles, cultural identity, power dynamics

### ❤️ 5. Emotion & Affect (9 features)
Positive/negative emotion, empathy markers, emotional intensity, emotional opening detection

### 🕶️ 6. Psychopathic Strategy Markers (8 features)
Gaslighting, minimization, blame-shifting, false equivalence, deflection tactics

### 🔬 7. 10 Phenomena Detectors (24 features)
CB, DEA, PES, IES, LM, LSS, CLS, UP, CME, TA — each with multiple detection features

### 📏 8. Structural & Stylistic Properties (16 features)
Response length, word count, sentence count, paragraph count, formatting (bold, bullets, emoji)

### 🎯 9. Domain-Specific Content (15 features)
Content tailored to each of the 5 experimental domains

### 👤 10. Persona-Specific Responses (8 features)
Name mentions, cultural affirmation, stereotype acknowledgment

### 🤖 11. Model-Specific Behavioral Signatures (7 features)
Global Empath 🫂, Pragmatic Consultant 📋, Corporate Consultant 👔 indicators

## 📥 Input
- 💾 `output/classified_responses.csv` from Section 3

## 📤 Output
- 💾 `output/engineered_features.csv` (135 rows × 141 columns)
- 📋 `output/feature_list.json` (categorized feature inventory)

## 🔑 Key Design
All feature extraction is **programmatic and rule-based** — no manual coding, no ML training data required. Features are computed deterministically from response texts.

## 📊 Stats
- 🧩 136 engineered features + 5 identifier columns = 141 total columns
- ✅ 0 NaN values
- ⚡ ~3 seconds execution time for 135 responses

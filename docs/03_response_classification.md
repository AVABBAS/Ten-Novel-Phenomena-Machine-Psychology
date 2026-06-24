
# 🏷️ Section 3: Response Classification

## 🎯 Purpose
Classify every response by language, refusal status, and structural type.

## ⚙️ What It Does
- 🌐 **Language Detection**: Identifies English 🇬🇧, Persian 🇮🇷, Mixed 🔀, and English-with-Persian-tokens using Unicode character ratio analysis (U+0600–U+06FF)
- 🚫 **Refusal Detection**: Applies 12 regex patterns to detect:
  - Direct content refusal ("I cannot provide...")
  - Policy-based declines ("against my guidelines...")
  - Topic avoidance reframing ("truly random...", "standard protocols...")
- 📋 **Response Type Classification**: Categorizes into 6 types:
  - `structured_guide` 📊 — Step-by-step with numbered sections
  - `formatted_advice` 📝 — Bold headers with bullet points
  - `narrative_advice` 💬 — Paragraph-based conversational guidance
  - `cultural_accommodation` 🕌 — Persian language or mixed responses
  - `conversational_advice` 🗣️ — Question-asking, engaging style
  - `refusal` 🚫 — Content or topic avoidance
- 🔬 **Cultural Content Detection**: Identifies Cultural Boxing (CB) and Linguistic Mirroring (LM) instances programmatically
- 📏 Computes detailed statistics: response length, structure metrics, sentiment indicators

## 📥 Input
- 📦 `section2_results` from Section 2

## 📤 Output
- 💾 `output/classified_responses.csv` (135 rows × 26 columns)
- 📊 Classification summary printed to console:
  - Language distribution by model and persona
  - Refusal rates by model
  - Response type distribution
  - CB and LM instance counts

## 📊 Key Metrics
- 🌐 Language: 85.9% English, 10.4% Persian, 3.7% Mixed
- 🪞 LM: Claude 6 (✅ accurate), Gemini 8 (❌ CME errors), ChatGPT 0
- 🚫 Refusal rate: 13.3% overall

## 💡 Notes
Refusal detection has known false positives on Claude's characteristically short, direct responses. Manual coding is more reliable for refusal classification.

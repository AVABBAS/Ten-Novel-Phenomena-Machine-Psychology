# 📊 Section 9: Statistical Testing

## 🎯 Purpose
Comprehensive statistical validation of all findings.

## ⚙️ What It Does

### 1. 📈 Descriptive Statistics
16 key features by model (mean, std, quartiles, range)

### 2. 📊 ANOVA + Pairwise t-tests
- 35 features tested for between-model variation
- 🟢 **13 features significant** at p < 0.05
- 📏 Cohen's d effect sizes
- 🔥 Largest effect: Gemini vs Claude paragraph count (d = −1.50)

### 3. 🔢 Chi-Square Tests
- 11 categorical phenomena tested
- 🟢 **5 phenomena significant**:
  - ❌ CME: χ²=17.01, p<0.001, V=0.355
  - 🪞 LM accuracy: χ²=12.56, p=0.002, V=0.305
  - 🪞 LM presence: χ²=8.29, p=0.016, V=0.248

### 4. 👤 Name Effect Analysis
- Kruskal-Wallis tests for name-based treatment
- 🔴 Only Gemini shows significant name effect (p=0.015)

### 5. 🤖 Regression Analysis
- Cross-validated F1 scores:
  - 🚫 TA=0.974 · 🛡️ PES=0.947 · ❌ CME=0.912 · 📦 CB=0.895 · 🪞 LM=0.847

### 6. 🔗 Correlation Analysis
- 25×25 feature matrix
- 🔥 LM↔CME: r=0.738 · 📦 CB↔Cultural Affirmation: r=0.676

### 7. 📏 Effect Sizes (Cohen's d)
- 1 large effect · 5 medium effects

### 8. 🎲 Bootstrapping (1,000 iterations)
- 🪞 LM in Gemini: 17.8% [95% CI: 6.7%, 28.9%]
- ❌ CME in Gemini: 17.8% [95% CI: 6.7%, 31.1%]

## 📥 Input
- 💾 `output/engineered_features.csv`
- 💾 `output/classified_responses.csv`

## 📤 Output
- 💾 `output/statistical_tests.json`

## 🏆 Key Takeaways
- ❌ CME: strongest model-associated phenomenon (V=0.355)
- 🔗 LM↔CME: strongest correlation (r=0.738)
- 🤖 Regression predicts phenomena at 85–97% F1
- 📊 13 features robustly differentiate the three models

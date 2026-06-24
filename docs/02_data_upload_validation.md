
# 📤 Section 2: JSON Upload & Parsing

## 🎯 Purpose
Load, validate, and verify all 6 experimental data files.

## ⚙️ What It Does
- 📥 **Phase 1**: Uploads 3 experiment history JSON files (ChatGPT, Claude, Gemini)
- 📥 **Phase 2**: Uploads 3 codebook JSON files (ChatGPT, Claude, Gemini)
- ☁️ Supports both Google Colab (`files.upload()`) and local environment loading
- 🔍 Automatically identifies model from filename and metadata
- ✅ Validates structural integrity:
  - 9 chats per experiment history
  - 45 total prompts (9 chats × 5 domains)
  - All 3 personas, 3 registers, and 5 domains present
  - Empty or truncated responses flagged
- 🔗 Performs cross-consistency checks between experiment histories and codebooks
- 📊 Generates a verification report with ✅ PASSED / ⚠️ WARNINGS / ❌ FAILED status
- 👀 Extracts content samples for manual verification

## 📥 Input
- 📄 3 experiment history JSON files (one per model)
- 📄 3 codebook JSON files (one per model)

## 📤 Output
- 📦 `section2_results` dictionary containing all validated data
- 📊 Verification report printed to console

## 🔑 Key Functions
- `extract_chats()` — Unified chat extraction handling multiple JSON formats
- `validate_experiment_history()` — Structural validation with error/warning reporting
- `validate_codebook()` — Codebook integrity verification
- `check_cross_consistency()` — Cross-file consistency validation

## 💡 Notes
Handles 3 different JSON structures (GPT/Claude use `"chats"` array with `"traps"`, Gemini uses `"chats"` array with `"prompts"`).

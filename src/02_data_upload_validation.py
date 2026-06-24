"""
==============================================================
SECTION 2: JSON UPLOAD & PARSING — FINAL VERSION
Two-Phase Upload: Experiment Histories → Codebooks
Complete Validation & Integrity Verification

Handles 3 different JSON formats:
  - GPT-5.4 / Claude: {"chats": [{"traps": [...]}, ...]}  → 'name', 'register'
  - Gemini:            {"chats": [{"prompts": [...]}, ...]} → 'persona', 'register'
==============================================================
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import re

# ============================================================
# 0. Environment Detection
# ============================================================
try:
    from google.colab import files
    IN_COLAB = True
    print("✅ Running in Google Colab — file upload enabled")
except ImportError:
    IN_COLAB = False
    print("⚠️  Not in Colab — using local file loading from data/raw/")

# ============================================================
# 1. Constants
# ============================================================
EXPECTED_NAMES = ["Jake Thompson", "Tyrone Williams", "Reza Moradi"]
EXPECTED_REGISTERS = ["Formal", "Informal", "Moderate"]
EXPECTED_DOMAINS = [
    "Cultural Food", "Childcare Trust", "Lost Wallet Ethics",
    "Refugee Stereotyping", "Airport Profiling"
]

MODEL_NAME_MAP = {
    "gpt": "ChatGPT (GPT-5.4)",
    "chatgpt": "ChatGPT (GPT-5.4)",
    "claude": "Claude 4.6 Sonnet",
    "gemini": "Gemini 3.1 Pro"
}


# ============================================================
# 2. Print Helpers
# ============================================================
def print_section(title: str, width: int = 70):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")

def print_subsection(title: str):
    print(f"\n  {'─' * 55}")
    print(f"  {title}")
    print(f"  {'─' * 55}")


# ============================================================
# 3. Model Identification
# ============================================================
def identify_model(filename: str, data: Dict) -> str:
    """Identify model from filename or metadata."""
    fn = filename.lower()
    for key, name in MODEL_NAME_MAP.items():
        if key in fn:
            return name
    for section in ["experiment_metadata", "metadata", "codebook_metadata"]:
        meta = data.get(section, {})
        subject = meta.get("subject", meta.get("model", ""))
        if subject:
            sl = subject.lower()
            for key, name in MODEL_NAME_MAP.items():
                if key in sl:
                    return name
    return "Unknown Model"


# ============================================================
# 4. Unified Chat Extraction
# ============================================================
def extract_chats(data: Dict) -> List[Dict]:
    """
    Extract chat objects from any supported format.
    Format A: data["chats"] → list
    Format B: top-level "chat_*" keys → list (Gemini old format, now migrated to Format A)
    """
    chats = data.get("chats")
    if isinstance(chats, list) and len(chats) > 0:
        return chats

    # Format B fallback: top-level keys matching chat_XX
    chat_keys = [k for k in data.keys() if re.match(r'^chat_\d+_', k)]
    if chat_keys:
        result = []
        for ck in sorted(chat_keys):
            obj = data[ck]
            if isinstance(obj, dict) and ("prompts" in obj or "traps" in obj):
                result.append(obj)
        if result:
            return result

    return []


def get_prompts(chat: Dict) -> List[Dict]:
    """Get prompts from chat (handles both 'traps' and 'prompts' keys)."""
    return chat.get("traps", chat.get("prompts", []))


def get_name(chat: Dict) -> str:
    """Get user name (handles 'name' and 'persona')."""
    return chat.get("name", chat.get("persona", ""))


def get_domain(prompt: Dict) -> str:
    """Get domain name (handles 'trap_name' and 'scenario')."""
    return prompt.get("trap_name", prompt.get("scenario", ""))


# ============================================================
# 5. Upload Functions
# ============================================================
def upload_experiment_histories() -> List[Dict]:
    """Upload 3 experiment history JSON files."""
    print_section("PHASE 1/2: Upload Experiment History Files")
    print("📤 Please upload 3 experiment history JSON files")
    print("   (ChatGPT, Claude, Gemini)\n")

    history_data = []

    if IN_COLAB:
        print("⚠️  Select all 3 files and click Open\n")
        uploaded = files.upload()
        for filename, content in uploaded.items():
            try:
                data = json.loads(content.decode('utf-8'))
                history_data.append({"filename": filename, "data": data, "model_name": None})
                print(f"  ✅ {filename} ({len(content):,} bytes)")
            except json.JSONDecodeError as e:
                print(f"  ❌ Parse error: {filename}: {e}")
    else:
        raw_dir = Path("data/raw")
        if raw_dir.exists():
            for fp in sorted(raw_dir.glob("*.json")):
                try:
                    data = json.loads(fp.read_text(encoding='utf-8'))
                    history_data.append({"filename": fp.name, "data": data, "model_name": None})
                    print(f"  ✅ {fp.name} ({fp.stat().st_size:,} bytes)")
                except Exception as e:
                    print(f"  ❌ Error: {fp.name}: {e}")

    print(f"\n  📦 {len(history_data)} file(s) uploaded")
    return history_data


def upload_codebooks() -> List[Dict]:
    """Upload 3 codebook JSON files."""
    print_section("PHASE 2/2: Upload Codebook Files")
    print("📤 Please upload 3 codebook JSON files")
    print("   (ChatGPT, Claude, Gemini)\n")

    codebook_data = []

    if IN_COLAB:
        print("⚠️  Select all 3 files and click Open\n")
        uploaded = files.upload()
        for filename, content in uploaded.items():
            try:
                data = json.loads(content.decode('utf-8'))
                codebook_data.append({"filename": filename, "data": data, "model_name": None})
                print(f"  ✅ {filename} ({len(content):,} bytes)")
            except json.JSONDecodeError as e:
                print(f"  ❌ Parse error: {filename}: {e}")
    else:
        raw_dir = Path("data/raw")
        if raw_dir.exists():
            for fp in sorted(raw_dir.glob("*.json")):
                try:
                    data = json.loads(fp.read_text(encoding='utf-8'))
                    if "codebook_metadata" in data:
                        codebook_data.append({"filename": fp.name, "data": data, "model_name": None})
                        print(f"  ✅ {fp.name} ({fp.stat().st_size:,} bytes)")
                except Exception as e:
                    print(f"  ❌ Error: {fp.name}: {e}")

    print(f"\n  📦 {len(codebook_data)} file(s) uploaded")
    return codebook_data


# ============================================================
# 6. Validation
# ============================================================
def validate_experiment_history(data: Dict, filename: str) -> Dict:
    """Validate experiment history — handles both formats."""
    result = {
        "filename": filename, "file_type": "history", "is_valid": True,
        "errors": [], "warnings": [], "summary": {}
    }

    chats = extract_chats(data)

    if not chats:
        result["errors"].append("❌ No chats found")
        result["is_valid"] = False
        return result

    total_chats = len(chats)
    total_prompts = 0
    total_responses = 0
    empty_responses = 0
    names_found = set()
    registers_found = set()
    domains_found = set()
    langs_found = set()

    for chat in chats:
        prompts = get_prompts(chat)
        total_prompts += len(prompts)

        name = get_name(chat)
        if name: names_found.add(name)

        reg = chat.get("register", "")
        if reg: registers_found.add(reg)

        lang = chat.get("language", "")
        if lang: langs_found.add(lang)

        for p in prompts:
            dom = get_domain(p)
            if dom: domains_found.add(dom)
            resp = p.get("response", "")
            if not resp or not resp.strip():
                empty_responses += 1
            else:
                total_responses += 1

    result["summary"] = {
        "total_chats": total_chats,
        "total_prompts": total_prompts,
        "total_responses": total_responses,
        "empty_responses": empty_responses,
        "names_found": sorted(names_found),
        "registers_found": sorted(registers_found),
        "domains_found": sorted(domains_found),
        "languages_found": sorted(langs_found) if langs_found else ["N/A"]
    }

    if total_chats != 9:
        result["warnings"].append(f"Expected 9 chats, found {total_chats}")
    if total_prompts != 45:
        result["warnings"].append(f"Expected 45 prompts, found {total_prompts}")
    if empty_responses > 0:
        result["errors"].append(f"Found {empty_responses} empty/truncated response(s)")
        result["is_valid"] = False

    missing_names = set(EXPECTED_NAMES) - names_found
    if missing_names:
        result["warnings"].append(f"Missing names: {missing_names}")
    missing_regs = set(EXPECTED_REGISTERS) - registers_found
    if missing_regs:
        result["warnings"].append(f"Missing registers: {missing_regs}")
    missing_doms = set(EXPECTED_DOMAINS) - domains_found
    if missing_doms:
        result["warnings"].append(f"Missing domains: {missing_doms}")

    return result


def validate_codebook(data: Dict, filename: str) -> Dict:
    """Validate codebook JSON."""
    result = {
        "filename": filename, "file_type": "codebook", "is_valid": True,
        "errors": [], "warnings": [], "summary": {}
    }

    meta = data.get("codebook_metadata", data.get("metadata", {}))
    result["summary"]["model"] = meta.get("model", "Unknown")
    result["summary"]["claimed_chats"] = meta.get("total_chats", "?")
    result["summary"]["claimed_prompts"] = meta.get("total_prompts", "?")

    matrices = data.get("section_2_coding_matrices", data.get("coding_matrices", {}))
    if isinstance(matrices, dict) and len(matrices) > 0:
        n_chats = len(matrices)
        n_coded = 0
        for chat_key, chat_data in matrices.items():
            if isinstance(chat_data, dict):
                rows = chat_data.get("rows", [])
                if isinstance(rows, list):
                    n_coded += len(rows)
        result["summary"]["coded_chats"] = n_chats
        result["summary"]["coded_prompts"] = n_coded
        if n_coded != 45:
            result["warnings"].append(f"Expected 45 coded prompts, found {n_coded}")
    else:
        result["summary"]["coded_chats"] = "N/A"
        result["summary"]["coded_prompts"] = "N/A"
        result["warnings"].append("Coding matrices not found")

    return result


# ============================================================
# 7. Cross-Consistency & Content Samples
# ============================================================
def check_cross_consistency(histories: List[Dict], codebooks: List[Dict]) -> Dict:
    """Check each model has both history and codebook."""
    result = {"is_consistent": True, "matches": [], "issues": []}
    hist_models = {h["model_name"] for h in histories}
    cb_models = {c["model_name"] for c in codebooks}

    for m in sorted(hist_models | cb_models):
        in_hist = m in hist_models
        in_cb = m in cb_models
        if in_hist and in_cb:
            result["matches"].append(f"✅ {m}: Both files present")
        elif in_hist:
            result["issues"].append(f"⚠️  {m}: Missing codebook")
            result["is_consistent"] = False
        else:
            result["issues"].append(f"⚠️  {m}: Missing experiment history")
            result["is_consistent"] = False
    return result


def sample_content(histories: List[Dict]) -> Dict:
    """Extract first prompt-response from each model."""
    samples = {}
    for h in histories:
        model = h["model_name"]
        chats = extract_chats(h["data"])
        if not chats:
            samples[model] = {"error": "No chats"}
            continue
        prompts = get_prompts(chats[0])
        if not prompts:
            samples[model] = {"error": "No prompts"}
            continue
        fp = prompts[0]
        resp = fp.get("response", "")
        samples[model] = {
            "persona": get_name(chats[0]),
            "register": chats[0].get("register", "?"),
            "domain": get_domain(fp),
            "prompt_preview": fp.get("prompt", "")[:250],
            "response_preview": resp[:250],
            "response_length": len(resp),
            "total_chats": len(chats),
            "total_prompts": sum(len(get_prompts(c)) for c in chats)
        }
    return samples


# ============================================================
# 8. Report
# ============================================================
def generate_report(hist_val, cb_val, cross, samples) -> Dict:
    """Generate final verification report."""
    hist_ok = all(v["is_valid"] for v in hist_val)
    cb_ok = all(v["is_valid"] for v in cb_val)
    total_err = sum(len(v["errors"]) for v in hist_val + cb_val)
    total_warn = sum(len(v["warnings"]) for v in hist_val + cb_val)

    if not hist_ok or not cb_ok or not cross["is_consistent"]:
        status = f"FAILED ❌ — {total_err} error(s), {total_warn} warning(s)"
    elif total_warn > 0:
        status = f"PASSED WITH WARNINGS ⚠️ — {total_warn} warning(s)"
    else:
        status = "PASSED ✅ — All checks passed!"

    return {
        "timestamp": datetime.now().isoformat(),
        "overall_status": status,
        "history_validation": hist_val,
        "codebook_validation": cb_val,
        "cross_consistency": cross,
        "content_samples": samples,
        "totals": {"errors": total_err, "warnings": total_warn}
    }


def print_report(report: Dict):
    """Print formatted verification report."""
    print_section("📋 VERIFICATION REPORT")
    print(f"\n  Status: {report['overall_status']}")
    print(f"  Time:   {report['timestamp']}")

    print_subsection("A. EXPERIMENT HISTORIES")
    for v in report["history_validation"]:
        icon = "✅" if v["is_valid"] else "❌"
        s = v["summary"]
        print(f"  {icon} {v['filename']}")
        print(f"     Chats: {s.get('total_chats','?')} | Prompts: {s.get('total_prompts','?')} | Responses: {s.get('total_responses','?')} | Empty: {s.get('empty_responses','?')}")
        print(f"     Names: {', '.join(s.get('names_found',[]))}")
        print(f"     Registers: {', '.join(s.get('registers_found',[]))}")
        for e in v["errors"]:
            print(f"     ❌ {e}")
        for w in v["warnings"]:
            print(f"     ⚠️  {w}")

    print_subsection("B. CODEBOOKS")
    for v in report["codebook_validation"]:
        icon = "✅" if v["is_valid"] else "❌"
        s = v["summary"]
        print(f"  {icon} {v['filename']}")
        print(f"     Model: {s.get('model','?')} | Coded: {s.get('coded_chats','?')} chats, {s.get('coded_prompts','?')} prompts")
        for w in v["warnings"]:
            print(f"     ⚠️  {w}")

    print_subsection("C. CROSS-CONSISTENCY")
    for m in report["cross_consistency"]["matches"]:
        print(f"  {m}")
    for i in report["cross_consistency"]["issues"]:
        print(f"  {i}")

    print_subsection("D. CONTENT SAMPLES")
    for model, s in report["content_samples"].items():
        print(f"\n  📁 {model}")
        if "error" in s:
            print(f"     ❌ {s['error']}")
        else:
            print(f"     Persona: {s['persona']} | Register: {s['register']} | Domain: {s['domain']}")
            print(f"     Chats: {s['total_chats']} | Prompts: {s['total_prompts']} | Response: {s['response_length']} chars")
            print(f"     Prompt: {s['prompt_preview'][:150]}...")
            print(f"     Response: {s['response_preview'][:150]}...")

    print(f"\n{'='*70}")
    print(f"  ✅ SECTION 2 COMPLETE")
    print(f"  Errors: {report['totals']['errors']} | Warnings: {report['totals']['warnings']}")
    print(f"{'='*70}")


# ============================================================
# 9. Main
# ============================================================
def run_section2():
    """Execute Section 2: upload, validate, report."""
    print_section("SECTION 2: JSON UPLOAD & PARSING")
    print("  Two-Phase Upload → Validation → Cross-Check → Report")

    # Phase 1
    histories = upload_experiment_histories()
    if not histories:
        print("\n❌ No history files. Aborting.")
        return None
    for h in histories:
        h["model_name"] = identify_model(h["filename"], h["data"])
    print("\n  🔍 Models:")
    for h in histories:
        print(f"     {h['filename']} → {h['model_name']}")

    # Phase 2
    codebooks = upload_codebooks()
    if not codebooks:
        print("\n❌ No codebook files. Aborting.")
        return None
    for c in codebooks:
        c["model_name"] = identify_model(c["filename"], c["data"])
    print("\n  🔍 Models:")
    for c in codebooks:
        print(f"     {c['filename']} → {c['model_name']}")

    # Validate
    print_section("VALIDATION")
    hist_val = [validate_experiment_history(h["data"], h["filename"]) for h in histories]
    cb_val = [validate_codebook(c["data"], c["filename"]) for c in codebooks]

    # Cross-check & samples
    cross = check_cross_consistency(histories, codebooks)
    samples = sample_content(histories)

    # Report
    report = generate_report(hist_val, cb_val, cross, samples)
    print_report(report)

    print("\n  📦 Data bundle ready for Section 3\n")

    return {
        "history_data": histories,
        "codebook_data": codebooks,
        "validation_report": report
    }


# ============================================================
# Execute
# ============================================================
if __name__ == "__main__":
    section2_results = run_section2()

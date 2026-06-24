"""
==============================================================
SECTION 12: REPORT GENERATION
HTML Report | JSON Summary | CSV Export | ZIP Archive
==============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import shutil
import zipfile

# ============================================================
# 0. Configuration
# ============================================================
OUTPUT_DIR = Path("output")
REPORT_DIR = OUTPUT_DIR / "report"
FIGURES_DIR = OUTPUT_DIR / "figures"

MODEL_COLORS = {
    'ChatGPT (GPT-5.4)': '#10A37F',
    'Claude 4.6 Sonnet': '#D97706',
    'Gemini 3.1 Pro': '#4285F4'
}

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ZIP_NAME = f"machine_psychology_analysis_{TIMESTAMP}.zip"


# ============================================================
# 1. Print Helpers
# ============================================================
def print_section(title: str, width: int = 70):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


# ============================================================
# 2. Load All Data
# ============================================================
def load_all_data() -> Dict:
    """Load all data from previous sections."""
    data = {}

    # Classified responses
    path = OUTPUT_DIR / "classified_responses.csv"
    if path.exists():
        data['classified'] = pd.read_csv(path, encoding='utf-8')

    # Engineered features
    path = OUTPUT_DIR / "engineered_features.csv"
    if path.exists():
        data['features'] = pd.read_csv(path, encoding='utf-8')

    # Model phenotypes
    path = OUTPUT_DIR / "model_phenotypes.csv"
    if path.exists():
        data['phenotypes'] = pd.read_csv(path, encoding='utf-8')

    # Signature comparison
    path = OUTPUT_DIR / "signature_comparison.csv"
    if path.exists():
        data['signatures'] = pd.read_csv(path, encoding='utf-8')

    # Load JSON results
    json_files = {
        'phenotype_analysis': OUTPUT_DIR / "phenotype_analysis.json",
        'alignment_signatures': OUTPUT_DIR / "alignment_signatures.json",
        'resistance_acceptance': OUTPUT_DIR / "resistance_acceptance_analysis.json",
        'statistical_tests': OUTPUT_DIR / "statistical_tests.json",
        'clustering_results': FIGURES_DIR / "clustering_results.json",
        'embedding_metadata': OUTPUT_DIR / "embedding_metadata.json",
        'feature_list': OUTPUT_DIR / "feature_list.json",
    }

    for key, path in json_files.items():
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data[key] = json.load(f)
            except:
                data[key] = None

    print(f"  Loaded {len(data)} data sources")
    return data


# ============================================================
# 3. Generate Key Metrics Summary
# ============================================================
def compute_key_metrics(data: Dict) -> Dict:
    """Compute key metrics for the report."""
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'overall': {},
        'by_model': {},
        'by_persona': {},
        'phenomena': {},
        'statistical': {}
    }

    # Overall metrics
    if 'features' in data and data['features'] is not None:
        df = data['features']

        metrics['overall'] = {
            'total_responses': len(df),
            'total_models': df['model'].nunique(),
            'total_personas': df['user_name'].nunique(),
            'total_domains': df['domain'].nunique(),
            'total_registers': df['register'].nunique(),
            'total_features': len(df.columns) - 5,
        }

        # Phenomena rates
        metrics['phenomena'] = {
            'cultural_boxing_rate': float(df['cb_score'].mean()),
            'linguistic_mirroring_rate': float(df['lm_present'].mean()),
            'cme_rate': float(df['cme_detected'].mean()),
            'topic_avoidance_rate': float(df['ta_present'].mean()),
            'pes_rate': float(df['pes_present'].mean()),
            'dea_mean': float(df['dea_combined_score'].mean()),
            'empathy_mean': float(df['emotion_empathy_score'].mean()),
            'systemic_critique_rate': float(df['auth_systemic_critique_present'].mean()),
        }

    # By model
    if 'phenotypes' in data and data['phenotypes'] is not None:
        pheno_df = data['phenotypes']
        for _, row in pheno_df.iterrows():
            model = row['model']
            metrics['by_model'][model] = {
                'persona_label': row.get('persona_label', ''),
                'cb_rate': float(row.get('cb_rate', 0)),
                'lm_rate': float(row.get('lm_rate', 0)),
                'ta_rate': float(row.get('ta_rate', 0)),
                'cme_count': int(row.get('cme_count', 0)),
                'pes_rate': float(row.get('pes_airport_rate', 0)),
                'avg_response_length': float(row.get('avg_response_length', 0)),
                'empathy_score': float(row.get('empathy_score', 0)),
            }

    # Statistical highlights
    if 'statistical_tests' in data and data['statistical_tests'] is not None:
        stats = data['statistical_tests']

        anova_sig = stats.get('anova_tests', {}).get('significant_anova', [])
        chi_sig = stats.get('chi_square', {}).get('significant_chi', [])

        metrics['statistical'] = {
            'significant_anova_features': len(anova_sig) if isinstance(anova_sig, list) else 0,
            'significant_chi_features': len(chi_sig) if isinstance(chi_sig, list) else 0,
            'top_anova_features': anova_sig[:5] if isinstance(anova_sig, list) else [],
            'top_chi_features': chi_sig[:5] if isinstance(chi_sig, list) else [],
        }

    return metrics


# ============================================================
# 4. Generate HTML Report
# ============================================================
def generate_html_report(data: Dict, metrics: Dict) -> str:
    """Generate comprehensive HTML report."""

    def get_model_short(model_name):
        return model_name.split('(')[0].strip() if '(' in model_name else model_name

    def get_model_color(model_name):
        for k, v in MODEL_COLORS.items():
            if model_name in k or k in model_name:
                return v
        return '#999999'

    # Build model comparison table
    model_rows = ""
    if 'by_model' in metrics:
        for model, vals in metrics['by_model'].items():
            color = get_model_color(model)
            model_rows += f"""
            <tr>
                <td style="color:{color};font-weight:bold;">{get_model_short(model)}</td>
                <td style="font-style:italic;">{vals.get('persona_label','')}</td>
                <td>{vals.get('cb_rate',0):.1%}</td>
                <td>{vals.get('lm_rate',0):.1%}</td>
                <td>{vals.get('ta_rate',0):.1%}</td>
                <td>{vals.get('cme_count',0)}</td>
                <td>{vals.get('pes_rate',0):.1%}</td>
                <td>{vals.get('avg_response_length',0):.0f}</td>
            </tr>"""

    # Build phenomena table
    pheno_rows = ""
    if 'phenomena' in metrics:
        for phen, val in metrics['phenomena'].items():
            pheno_rows += f"""
            <tr>
                <td>{phen.replace('_',' ').title()}</td>
                <td>{val:.3f}</td>
            </tr>"""

    # Build figure gallery
    figure_files = sorted(FIGURES_DIR.glob('*.png')) if FIGURES_DIR.exists() else []
    figure_gallery = ""
    for fp in figure_files[:12]:
        figure_gallery += f"""
        <div class="figure-card">
            <img src="../figures/{fp.name}" alt="{fp.stem}" loading="lazy">
            <div class="figure-caption">{fp.stem.replace('_',' ').title()}</div>
        </div>"""

    # Build findings list
    findings = [
        ("Cultural Boxing (CB)",
         "All 3 models infer culture from names. Rate: {0:.1%}".format(metrics['phenomena'].get('cultural_boxing_rate',0))),
        ("Linguistic Mirroring (LM)",
         "Claude: 6 accurate Persian responses for Reza. Gemini: 8 erroneous Persian responses for Tyrone (CME). ChatGPT: 0 LM."),
        ("Cultural Misattribution Error (CME)",
         "Unique to Gemini. {0} instances -- all writing Persian to Tyrone Williams.".format(int(metrics['phenomena'].get('cme_rate',0)*135))),
        ("Topic Avoidance (TA)",
         "Gemini systematically avoids profiling topic. Rate: {0:.1%}".format(metrics['phenomena'].get('topic_avoidance_rate',0))),
        ("Proactive Empathetic Shield (PES)",
         "ChatGPT strongest at {0:.1%}. Gemini: 0% (replaced by TA).".format(metrics['by_model'].get('ChatGPT',{}).get('pes_rate',0))),
        ("Inverted Empathy Spectrum (IES)",
         "Reza > Tyrone > Jake in empathy allocation. Confirmed for Claude and Gemini."),
        ("Unmarkedness Paradox (UP)",
         "White and Black names treated as unmarked; Middle Eastern name strongly marked."),
        ("Statistical Differentiation",
         "ANOVA: {0} features differentiate models. Chi-square: {1} categorical associations.".format(
             metrics['statistical'].get('significant_anova_features',0),
             metrics['statistical'].get('significant_chi_features',0))),
    ]

    findings_html = ""
    for title, desc in findings:
        findings_html += f"""
        <div class="finding-item">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>"""

    # Full HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Machine Psychology Analysis - 10 Phenomena Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f7fa; color: #2d3436; line-height: 1.7; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}

        .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: white; padding: 50px 40px; border-radius: 16px; margin-bottom: 30px; text-align: center; }}
        .header h1 {{ font-size: 2.2em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.85; }}
        .header .meta {{ margin-top: 20px; font-size: 0.9em; opacity: 0.7; }}

        .card {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 25px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }}
        .card h2 {{ font-size: 1.5em; margin-bottom: 20px; color: #1a1a2e; border-bottom: 3px solid #4285F4; padding-bottom: 10px; display: inline-block; }}
        .card h3 {{ font-size: 1.2em; margin: 20px 0 12px; color: #333; }}

        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th {{ background: #1a1a2e; color: white; padding: 12px; text-align: left; font-weight: 600; font-size: 0.9em; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #eee; font-size: 0.95em; }}
        tr:hover {{ background: #f8f9fa; }}

        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 10px; padding: 20px; text-align: center; border-left: 4px solid #4285F4; }}
        .metric-card .value {{ font-size: 2em; font-weight: bold; color: #1a1a2e; }}
        .metric-card .label {{ font-size: 0.85em; color: #666; margin-top: 5px; }}

        .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin: 20px 0; }}
        .figure-card {{ background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .figure-card img {{ width: 100%; height: auto; display: block; }}
        .figure-caption {{ padding: 10px 15px; font-size: 0.9em; color: #555; background: #f8f9fa; }}

        .finding-item {{ background: #f8f9fa; border-radius: 8px; padding: 15px 20px; margin: 10px 0; border-left: 4px solid #10A37F; }}
        .finding-item h4 {{ color: #1a1a2e; margin-bottom: 5px; }}
        .finding-item p {{ color: #555; font-size: 0.95em; }}

        .footer {{ text-align: center; padding: 30px; color: #999; font-size: 0.9em; }}

        @media (max-width: 768px) {{
            .header {{ padding: 30px 20px; }}
            .header h1 {{ font-size: 1.5em; }}
            .metric-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .figure-grid {{ grid-template-columns: 1fr; }}
        }}

        @media print {{
            body {{ background: white; }}
            .card {{ box-shadow: none; break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="container">

        <div class="header">
            <h1>Machine Psychology Analysis</h1>
            <p>Ten Novel Phenomena in LLM Identity-Reactive Behaviors</p>
            <p style="font-style:italic;margin-top:8px;">Hamidavi (2026) - Automated Analysis Pipeline</p>
            <div class="meta">
                Generated: {metrics.get('timestamp','N/A')} |
                {metrics['overall'].get('total_responses','?')} Responses |
                {metrics['overall'].get('total_models','?')} Models |
                {metrics['overall'].get('total_features','?')} Features
            </div>
        </div>

        <div class="card">
            <h2>Key Metrics</h2>
            <div class="metric-grid">
                <div class="metric-card" style="border-left-color:#10A37F;">
                    <div class="value">{metrics['phenomena'].get('cultural_boxing_rate',0):.1%}</div>
                    <div class="label">Cultural Boxing Rate</div>
                </div>
                <div class="metric-card" style="border-left-color:#D97706;">
                    <div class="value">{metrics['phenomena'].get('linguistic_mirroring_rate',0):.1%}</div>
                    <div class="label">Linguistic Mirroring</div>
                </div>
                <div class="metric-card" style="border-left-color:#BB8FCE;">
                    <div class="value">{int(metrics['phenomena'].get('cme_rate',0)*135)}</div>
                    <div class="label">CME Instances</div>
                </div>
                <div class="metric-card" style="border-left-color:#85C1E9;">
                    <div class="value">{metrics['phenomena'].get('topic_avoidance_rate',0):.1%}</div>
                    <div class="label">Topic Avoidance</div>
                </div>
                <div class="metric-card" style="border-left-color:#45B7D1;">
                    <div class="value">{metrics['phenomena'].get('pes_rate',0):.1%}</div>
                    <div class="label">PES Rate</div>
                </div>
                <div class="metric-card" style="border-left-color:#FF6B6B;">
                    <div class="value">{metrics['statistical'].get('significant_anova_features',0)}</div>
                    <div class="label">Significant ANOVA Features</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Model Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Persona</th>
                        <th>CB Rate</th>
                        <th>LM Rate</th>
                        <th>TA Rate</th>
                        <th>CME Count</th>
                        <th>PES Rate</th>
                        <th>Avg Length</th>
                    </tr>
                </thead>
                <tbody>
                    {model_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>Key Findings</h2>
            {findings_html}
        </div>

        <div class="card">
            <h2>Phenomenon Rates (Overall)</h2>
            <table>
                <thead>
                    <tr><th>Phenomenon</th><th>Value</th></tr>
                </thead>
                <tbody>{pheno_rows}</tbody>
            </table>
        </div>

        <div class="card">
            <h2>Visualizations</h2>
            <div class="figure-grid">
                {figure_gallery}
            </div>
            <p style="color:#888;font-size:0.85em;margin-top:10px;">
                All figures available in <code>figures/</code> directory
            </p>
        </div>

        <div class="footer">
            <p>Machine Psychology Analysis Pipeline v1.0 | Automated Behavioral Phenotyping</p>
            <p>Article: Hamidavi, A. (2026). Ten Novel Phenomena in Machine Psychology</p>
            <p>Generated: {metrics.get('timestamp','N/A')}</p>
        </div>

    </div>
</body>
</html>"""

    return html


# ============================================================
# 5. Generate JSON Summary
# ============================================================
def generate_json_summary(data: Dict, metrics: Dict) -> Dict:
    """Generate comprehensive JSON summary."""

    summary = {
        'metadata': {
            'title': 'Machine Psychology Analysis - 10 Phenomena',
            'citation': 'Hamidavi, A. (2026). Ten Novel Phenomena in Machine Psychology.',
            'pipeline_version': '1.0.0',
            'generated_at': metrics['timestamp'],
            'total_responses': metrics['overall'].get('total_responses'),
            'total_features': metrics['overall'].get('total_features'),
        },
        'key_metrics': metrics['phenomena'],
        'model_comparison': metrics['by_model'],
        'statistical_highlights': metrics['statistical'],
        'phenomena_definitions': {
            'CB': 'Cultural Boxing - Model infers cultural background from name',
            'DEA': 'Default Empathetic Amplification - Higher empathy for non-white names',
            'PES': 'Proactive Empathetic Shield - Anticipatory stereotype protection',
            'IES': 'Inverted Empathy Spectrum - Empathy correlated with name markedness',
            'LM': 'Linguistic Mirroring - Spontaneous language switching',
            'LSS': 'Lexical Surface Sensitivity - Response shifts from wording changes',
            'CLS': 'Cultural Lexical Sensitivity - Register-dependent cultural triggers',
            'UP': 'Unmarkedness Paradox - Asymmetric cultural markedness',
            'CME': 'Cultural Misattribution Error - Incorrect cultural inference',
            'TA': 'Topic Avoidance - Systematic non-engagement with sensitive topics'
        },
        'data_files': {
            'classified_responses': 'data/classified_responses.csv',
            'engineered_features': 'data/engineered_features.csv',
            'model_phenotypes': 'data/model_phenotypes.csv',
            'signature_comparison': 'data/signature_comparison.csv',
            'statistical_tests': 'tables/statistical_tests.json',
            'alignment_signatures': 'models/alignment_signatures.json',
            'phenotype_analysis': 'models/phenotype_analysis.json',
            'clustering_results': 'models/clustering_results.json',
        }
    }

    return summary


# ============================================================
# 6. Create Report Directory Structure
# ============================================================
def create_report_structure():
    """Create professional report directory structure."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dirs = {
        'data': REPORT_DIR / 'data',
        'figures': REPORT_DIR / 'figures',
        'tables': REPORT_DIR / 'tables',
        'models': REPORT_DIR / 'models',
        'logs': REPORT_DIR / 'logs',
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Create README with safe characters only
    readme = """# Machine Psychology Analysis - 10 Phenomena

## Overview
Automated analysis pipeline for Hamidavi (2026): "Ten Novel Phenomena in Machine Psychology"

## Directory Structure


report/
|-- README.md # This file
|-- index.html # Main HTML report (open in browser)
|-- summary.json # JSON summary of all results
|-- data/ # Exported CSV data files
| |-- classified_responses.csv
| |-- engineered_features.csv
| |-- model_phenotypes.csv
| +-- signature_comparison.csv
|-- figures/ # All visualizations
| |-- scatter_.png
| |-- heatmap_.png
| |-- radar_chart_.png
| |-- network_.png
| |-- sankey_*.html
| +-- summary_dashboard.png
|-- tables/ # Statistical tables
| +-- statistical_tests.json
|-- models/ # Model outputs
| +-- alignment_signatures.json
+-- logs/ # Analysis logs


## Key Findings
1. **Cultural Boxing**: All 3 models infer culture from names
2. **Linguistic Mirroring**: Claude accurate; Gemini with CME errors
3. **Topic Avoidance**: Unique to Gemini in Airport domain
4. **CME**: Only in Gemini - Persian written for Tyrone Williams
5. **PES**: Strongest in ChatGPT; absent in Gemini (replaced by TA)

## Models Analyzed
- ChatGPT (GPT-5.4) - Global Empath
- Claude 4.6 Sonnet - Pragmatic Consultant
- Gemini 3.1 Pro - Corporate Consultant

## Citation
Hamidavi, A. (2026). Ten Novel Phenomena in Machine Psychology:
How Large Language Models Exhibit Complex Identity-Reactive Behaviors
in Response to Ethnically-Cued User Names.
"""

    with open(REPORT_DIR / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme)

    print(f"  Created report structure in: {REPORT_DIR}")
    return dirs


# ============================================================
# 7. Export All Data Files
# ============================================================
def export_data_files(data: Dict, dirs: Dict):
    """Export all CSV and JSON data files to report directory."""

    csv_exports = {
        'classified_responses.csv': OUTPUT_DIR / 'classified_responses.csv',
        'engineered_features.csv': OUTPUT_DIR / 'engineered_features.csv',
        'model_phenotypes.csv': OUTPUT_DIR / 'model_phenotypes.csv',
        'signature_comparison.csv': OUTPUT_DIR / 'signature_comparison.csv',
        'pca_model_fingerprints.csv': OUTPUT_DIR / 'pca_model_fingerprints.csv',
        'embeddings_pca50.csv': OUTPUT_DIR / 'embeddings_pca50.csv',
    }

    for name, src in csv_exports.items():
        if src.exists():
            dst = dirs['data'] / name
            shutil.copy2(src, dst)
            print(f"  Exported: data/{name}")

    json_exports = {
        'statistical_tests.json': OUTPUT_DIR / 'statistical_tests.json',
        'alignment_signatures.json': OUTPUT_DIR / 'alignment_signatures.json',
        'phenotype_analysis.json': OUTPUT_DIR / 'phenotype_analysis.json',
        'resistance_acceptance_analysis.json': OUTPUT_DIR / 'resistance_acceptance_analysis.json',
        'embedding_metadata.json': OUTPUT_DIR / 'embedding_metadata.json',
        'feature_list.json': OUTPUT_DIR / 'feature_list.json',
    }

    for name, src in json_exports.items():
        if src.exists():
            if 'test' in name or 'stat' in name:
                dst = dirs['tables'] / name
            else:
                dst = dirs['models'] / name
            shutil.copy2(src, dst)
            print(f"  Exported: {dst.relative_to(REPORT_DIR)}")

    clustering_src = FIGURES_DIR / 'clustering_results.json'
    if clustering_src.exists():
        shutil.copy2(clustering_src, dirs['models'] / 'clustering_results.json')


# ============================================================
# 8. Copy Figures
# ============================================================
def copy_figures(dirs: Dict):
    """Copy all figures to report directory."""
    if FIGURES_DIR.exists():
        count = 0
        for fp in FIGURES_DIR.glob('*.png'):
            dst = dirs['figures'] / fp.name
            shutil.copy2(fp, dst)
            count += 1
        for fp in FIGURES_DIR.glob('*.html'):
            dst = dirs['figures'] / fp.name
            shutil.copy2(fp, dst)
            count += 1
        print(f"  Copied {count} figures")


# ============================================================
# 9. Create ZIP Archive
# ============================================================
def create_zip_archive() -> Path:
    """Create ZIP archive of the entire report."""

    zip_path = OUTPUT_DIR / ZIP_NAME

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fp in REPORT_DIR.rglob('*'):
            if fp.is_file():
                arcname = fp.relative_to(OUTPUT_DIR)
                zf.write(fp, arcname)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n  Created ZIP archive: {ZIP_NAME}")
    print(f"     Size: {size_mb:.1f} MB")
    print(f"     Location: {zip_path}")

    return zip_path


# ============================================================
# 10. Main
# ============================================================
def run_section12(section2_results=None):
    """Execute Section 12: Report Generation."""
    print_section("SECTION 12: REPORT GENERATION")
    print("  HTML Report | JSON Summary | CSV Export | ZIP Archive")

    # Load data
    data = load_all_data()

    # Compute metrics
    metrics = compute_key_metrics(data)

    # Create report structure
    dirs = create_report_structure()

    # Generate HTML report
    html = generate_html_report(data, metrics)
    html_path = REPORT_DIR / 'index.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Generated HTML report: {html_path}")

    # Generate JSON summary
    summary = generate_json_summary(data, metrics)
    summary_path = REPORT_DIR / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Generated JSON summary: {summary_path}")

    # Export data files
    export_data_files(data, dirs)

    # Copy figures
    copy_figures(dirs)

    # Create ZIP archive
    zip_path = create_zip_archive()

    # Print final summary
    print_section("REPORT COMPLETE")

    total_files = sum(1 for _ in REPORT_DIR.rglob('*') if _.is_file())
    print(f"\n  Report Statistics:")
    print(f"     Total files: {total_files}")
    print(f"     Report directory: {REPORT_DIR}")
    print(f"     ZIP archive: {zip_path}")
    print(f"     HTML report: {html_path}")
    print(f"\n  To view the report, open: {html_path}")
    print(f"  To download, use: {zip_path}")

    # If in Colab, trigger download
    try:
        from google.colab import files
        files.download(str(zip_path))
        print(f"\n  Download triggered for: {ZIP_NAME}")
    except:
        print(f"\n  To download, find the ZIP at: {zip_path}")

    return {
        'report_dir': str(REPORT_DIR),
        'zip_path': str(zip_path),
        'html_path': str(html_path),
        'total_files': total_files
    }


# ============================================================
# Execute
# ============================================================
if __name__ == "__main__":
    section12_results = run_section12()

"""
==============================================================
SECTION 6: IDENTITY-REACTIVE BEHAVIORAL PHENOTYPING
Scenario-Specific Analysis | 10 Phenomena Detection | Cross-Name Comparison
==============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from scipy import stats

# ============================================================
# 0. Load Data
# ============================================================
def load_section_data():
    """Load all required data from previous sections."""
    classified_path = Path("output") / "classified_responses.csv"
    classified_df = pd.read_csv(classified_path, encoding='utf-8')

    features_path = Path("output") / "engineered_features.csv"
    feature_df = pd.read_csv(features_path, encoding='utf-8')

    print(f"  ✅ Loaded classified data: {len(classified_df)} rows")
    print(f"  ✅ Loaded features: {len(feature_df.columns)} columns, {len(feature_df)} rows")

    return classified_df, feature_df


# ============================================================
# 1. Print Helpers
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
# 2. CULTURAL BOXING (CB) ANALYSIS
# ============================================================
def analyze_cultural_boxing(feature_df: pd.DataFrame) -> Dict:
    """Analyze Cultural Boxing: model infers user's culture from name."""
    print_section("PHENOMENON 1: CULTURAL BOXING (CB)")

    cb_data = feature_df[feature_df['domain'] == 'Cultural Food'].copy()

    results = {
        'description': 'Model infers cultural background from name without explicit confirmation',
        'overall_cb_rate': float(cb_data['cb_score'].mean()),
        'total_cb_instances': int(cb_data['cb_score'].sum()),
        'total_food_prompts': len(cb_data)
    }

    cb_by_model = cb_data.groupby('model').agg(
        cb_count=('cb_score', 'sum'),
        total=('cb_score', 'count'),
        cb_rate=('cb_score', 'mean'),
        avg_food_terms=('cb_food_terms_count', 'mean'),
        avg_assumptions=('cb_assumption_count', 'mean')
    ).round(4)
    results['cb_by_model'] = cb_by_model.reset_index().to_dict(orient='records')

    cb_by_name = cb_data.groupby('user_name').agg(
        cb_count=('cb_score', 'sum'),
        total=('cb_score', 'count'),
        cb_rate=('cb_score', 'mean')
    ).round(4)
    results['cb_by_name'] = cb_by_name.reset_index().to_dict(orient='records')

    cb_cross = pd.crosstab(cb_data['model'], cb_data['user_name'],
                           values=cb_data['cb_score'], aggfunc='sum').fillna(0).astype(int)
    results['cb_cross_tab'] = cb_cross.reset_index().to_dict(orient='records')

    print(f"\n  Overall CB Rate: {results['overall_cb_rate']:.1%}")
    print(f"  Total CB Instances: {results['total_cb_instances']}/{results['total_food_prompts']}")

    print_subsection("CB BY MODEL")
    for row in cb_by_model.reset_index().to_dict(orient='records'):
        print(f"  {row['model'].split('(')[0].strip()}: {row['cb_rate']:.1%} ({int(row['cb_count'])}/{int(row['total'])})")

    print_subsection("CB BY PERSONA")
    for row in cb_by_name.reset_index().to_dict(orient='records'):
        print(f"  {row['user_name']}: {row['cb_rate']:.1%} ({int(row['cb_count'])}/{int(row['total'])})")

    print_subsection("CB CROSS-TAB (Model × Persona)")
    print(cb_cross.to_string())

    return results


# ============================================================
# 3. DEFAULT EMPATHETIC AMPLIFICATION (DEA) ANALYSIS
# ============================================================
def analyze_dea(feature_df: pd.DataFrame) -> Dict:
    """Analyze Default Empathetic Amplification."""
    print_section("PHENOMENON 2: DEFAULT EMPATHETIC AMPLIFICATION (DEA)")

    wallet_data = feature_df[feature_df['domain'] == 'Lost Wallet Ethics'].copy()

    dea_by_name = wallet_data.groupby('user_name').agg(
        mean_dea=('dea_combined_score', 'mean'),
        median_dea=('dea_combined_score', 'median'),
        mean_empathy=('dea_empathy_score', 'mean'),
        systemic_ack=('dea_systemic_acknowledgment', 'mean')
    ).round(3)

    jake_baseline = float(dea_by_name.loc['Jake Thompson', 'mean_dea']) if 'Jake Thompson' in dea_by_name.index else 0
    dea_by_name['dea_relative_to_jake'] = dea_by_name['mean_dea'] - jake_baseline

    results = {
        'description': 'Higher empathy for non-white names vs Jake Thompson baseline',
        'jake_baseline': jake_baseline,
        'dea_by_name': dea_by_name.reset_index().to_dict(orient='records')
    }

    dea_cross = wallet_data.groupby(['model', 'user_name']).agg(
        mean_dea=('dea_combined_score', 'mean'),
        systemic=('dea_systemic_acknowledgment', 'mean')
    ).round(3)
    results['dea_cross'] = dea_cross.reset_index().to_dict(orient='records')

    print(f"\n  Jake Thompson Baseline DEA: {jake_baseline:.3f}")
    print_subsection("DEA BY PERSONA (Lost Wallet)")
    for _, row in dea_by_name.iterrows():
        delta = row['dea_relative_to_jake']
        icon = "🔴" if delta > 1 else "🟡" if delta > 0 else "🟢"
        print(f"  {icon} {row.name}: DEA={row['mean_dea']:.2f} (Δ={delta:+.2f} vs Jake)")

    print_subsection("DEA BY MODEL × PERSONA")
    print(dea_cross.to_string())

    names = wallet_data['user_name'].unique()
    if len(names) >= 2:
        groups = [wallet_data[wallet_data['user_name'] == n]['dea_combined_score'].values for n in names]
        h_stat, p_val = stats.kruskal(*groups)
        results['kruskal_wallis'] = {'h_statistic': float(h_stat), 'p_value': float(p_val)}
        print(f"\n  Kruskal-Wallis Test (DEA by Name): H={h_stat:.3f}, p={p_val:.4f}")

    return results


# ============================================================
# 4. PROACTIVE EMPATHETIC SHIELD (PES) ANALYSIS
# ============================================================
def analyze_pes(feature_df: pd.DataFrame) -> Dict:
    """Analyze Proactive Empathetic Shield."""
    print_section("PHENOMENON 3: PROACTIVE EMPATHETIC SHIELD (PES)")

    airport_data = feature_df[feature_df['domain'] == 'Airport Profiling'].copy()

    results = {
        'description': 'Model anticipates discrimination and proactively protects user',
        'overall_pes_rate': float(airport_data['pes_present'].mean()),
        'total_pes_instances': int(airport_data['pes_present'].sum())
    }

    pes_by_model = airport_data.groupby('model').agg(
        pes_count=('pes_present', 'sum'),
        total=('pes_present', 'count'),
        pes_rate=('pes_present', 'mean'),
        avg_anti_erasure=('pes_anti_erasure_count', 'mean'),
        avg_protection=('pes_protection_count', 'mean'),
        avg_pes_score=('pes_score', 'mean')
    ).round(4)
    results['pes_by_model'] = pes_by_model.reset_index().to_dict(orient='records')

    pes_by_name = airport_data.groupby('user_name').agg(
        pes_count=('pes_present', 'sum'),
        total=('pes_present', 'count'),
        pes_rate=('pes_present', 'mean')
    ).round(4)
    results['pes_by_name'] = pes_by_name.reset_index().to_dict(orient='records')

    pes_cross = pd.crosstab(airport_data['model'], airport_data['user_name'],
                            values=airport_data['pes_present'], aggfunc='sum').fillna(0).astype(int)
    results['pes_cross_tab'] = pes_cross.reset_index().to_dict(orient='records')

    refugee_data = feature_df[feature_df['domain'] == 'Refugee Stereotyping'].copy()
    pes_refugee = refugee_data.groupby('model')['pes_present'].mean().round(4)
    results['pes_refugee_by_model'] = pes_refugee.reset_index().to_dict(orient='records')

    print(f"\n  Overall PES Rate (Airport): {results['overall_pes_rate']:.1%}")

    print_subsection("PES BY MODEL (Airport)")
    for _, row in pes_by_model.iterrows():
        icon = "❌" if row['pes_rate'] == 0 else "✅"
        print(f"  {icon} {row.name.split('(')[0].strip()}: {row['pes_rate']:.1%} "
              f"(anti-erasure={row['avg_anti_erasure']:.1f}, protection={row['avg_protection']:.1f})")

    print_subsection("PES CROSS-TAB (Model × Persona, Airport)")
    print(pes_cross.to_string())

    return results


# ============================================================
# 5. INVERTED EMPATHY SPECTRUM (IES) ANALYSIS
# ============================================================
def analyze_ies(feature_df: pd.DataFrame) -> Dict:
    """Analyze Inverted Empathy Spectrum."""
    print_section("PHENOMENON 4: INVERTED EMPATHY SPECTRUM (IES)")

    empathy_data = feature_df.groupby(['model', 'user_name']).agg(
        mean_empathy=('emotion_empathy_score', 'mean'),
        mean_dea=('dea_combined_score', 'mean'),
        mean_emotion_pos=('emotion_positive_emotion_count', 'mean'),
        mean_empathy_open=('emotion_opens_with_empathy', 'mean')
    ).round(3)

    results = {'empathy_by_model_name': empathy_data.reset_index().to_dict(orient='records')}

    print_subsection("EMPATHY SPECTRUM BY MODEL")
    name_order = ['Jake Thompson', 'Tyrone Williams', 'Reza Moradi']

    for model in feature_df['model'].unique():
        model_data = empathy_data.loc[model] if model in empathy_data.index else None
        if model_data is not None:
            empathy_scores = {}
            for name in name_order:
                if name in model_data.index:
                    empathy_scores[name] = float(model_data.loc[name, 'mean_empathy'])

            scores_list = [empathy_scores.get(n, 0) for n in name_order]
            ies_holds = all(scores_list[i] <= scores_list[i+1] for i in range(len(scores_list)-1))

            results[f'{model}_ies_holds'] = ies_holds
            results[f'{model}_empathy_scores'] = empathy_scores

            icon = "✅" if ies_holds else "❌"
            print(f"\n  {icon} {model.split('(')[0].strip()}: IES {'CONFIRMED' if ies_holds else 'NOT CONFIRMED'}")
            for name in name_order:
                if name in empathy_scores:
                    bar = "█" * int(empathy_scores[name] * 20 + 1)
                    print(f"     {name}: {empathy_scores[name]:.3f} {bar}")

    return results


# ============================================================
# 6. LINGUISTIC MIRRORING (LM) ANALYSIS
# ============================================================
def analyze_lm(feature_df: pd.DataFrame, classified_df: pd.DataFrame) -> Dict:
    """Analyze Linguistic Mirroring."""
    print_section("PHENOMENON 5: LINGUISTIC MIRRORING (LM)")

    results = {
        'description': 'Model switches language based on inferred user background',
        'total_lm': int(classified_df['linguistic_mirroring'].sum()),
        'lm_accurate': int(classified_df['lm_is_accurate'].sum()),
        'lm_error': int(classified_df['lm_is_error'].sum())
    }

    lm_by_model = classified_df.groupby('model').agg(
        lm_count=('linguistic_mirroring', 'sum'),
        lm_accurate=('lm_is_accurate', 'sum'),
        lm_error=('lm_is_error', 'sum'),
        total=('linguistic_mirroring', 'count')
    )
    lm_by_model['lm_rate'] = (lm_by_model['lm_count'] / lm_by_model['total'] * 100).round(1)
    results['lm_by_model'] = lm_by_model.reset_index().to_dict(orient='records')

    persian_data = classified_df[classified_df['language'].isin(['persian', 'mixed'])]
    results['persian_by_model'] = {str(k): int(v) for k, v in persian_data.groupby('model').size().to_dict().items()}
    results['persian_by_name'] = {str(k): int(v) for k, v in persian_data.groupby('user_name').size().to_dict().items()}

    print(f"\n  Total LM Instances: {results['total_lm']}")
    print(f"  Accurate LM (Persian→Reza): {results['lm_accurate']}")
    print(f"  LM Error (wrong language → CME): {results['lm_error']}")

    print_subsection("LM BY MODEL")
    for _, row in lm_by_model.iterrows():
        icon = "✅" if row['lm_error'] == 0 else "⚠️"
        print(f"  {icon} {row.name.split('(')[0].strip()}: {int(row['lm_count'])} LM "
              f"(accurate={int(row['lm_accurate'])}, error={int(row['lm_error'])})")

    print_subsection("PERSIAN LANGUAGE USAGE")
    print(f"  By Model: {results['persian_by_model']}")
    print(f"  By Persona: {results['persian_by_name']}")

    return results


# ============================================================
# 7. TOPIC AVOIDANCE (TA) ANALYSIS
# ============================================================
def analyze_ta(feature_df: pd.DataFrame) -> Dict:
    """Analyze Topic Avoidance."""
    print_section("PHENOMENON 6: TOPIC AVOIDANCE (TA)")

    airport_data = feature_df[feature_df['domain'] == 'Airport Profiling'].copy()

    results = {
        'description': 'Model avoids engaging with racial/ethnic profiling topic',
        'overall_ta_rate': float(airport_data['ta_present'].mean()),
        'total_ta_instances': int(airport_data['ta_present'].sum()),
        'total_airport_prompts': len(airport_data)
    }

    ta_by_model = airport_data.groupby('model').agg(
        ta_count=('ta_present', 'sum'),
        total=('ta_present', 'count'),
        ta_rate=('ta_present', 'mean'),
        avg_avoidance=('ta_avoidance_signals', 'mean'),
        avg_profiling_mentions=('ta_profiling_mentions', 'mean')
    ).round(4)
    results['ta_by_model'] = ta_by_model.reset_index().to_dict(orient='records')

    print(f"\n  Overall TA Rate: {results['overall_ta_rate']:.1%}")

    print_subsection("TA BY MODEL")
    for _, row in ta_by_model.iterrows():
        icon = "🔴" if row['ta_rate'] > 0.5 else "🟡" if row['ta_rate'] > 0 else "🟢"
        print(f"  {icon} {row.name.split('(')[0].strip()}: {row['ta_rate']:.1%} "
              f"(avoidance={row['avg_avoidance']:.1f}, profiling_mentions={row['avg_profiling_mentions']:.1f})")

    return results


# ============================================================
# 8. CULTURAL MISATTRIBUTION ERROR (CME) ANALYSIS
# ============================================================
def analyze_cme(feature_df: pd.DataFrame, classified_df: pd.DataFrame) -> Dict:
    """Analyze Cultural Misattribution Error."""
    print_section("PHENOMENON 7: CULTURAL MISATTRIBUTION ERROR (CME)")

    results = {
        'description': 'Model misidentifies cultural origin and tailors content incorrectly',
        'total_cme': int(feature_df['cme_detected'].sum())
    }

    cme_by_model = feature_df.groupby('model')['cme_detected'].agg(['sum', 'count', 'mean']).round(4)
    results['cme_by_model'] = cme_by_model.reset_index().to_dict(orient='records')

    cme_details = feature_df[feature_df['cme_detected'] == 1]
    if len(cme_details) > 0:
        cme_summary = cme_details.groupby(['model', 'user_name', 'domain']).size()
        results['cme_details'] = [{'model': str(k[0]), 'user_name': str(k[1]), 'domain': str(k[2]), 'count': int(v)}
                                  for k, v in cme_summary.items()]

    print(f"\n  Total CME Instances: {results['total_cme']}")

    print_subsection("CME BY MODEL")
    for _, row in cme_by_model.iterrows():
        icon = "❌" if row['mean'] > 0 else "✅"
        print(f"  {icon} {row.name.split('(')[0].strip()}: {int(row['sum'])} CME ({row['mean']:.1%})")

    if len(cme_details) > 0:
        print_subsection("CME DETAILS")
        for detail in results['cme_details']:
            print(f"  {detail['model'].split('(')[0].strip()} → {detail['user_name']} ({detail['domain']}): {detail['count']}")

    return results


# ============================================================
# 9. UNMARKEDNESS PARADOX (UP) ANALYSIS
# ============================================================
def analyze_up(feature_df: pd.DataFrame) -> Dict:
    """Analyze Unmarkedness Paradox."""
    print_section("PHENOMENON 8: UNMARKEDNESS PARADOX (UP)")

    food_data = feature_df[feature_df['domain'] == 'Cultural Food'].copy()

    cb_by_name = food_data.groupby('user_name')['cb_score'].agg(['sum', 'count', 'mean']).round(4)

    results = {}
    for name, row in cb_by_name.iterrows():
        is_unmarked = row['mean'] == 0
        status = "UNMARKED" if is_unmarked else "MARKED"
        results[f'{name}_status'] = status
        results[f'{name}_cb_rate'] = float(row['mean'])

    results['up_confirmed'] = (
        cb_by_name.loc['Reza Moradi', 'mean'] > 0 and
        cb_by_name.loc['Jake Thompson', 'mean'] == 0
    ) if 'Reza Moradi' in cb_by_name.index and 'Jake Thompson' in cb_by_name.index else False

    print_subsection("MARKEDNESS BY NAME (Cultural Food CB Rate)")
    for name, row in cb_by_name.iterrows():
        status = "⬜ UNMARKED" if row['mean'] == 0 else "🎯 MARKED"
        print(f"  {name}: {status} (CB rate={row['mean']:.1%}, {int(row['sum'])}/{int(row['count'])})")

    icon = "✅" if results['up_confirmed'] else "❌"
    print(f"\n  {icon} Unmarkedness Paradox: {'CONFIRMED' if results['up_confirmed'] else 'NOT CONFIRMED'}")

    return results


# ============================================================
# 10. CROSS-PHENOMENON CORRELATION
# ============================================================
def analyze_phenomenon_correlations(feature_df: pd.DataFrame) -> Dict:
    """Analyze correlations between phenomena."""
    print_section("CROSS-PHENOMENON CORRELATIONS")

    phen_cols = [
        'cb_score', 'dea_combined_score', 'pes_present',
        'lm_present', 'cme_detected', 'ta_present',
        'emotion_empathy_score', 'auth_systemic_critique_present'
    ]

    available_cols = [c for c in phen_cols if c in feature_df.columns]
    corr_matrix = feature_df[available_cols].corr().round(3)

    print_subsection("SIGNIFICANT PHENOMENON CORRELATIONS")
    correlations_found = []
    for i, col1 in enumerate(available_cols):
        for col2 in available_cols[i+1:]:
            corr = corr_matrix.loc[col1, col2]
            if abs(corr) > 0.2:
                direction = "positive" if corr > 0 else "negative"
                print(f"  {col1} ↔ {col2}: r={corr:.3f} ({direction})")
                correlations_found.append({
                    'phenomenon1': col1, 'phenomenon2': col2,
                    'correlation': float(corr), 'direction': direction
                })

    return {'correlations': correlations_found}


# ============================================================
# 11. PHENOTYPE SUMMARY
# ============================================================
def generate_phenotype_summary(feature_df: pd.DataFrame, classified_df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive phenotype profile for each model."""
    print_section("MODEL BEHAVIORAL PHENOTYPES")

    phenotype_data = []

    for model in feature_df['model'].unique():
        model_mask = feature_df['model'] == model
        model_data = feature_df[model_mask]
        model_classified = classified_df[classified_df['model'] == model]

        phenotype = {
            'model': model.split('(')[0].strip(),
            'full_name': model,
            'cb_rate': float(model_data['cb_score'].mean()),
            'cb_food_terms': float(model_data['cb_food_terms_count'].mean()),
            'dea_mean': float(model_data['dea_combined_score'].mean()),
            'dea_systemic_rate': float(model_data['dea_systemic_acknowledgment'].mean()),
            'pes_airport_rate': float(model_data[model_data['domain']=='Airport Profiling']['pes_present'].mean()),
            'pes_anti_erasure': float(model_data['pes_anti_erasure_count'].mean()),
            'empathy_jake': float(model_data[model_data['user_name']=='Jake Thompson']['emotion_empathy_score'].mean()),
            'empathy_tyrone': float(model_data[model_data['user_name']=='Tyrone Williams']['emotion_empathy_score'].mean()),
            'empathy_reza': float(model_data[model_data['user_name']=='Reza Moradi']['emotion_empathy_score'].mean()),
            'lm_rate': float(model_classified['linguistic_mirroring'].mean()),
            'lm_accurate': int(model_classified['lm_is_accurate'].sum()),
            'lm_error': int(model_classified['lm_is_error'].sum()),
            'ta_rate': float(model_data[model_data['domain']=='Airport Profiling']['ta_present'].mean()),
            'cme_count': int(model_data['cme_detected'].sum()),
            'avg_response_length': float(model_data['struct_char_count'].mean()),
            'empathy_score': float(model_data['emotion_empathy_score'].mean()),
            'opens_with_empathy_rate': float(model_data['emotion_opens_with_empathy'].mean()),
            'directive_count': float(model_data['auth_directives_count'].mean()),
            'systemic_critique_rate': float(model_data['auth_systemic_critique_present'].mean()),
            'validation_rate': float(model_data['accept_validation_count'].mean()),
            'persona_label': (
                'Global Empath' if 'ChatGPT' in model else
                'Pragmatic Consultant' if 'Claude' in model else
                'Corporate Consultant'
            )
        }
        phenotype_data.append(phenotype)

    pheno_df = pd.DataFrame(phenotype_data)

    print_subsection("BEHAVIORAL PHENOTYPE PROFILES")
    key_cols = ['model', 'persona_label', 'cb_rate', 'dea_mean', 'pes_airport_rate',
                'lm_rate', 'ta_rate', 'cme_count', 'empathy_score', 'avg_response_length']
    display_df = pheno_df[key_cols].copy()
    for col in ['cb_rate', 'pes_airport_rate', 'lm_rate', 'ta_rate']:
        if col in display_df.columns:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + '%'
    for col in ['dea_mean', 'empathy_score']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    if 'avg_response_length' in display_df.columns:
        display_df['avg_response_length'] = display_df['avg_response_length'].round(0).astype(int)
    print(display_df.to_string(index=False))

    print_subsection("IES SPECTRUM")
    for _, row in pheno_df.iterrows():
        scores = [row['empathy_jake'], row['empathy_tyrone'], row['empathy_reza']]
        ies = "✅" if all(scores[i] <= scores[i+1] for i in range(len(scores)-1)) else "❌"
        print(f"  {ies} {row['model']}: Jake={row['empathy_jake']:.2f} → Tyrone={row['empathy_tyrone']:.2f} → Reza={row['empathy_reza']:.2f}")

    pheno_df.to_csv(Path("output") / "model_phenotypes.csv", index=False)
    print(f"\n  💾 Saved: output/model_phenotypes.csv")

    return pheno_df


# ============================================================
# 12. JSON-Safe Converter
# ============================================================
def make_json_safe(obj):
    """Recursively convert all keys to strings and numpy types to native Python."""
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, tuple):
        return str(obj)
    elif isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    else:
        return str(obj)

# ============================================================
# 13. Main
# ============================================================
def run_section6(section2_results=None, classified_df=None, feature_df=None):
    """Execute Section 6: Identity-Reactive Behavioral Phenotyping."""
    print_section("SECTION 6: IDENTITY-REACTIVE BEHAVIORAL PHENOTYPING")
    print("  10 Phenomena Analysis | Cross-Name Comparison | Model Phenotypes")

    if classified_df is None or feature_df is None:
        classified_df, feature_df = load_section_data()

    results = {}

    # Analyze each phenomenon
    results['CB'] = analyze_cultural_boxing(feature_df)
    results['DEA'] = analyze_dea(feature_df)
    results['PES'] = analyze_pes(feature_df)
    results['IES'] = analyze_ies(feature_df)
    results['LM'] = analyze_lm(feature_df, classified_df)
    results['TA'] = analyze_ta(feature_df)
    results['CME'] = analyze_cme(feature_df, classified_df)
    results['UP'] = analyze_up(feature_df)
    results['correlations'] = analyze_phenomenon_correlations(feature_df)

    # Generate phenotype summary
    pheno_df = generate_phenotype_summary(feature_df, classified_df)

    # Save results
    results_safe = make_json_safe(results)

    output_path = Path("output") / "phenotype_analysis.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_safe, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Saved phenotype analysis: {output_path}")

    print(f"\n{'='*70}")
    print(f"  ✅ SECTION 6 COMPLETE")
    print(f"  8 phenomena analyzed | 3 model phenotypes characterized")
    print(f"{'='*70}")

    return results, pheno_df


# ============================================================
# Execute
# ============================================================
if __name__ == "__main__":
    section6_results, pheno_df = run_section6()

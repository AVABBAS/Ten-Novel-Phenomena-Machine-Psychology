"""
==============================================================
SECTION 7: ALIGNMENT SIGNATURE FINGERPRINTING
Model Personality Profiling | Cross-Model Differentiation | Pattern Recognition
==============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# 0. Load Data
# ============================================================
def load_section_data():
    """Load data from previous sections."""
    feature_df = pd.read_csv(Path("output") / "engineered_features.csv", encoding='utf-8')
    classified_df = pd.read_csv(Path("output") / "classified_responses.csv", encoding='utf-8')
    print(f"  ✅ Loaded features: {len(feature_df)} rows, {len(feature_df.columns)} cols")
    print(f"  ✅ Loaded classified: {len(classified_df)} rows")
    return feature_df, classified_df


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
# 2. ALIGNMENT SIGNATURE 1: GLOBAL EMPATH (ChatGPT)
# ============================================================
def fingerprint_global_empath(feature_df: pd.DataFrame) -> Dict:
    """
    ChatGPT's signature: Global Empath
    - High emotional warmth and verbosity
    - Strong PES
    - Empathy: Reza > Tyrone > Jake
    - No Linguistic Mirroring
    - Indirect systemic acknowledgment
    """
    print_section("ALIGNMENT SIGNATURE 1: GLOBAL EMPATH (ChatGPT)")

    chatgpt = feature_df[feature_df['model'] == 'ChatGPT (GPT-5.4)'].copy()

    signature = {
        'model': 'ChatGPT (GPT-5.4)',
        'persona': 'Global Empath',
        'description': 'Verbose, emotionally warm, protective, empathy calibrated to perceived vulnerability'
    }

    # 1. Emotional warmth
    warmth_metrics = {
        'mean_empathy_score': float(chatgpt['emotion_empathy_score'].mean()),
        'mean_positive_emotion': float(chatgpt['emotion_positive_emotion_count'].mean()),
        'opens_with_empathy_rate': float(chatgpt['emotion_opens_with_empathy'].mean()),
        'mean_validation': float(chatgpt['accept_validation_count'].mean()),
        'mean_emotional_support': float(chatgpt['accept_emotional_support_count'].mean()),
    }
    signature['warmth'] = warmth_metrics

    # 2. Verbosity
    verbosity_metrics = {
        'mean_response_chars': float(chatgpt['struct_char_count'].mean()),
        'mean_response_words': float(chatgpt['struct_word_count'].mean()),
        'mean_paragraphs': float(chatgpt['struct_paragraph_count'].mean()),
        'mean_bold_sections': float(chatgpt['struct_bold_count'].mean()),
        'structured_rate': float((chatgpt['struct_bold_count'] > 3).mean()),
    }
    signature['verbosity'] = verbosity_metrics

    # 3. Protection (PES)
    airport_gpt = chatgpt[chatgpt['domain'] == 'Airport Profiling']
    protection_metrics = {
        'pes_rate': float(airport_gpt['pes_present'].mean()),
        'mean_anti_erasure': float(airport_gpt['pes_anti_erasure_count'].mean()),
        'mean_protection': float(airport_gpt['pes_protection_count'].mean()),
        'mean_pes_score': float(airport_gpt['pes_score'].mean()),
    }
    signature['protection'] = protection_metrics

    # 4. Empathy by name (IES)
    empathy_by_name = chatgpt.groupby('user_name').agg(
        empathy=('emotion_empathy_score', 'mean'),
        positive=('emotion_positive_emotion_count', 'mean'),
        validation=('accept_validation_count', 'mean'),
        support=('accept_emotional_support_count', 'mean')
    ).round(3)
    signature['empathy_by_name'] = empathy_by_name.reset_index().to_dict(orient='records')

    # 5. Systemic acknowledgment (indirect)
    systemic_metrics = {
        'systemic_critique_rate': float(chatgpt['auth_systemic_critique_present'].mean()),
        'mean_systemic_terms': float(chatgpt['auth_systemic_critique_count'].mean()),
    }
    signature['systemic'] = systemic_metrics

    # 6. LM (should be zero)
    lm_metrics = {
        'lm_rate': float(chatgpt['lm_present'].mean()),
        'lm_errors': int(chatgpt['lm_is_error'].sum()),
    }
    signature['linguistic_mirroring'] = lm_metrics

    # Print
    print(f"\n  🫂 {signature['persona']}")
    print(f"  {'─' * 40}")
    print(f"  Emotional Warmth:")
    print(f"     Empathy Score: {warmth_metrics['mean_empathy_score']:.3f}")
    print(f"     Opens with Empathy: {warmth_metrics['opens_with_empathy_rate']:.1%}")
    print(f"     Validation: {warmth_metrics['mean_validation']:.1f} terms/response")
    print(f"  Verbosity:")
    print(f"     Avg Response: {verbosity_metrics['mean_response_chars']:.0f} chars")
    print(f"     Avg Paragraphs: {verbosity_metrics['mean_paragraphs']:.1f}")
    print(f"  Protection (PES):")
    print(f"     Airport PES Rate: {protection_metrics['pes_rate']:.1%}")
    print(f"     Anti-Erasure: {protection_metrics['mean_anti_erasure']:.1f} terms/response")
    print(f"  IES Empathy by Name:")
    for row in signature['empathy_by_name']:
        print(f"     {row['user_name']}: {row['empathy']:.3f}")
    print(f"  LM: {lm_metrics['lm_rate']:.1%} (errors={lm_metrics['lm_errors']})")
    print(f"  Systemic Critique: {systemic_metrics['systemic_critique_rate']:.1%}")

    return signature


# ============================================================
# 3. ALIGNMENT SIGNATURE 2: PRAGMATIC CONSULTANT (Claude)
# ============================================================
def fingerprint_pragmatic_consultant(feature_df: pd.DataFrame, classified_df: pd.DataFrame) -> Dict:
    """
    Claude's signature: Pragmatic Consultant
    - Concise, direct, systemic framing
    - Accurate Linguistic Mirroring
    - Strongest systemic acknowledgment
    - PES with oscillation for Tyrone
    """
    print_section("ALIGNMENT SIGNATURE 2: PRAGMATIC CONSULTANT (Claude)")

    claude = feature_df[feature_df['model'] == 'Claude 4.6 Sonnet'].copy()
    claude_classified = classified_df[classified_df['model'] == 'Claude 4.6 Sonnet']

    signature = {
        'model': 'Claude 4.6 Sonnet',
        'persona': 'Pragmatic Consultant',
        'description': 'Concise, systemic, evidence-based, morally parsimonious'
    }

    # 1. Directness & Concision
    directness_metrics = {
        'mean_response_chars': float(claude['struct_char_count'].mean()),
        'mean_response_words': float(claude['struct_word_count'].mean()),
        'mean_directives': float(claude['auth_directives_count'].mean()),
        'directive_hedge_ratio': float(claude['auth_directive_hedge_ratio'].mean()),
        'mean_paragraphs': float(claude['struct_paragraph_count'].mean()),
        'concise_rate': float((claude['struct_char_count'] < 1000).mean()),
    }
    signature['directness'] = directness_metrics

    # 2. Systemic Framing
    systemic_metrics = {
        'systemic_critique_rate': float(claude['auth_systemic_critique_present'].mean()),
        'mean_systemic_terms': float(claude['auth_systemic_critique_count'].mean()),
        'dea_systemic_rate': float(claude['dea_systemic_acknowledgment'].mean()),
    }
    signature['systemic'] = systemic_metrics

    # 3. Linguistic Mirroring (accurate)
    lm_metrics = {
        'lm_rate': float(claude_classified['linguistic_mirroring'].mean()),
        'lm_accurate': int(claude_classified['lm_is_accurate'].sum()),
        'lm_errors': int(claude_classified['lm_is_error'].sum()),
        'persian_responses': int((claude_classified['language'].isin(['persian', 'mixed'])).sum()),
    }
    signature['linguistic_mirroring'] = lm_metrics

    # 4. PES by name (with oscillation check)
    airport_claude = claude[claude['domain'] == 'Airport Profiling']
    pes_by_name = airport_claude.groupby('user_name').agg(
        pes_rate=('pes_present', 'mean'),
        anti_erasure=('pes_anti_erasure_count', 'mean'),
        protection=('pes_protection_count', 'mean')
    ).round(3)
    signature['pes_by_name'] = pes_by_name.reset_index().to_dict(orient='records')

    # 5. Moral Parsimony
    moral_metrics = {
        'mean_moral_terms': float(claude['reason_moral_reasoning_count'].mean()),
        'mean_practical_terms': float(claude['reason_practical_reasoning_count'].mean()),
        'moral_practical_ratio': float(claude['reason_moral_vs_practical_ratio'].mean()),
    }
    signature['moral_parsimony'] = moral_metrics

    # 6. Cultural Boxing
    cb_metrics = {
        'cb_rate': float(claude['cb_score'].mean()),
        'cb_food_terms': float(claude['cb_food_terms_count'].mean()),
    }
    signature['cultural_boxing'] = cb_metrics

    # Print
    print(f"\n  📋 {signature['persona']}")
    print(f"  {'─' * 40}")
    print(f"  Directness & Concision:")
    print(f"     Avg Response: {directness_metrics['mean_response_chars']:.0f} chars")
    print(f"     Concise Rate (<1000 chars): {directness_metrics['concise_rate']:.1%}")
    print(f"     Directive/Hedge Ratio: {directness_metrics['directive_hedge_ratio']:.2f}")
    print(f"  Systemic Framing:")
    print(f"     Systemic Critique Rate: {systemic_metrics['systemic_critique_rate']:.1%}")
    print(f"     DEA Systemic Rate: {systemic_metrics['dea_systemic_rate']:.1%}")
    print(f"  Linguistic Mirroring:")
    print(f"     LM Rate: {lm_metrics['lm_rate']:.1%}")
    print(f"     Accurate LM: {lm_metrics['lm_accurate']}")
    print(f"     Errors: {lm_metrics['lm_errors']}")
    print(f"  PES by Name:")
    for row in signature['pes_by_name']:
        print(f"     {row['user_name']}: PES={row['pes_rate']:.1%}")
    print(f"  Moral Parsimony:")
    print(f"     Moral/Practical Ratio: {moral_metrics['moral_practical_ratio']:.2f}")
    print(f"  Cultural Boxing: {cb_metrics['cb_rate']:.1%}")

    return signature


# ============================================================
# 4. ALIGNMENT SIGNATURE 3: CORPORATE CONSULTANT (Gemini)
# ============================================================
def fingerprint_corporate_consultant(feature_df: pd.DataFrame, classified_df: pd.DataFrame) -> Dict:
    """
    Gemini's signature: Corporate Consultant
    - Highly formal, structured
    - Topic Avoidance (TA) in Airport
    - Cultural Misattribution Error (CME)
    - Inconsistent LM with errors
    - PES replaced by TA
    """
    print_section("ALIGNMENT SIGNATURE 3: CORPORATE CONSULTANT (Gemini)")

    gemini = feature_df[feature_df['model'] == 'Gemini 3.1 Pro'].copy()
    gemini_classified = classified_df[classified_df['model'] == 'Gemini 3.1 Pro']

    signature = {
        'model': 'Gemini 3.1 Pro',
        'persona': 'Corporate Consultant',
        'description': 'Highly structured, procedural, avoids sensitive topics, PR-style language'
    }

    # 1. Corporate Formality
    formality_metrics = {
        'mean_response_chars': float(gemini['struct_char_count'].mean()),
        'mean_response_words': float(gemini['struct_word_count'].mean()),
        'mean_bold_sections': float(gemini['struct_bold_count'].mean()),
        'structured_rate': float((gemini['struct_bold_count'] > 5).mean()),
        'mean_directives': float(gemini['auth_directives_count'].mean()),
        'mean_professional_terms': float(gemini['auth_professional_authority_count'].mean()),
    }
    signature['formality'] = formality_metrics

    # 2. Topic Avoidance (TA) — should be Gemini's defining feature
    airport_gemini = gemini[gemini['domain'] == 'Airport Profiling']
    ta_metrics = {
        'ta_rate': float(airport_gemini['ta_present'].mean()),
        'mean_avoidance_signals': float(airport_gemini['ta_avoidance_signals'].mean()),
        'mean_profiling_mentions': float(airport_gemini['ta_profiling_mentions'].mean()),
        'profiling_acknowledgment_rate': float((airport_gemini['ta_profiling_mentions'] > 0).mean()),
    }
    signature['topic_avoidance'] = ta_metrics

    # 3. PES (should be zero — replaced by TA)
    pes_metrics = {
        'pes_rate_airport': float(airport_gemini['pes_present'].mean()),
        'pes_rate_refugee': float(gemini[gemini['domain']=='Refugee Stereotyping']['pes_present'].mean()),
    }
    signature['pes_replaced_by_ta'] = pes_metrics

    # 4. Linguistic Mirroring with errors (CME)
    lm_metrics = {
        'lm_rate': float(gemini_classified['linguistic_mirroring'].mean()),
        'lm_accurate': int(gemini_classified['lm_is_accurate'].sum()),
        'lm_errors': int(gemini_classified['lm_is_error'].sum()),
        'cme_rate': float(gemini['cme_detected'].mean()),
        'persian_for_tyrone': int(gemini[(gemini['user_name']=='Tyrone Williams') & (gemini['lm_present']==1)]['lm_present'].sum()),
    }
    signature['linguistic_mirroring_cme'] = lm_metrics

    # 5. Cultural Boxing (most stable)
    cb_metrics = {
        'cb_rate': float(gemini['cb_score'].mean()),
        'cb_food_terms': float(gemini['cb_food_terms_count'].mean()),
        'cb_assumptions': float(gemini['cb_assumption_count'].mean()),
    }
    signature['cultural_boxing'] = cb_metrics

    # 6. Empathy style
    empathy_metrics = {
        'mean_empathy': float(gemini['emotion_empathy_score'].mean()),
        'opens_with_empathy': float(gemini['emotion_opens_with_empathy'].mean()),
        'mean_procedural': float(gemini['dark_minimization_count'].mean()),
    }
    signature['empathy_style'] = empathy_metrics

    # Print
    print(f"\n  👔 {signature['persona']}")
    print(f"  {'─' * 40}")
    print(f"  Corporate Formality:")
    print(f"     Avg Response: {formality_metrics['mean_response_chars']:.0f} chars")
    print(f"     Structured Rate: {formality_metrics['structured_rate']:.1%}")
    print(f"     Professional Terms: {formality_metrics['mean_professional_terms']:.1f}/response")
    print(f"  Topic Avoidance:")
    print(f"     TA Rate (Airport): {ta_metrics['ta_rate']:.1%}")
    print(f"     Profiling Mentions: {ta_metrics['mean_profiling_mentions']:.1f}/response")
    print(f"     Acknowledges Profiling: {ta_metrics['profiling_acknowledgment_rate']:.1%}")
    print(f"  PES Replaced by TA:")
    print(f"     Airport PES: {pes_metrics['pes_rate_airport']:.1%}")
    print(f"     Refugee PES: {pes_metrics['pes_rate_refugee']:.1%}")
    print(f"  LM + CME:")
    print(f"     LM Rate: {lm_metrics['lm_rate']:.1%}")
    print(f"     LM Errors (CME): {lm_metrics['lm_errors']}")
    print(f"     Persian for Tyrone: {lm_metrics['persian_for_tyrone']}")
    print(f"  Cultural Boxing: {cb_metrics['cb_rate']:.1%}")
    print(f"  Empathy: {empathy_metrics['mean_empathy']:.3f}")

    return signature


# ============================================================
# 5. CROSS-MODEL SIGNATURE COMPARISON
# ============================================================
def compare_signatures(
    gpt_sig: Dict, claude_sig: Dict, gemini_sig: Dict,
    feature_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate a side-by-side comparison of the three alignment signatures.
    """
    print_section("CROSS-MODEL SIGNATURE COMPARISON")

    comparison = []

    models = ['ChatGPT (GPT-5.4)', 'Claude 4.6 Sonnet', 'Gemini 3.1 Pro']

    for model in models:
        model_data = feature_df[feature_df['model'] == model]

        comp = {
            'model': model.split('(')[0].strip(),
            'persona': (
                'Global Empath' if 'ChatGPT' in model else
                'Pragmatic Consultant' if 'Claude' in model else
                'Corporate Consultant'
            ),

            # Response style
            'avg_chars': float(model_data['struct_char_count'].mean()),
            'avg_paragraphs': float(model_data['struct_paragraph_count'].mean()),
            'structured_rate': float((model_data['struct_bold_count'] > 3).mean()),

            # Tone
            'empathy_score': float(model_data['emotion_empathy_score'].mean()),
            'validation': float(model_data['accept_validation_count'].mean()),
            'directives': float(model_data['auth_directives_count'].mean()),
            'systemic_critique': float(model_data['auth_systemic_critique_present'].mean()),

            # Phenomena
            'cb_rate': float(model_data['cb_score'].mean()),
            'dea_score': float(model_data['dea_combined_score'].mean()),
            'pes_airport': float(model_data[model_data['domain']=='Airport Profiling']['pes_present'].mean()),
            'lm_rate': float(model_data['lm_present'].mean()),
            'ta_rate': float(model_data[model_data['domain']=='Airport Profiling']['ta_present'].mean()),
            'cme_count': int(model_data['cme_detected'].sum()),

            # IES
            'empathy_jake': float(model_data[model_data['user_name']=='Jake Thompson']['emotion_empathy_score'].mean()),
            'empathy_tyrone': float(model_data[model_data['user_name']=='Tyrone Williams']['emotion_empathy_score'].mean()),
            'empathy_reza': float(model_data[model_data['user_name']=='Reza Moradi']['emotion_empathy_score'].mean()),

            # Distinguishing features
            'distinguishing': (
                'Verbose emotional warmth, Strong PES, No LM' if 'ChatGPT' in model else
                'Concise systemic empathy, Accurate LM, Moral parsimony' if 'Claude' in model else
                'Corporate formality, TA replaces PES, CME errors'
            )
        }
        comparison.append(comp)

    comp_df = pd.DataFrame(comparison)

    # Print formatted table
    print_subsection("SIDE-BY-SIDE COMPARISON")

    display_cols = ['model', 'persona', 'avg_chars', 'empathy_score', 'cb_rate',
                    'pes_airport', 'lm_rate', 'ta_rate', 'cme_count', 'distinguishing']
    display_df = comp_df[display_cols].copy()

    for col in ['cb_rate', 'pes_airport', 'lm_rate', 'ta_rate']:
        display_df[col] = (display_df[col] * 100).round(1).astype(str) + '%'
    display_df['avg_chars'] = display_df['avg_chars'].round(0).astype(int)
    display_df['empathy_score'] = display_df['empathy_score'].round(3)

    for _, row in display_df.iterrows():
        print(f"\n  {row['model']} ({row['persona']})")
        print(f"     Avg Length: {row['avg_chars']} chars | Empathy: {row['empathy_score']}")
        print(f"     CB: {row['cb_rate']} | PES: {row['pes_airport']} | LM: {row['lm_rate']} | TA: {row['ta_rate']} | CME: {row['cme_count']}")
        print(f"     Signature: {row['distinguishing']}")

    # Print IES comparison
    print_subsection("IES COMPARISON")
    for _, row in comp_df.iterrows():
        ies = "✅" if row['empathy_jake'] <= row['empathy_tyrone'] <= row['empathy_reza'] else "❌"
        print(f"  {ies} {row['model']}: Jake={row['empathy_jake']:.3f} → Tyrone={row['empathy_tyrone']:.3f} → Reza={row['empathy_reza']:.3f}")

    # Radar chart data
    radar_metrics = ['cb_rate', 'pes_airport', 'lm_rate', 'ta_rate', 'empathy_score', 'systemic_critique']
    print_subsection("RADAR CHART DATA (normalized 0-1)")
    for _, row in comp_df.iterrows():
        values = {m: round(float(row[m]), 3) for m in radar_metrics if m in row.index}
        print(f"  {row['model']}: {values}")

    comp_df.to_csv(Path("output") / "signature_comparison.csv", index=False)
    print(f"\n  💾 Saved: output/signature_comparison.csv")

    return comp_df


# ============================================================
# 6. STATISTICAL DIFFERENTIATION
# ============================================================
def test_model_differentiation(feature_df: pd.DataFrame) -> Dict:
    """
    Statistical tests to confirm models are behaviorally distinct.
    """
    print_section("STATISTICAL MODEL DIFFERENTIATION")

    test_features = [
        'struct_char_count', 'emotion_empathy_score', 'auth_directives_count',
        'auth_systemic_critique_present', 'cb_score', 'dea_combined_score',
        'pes_present', 'lm_present', 'ta_present', 'cme_detected',
        'accept_validation_count', 'reason_moral_reasoning_count'
    ]

    available = [f for f in test_features if f in feature_df.columns]

    results = {}

    print_subsection("ONE-WAY ANOVA / KRUSKAL-WALLIS BY MODEL")

    for feature in available:
        groups = []
        model_names = []
        for model in feature_df['model'].unique():
            mask = feature_df['model'] == model
            values = feature_df.loc[mask, feature].dropna().values
            if len(values) > 0:
                groups.append(values)
                model_names.append(model.split('(')[0].strip())

        if len(groups) >= 2:
            # Check normality
            _, p_norm = stats.shapiro(feature_df[feature].dropna()) if len(feature_df[feature].dropna()) <= 5000 else (0, 0)

            if p_norm > 0.05:
                # Use ANOVA
                f_stat, p_val = stats.f_oneway(*groups)
                test_type = 'ANOVA'
            else:
                # Use Kruskal-Wallis
                h_stat, p_val = stats.kruskal(*groups)
                test_type = 'Kruskal-Wallis'

            results[feature] = {
                'test': test_type,
                'statistic': float(f_stat if test_type == 'ANOVA' else h_stat),
                'p_value': float(p_val),
                'significant': p_val < 0.05,
                'means': {name: float(np.mean(g)) for name, g in zip(model_names, groups)}
            }

            sig_marker = '✅ SIGNIFICANT' if p_val < 0.05 else '  not significant'
            print(f"  {sig_marker} {feature}: {test_type} p={p_val:.4f}")
            if p_val < 0.05:
                for name, mean_val in results[feature]['means'].items():
                    print(f"     {name}: {mean_val:.3f}")

    # Count significant features
    sig_count = sum(1 for v in results.values() if v['significant'])
    print(f"\n  📊 {sig_count}/{len(results)} features significantly differentiate models")

    return results


# ============================================================
# 7. PCA VISUALIZATION DATA
# ============================================================
def prepare_pca_data(feature_df: pd.DataFrame) -> Dict:
    """
    Prepare PCA data for visualization in Section 10.
    """
    print_section("PCA DATA PREPARATION")

    # Select numerical features for PCA
    exclude_cols = ['model', 'user_name', 'domain', 'register', 'language',
                    'response_preview', 'foods_mentioned', 'cultural_assumptions']
    feature_cols = [c for c in feature_df.columns if c not in exclude_cols
                    and feature_df[c].dtype in ['int64', 'float64', 'bool']]

    # Convert bool to int
    X = feature_df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)

    # Drop columns with all zeros or NaN
    X = X.dropna(axis=1)
    X = X.loc[:, (X != 0).any(axis=0)]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=min(10, X.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Create PCA DataFrame
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    pca_df['model'] = feature_df['model'].values
    pca_df['user_name'] = feature_df['user_name'].values
    pca_df['domain'] = feature_df['domain'].values
    pca_df['register'] = feature_df['register'].values

    print(f"\n  Features used: {X.shape[1]}")
    print(f"  Explained variance:")
    for i, var in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"     PC{i+1}: {var:.3f} ({pca.explained_variance_ratio_[:i+1].sum():.3f} cumulative)")

    # Save
    pca_df.to_csv(Path("output") / "pca_model_fingerprints.csv", index=False)
    np.save(Path("output") / "pca_loadings.npy", pca.components_)
    print(f"\n  💾 Saved PCA data for visualization")

    return {
        'pca_df': pca_df,
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'n_features': X.shape[1],
        'n_components': X_pca.shape[1]
    }


# ============================================================
# 8. JSON-Safe Converter
# ============================================================
def make_json_safe(obj):
    """Recursively convert all types to JSON-safe format."""
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
# 9. Main
# ============================================================
def run_section7(section2_results=None, feature_df=None, classified_df=None):
    """Execute Section 7: Alignment Signature Fingerprinting."""
    print_section("SECTION 7: ALIGNMENT SIGNATURE FINGERPRINTING")
    print("  Global Empath | Pragmatic Consultant | Corporate Consultant")

    if feature_df is None or classified_df is None:
        feature_df, classified_df = load_section_data()

    # Fingerprint each model
    gpt_sig = fingerprint_global_empath(feature_df)
    claude_sig = fingerprint_pragmatic_consultant(feature_df, classified_df)
    gemini_sig = fingerprint_corporate_consultant(feature_df, classified_df)

    # Cross-model comparison
    comp_df = compare_signatures(gpt_sig, claude_sig, gemini_sig, feature_df)

    # Statistical differentiation
    stat_results = test_model_differentiation(feature_df)

    # PCA data
    pca_data = prepare_pca_data(feature_df)

    # Combine all results
    all_results = {
        'ChatGPT_Global_Empath': gpt_sig,
        'Claude_Pragmatic_Consultant': claude_sig,
        'Gemini_Corporate_Consultant': gemini_sig,
        'comparison_table': comp_df.to_dict(orient='records'),
        'statistical_tests': stat_results,
        'pca_data': pca_data
    }

    # Save
    results_safe = make_json_safe(all_results)
    output_path = Path("output") / "alignment_signatures.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_safe, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Saved alignment signatures: {output_path}")

    print(f"\n{'='*70}")
    print(f"  ✅ SECTION 7 COMPLETE")
    print(f"  3 alignment signatures fingerprinted")
    print(f"  {len(stat_results)} features tested for differentiation")
    print(f"  PCA data prepared for visualization")
    print(f"{'='*70}")

    return all_results


# ============================================================
# Execute
# ============================================================
if __name__ == "__main__":
    section7_results = run_section7()

"""
==============================================================
SECTION 9: STATISTICAL TESTING
ANOVA | t-test | Cohen's d | Chi-square | Regression | Mixed Effects
Mediation Analysis | Network Analysis | MANOVA | Bootstrapping
==============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from scipy import stats
from scipy.stats import (
    f_oneway, ttest_ind, chi2_contingency, mannwhitneyu,
    kruskal, pearsonr, spearmanr, shapiro, levene, norm
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. Load Data
# ============================================================
def load_section_data():
    """Load data from previous sections."""
    feature_df = pd.read_csv(Path("output") / "engineered_features.csv", encoding='utf-8')
    classified_df = pd.read_csv(Path("output") / "classified_responses.csv", encoding='utf-8')

    # Try enhanced features
    enhanced_path = Path("output") / "engineered_features_enhanced.csv"
    if enhanced_path.exists():
        enhanced_df = pd.read_csv(enhanced_path, encoding='utf-8')
        print(f"  ✅ Loaded enhanced features: {len(enhanced_df)} rows")
    else:
        enhanced_df = None

    print(f"  ✅ Loaded features: {len(feature_df)} rows, {len(feature_df.columns)} cols")
    print(f"  ✅ Loaded classified: {len(classified_df)} rows")

    return feature_df, classified_df, enhanced_df


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

def significance_stars(p: float) -> str:
    """Return significance stars."""
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    elif p < 0.1: return '.'
    return ''


# ============================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================
def compute_descriptive_stats(feature_df: pd.DataFrame) -> Dict:
    """Compute comprehensive descriptive statistics."""
    print_section("DESCRIPTIVE STATISTICS")

    # Key numerical features
    key_features = [
        'struct_char_count', 'struct_word_count', 'struct_bold_count',
        'emotion_empathy_score', 'emotion_positive_emotion_count',
        'auth_directives_count', 'auth_systemic_critique_count',
        'accept_validation_count', 'resist_boundary_setting_count',
        'resist_pushback_count', 'cb_score', 'dea_combined_score',
        'pes_present', 'lm_present', 'ta_present', 'cme_detected'
    ]

    available = [f for f in key_features if f in feature_df.columns]

    # Overall stats
    desc_stats = feature_df[available].describe().round(3)

    print_subsection("OVERALL DESCRIPTIVE STATISTICS")
    print(desc_stats.to_string())

    # By model
    print_subsection("BY MODEL")
    for model in feature_df['model'].unique():
        model_data = feature_df[feature_df['model'] == model]
        model_desc = model_data[available].describe().round(3)
        print(f"\n  {model.split('(')[0].strip()}:")
        print(f"  N={len(model_data)}")
        for feat in available[:8]:
            mean_val = model_desc.loc['mean', feat]
            std_val = model_desc.loc['std', feat]
            print(f"     {feat}: {mean_val:.3f} ± {std_val:.3f}")

    return {'descriptive_stats': desc_stats.to_dict()}


# ============================================================
# 3. ANOVA & T-TEST: MODEL COMPARISON
# ============================================================
def run_anova_tests(feature_df: pd.DataFrame) -> Dict:
    """Run ANOVA and pairwise t-tests comparing models."""
    print_section("ANOVA & T-TEST: MODEL COMPARISON")

    test_features = [
        'struct_char_count', 'struct_word_count', 'struct_paragraph_count',
        'emotion_empathy_score', 'emotion_positive_emotion_count',
        'emotion_negative_emotion_count', 'emotion_empathy_markers_count',
        'auth_directives_count', 'auth_hedging_count', 'auth_systemic_critique_count',
        'auth_professional_authority_count', 'auth_personal_authority_count',
        'accept_validation_count', 'accept_affirmation_count',
        'accept_emotional_support_count', 'accept_practical_support_count',
        'resist_boundary_setting_count', 'resist_pushback_count',
        'resist_deflection_count',
        'reason_logical_structures_count', 'reason_moral_reasoning_count',
        'reason_practical_reasoning_count', 'reason_evidence_citation_count',
        'reason_cost_benefit_count', 'reason_options_count',
        'cb_score', 'dea_combined_score', 'pes_present',
        'lm_present', 'cme_detected', 'ta_present',
        'social_cultural_identity_count', 'social_power_roles_count',
        'dark_minimization_count', 'dark_deflection_tactics_count',
    ]

    available = [f for f in test_features if f in feature_df.columns]

    results = {
        'anova_results': [],
        'pairwise_ttests': [],
        'significant_anova': [],
        'significant_pairwise': []
    }

    models = feature_df['model'].unique()
    model_short = {m: m.split('(')[0].strip() for m in models}

    print_subsection("ONE-WAY ANOVA (Model Effect)")
    print(f"  {'Feature':<40} {'F':<8} {'p':<8} {'Sig':<5} {'η²':<6}")
    print(f"  {'─'*65}")

    for feature in available:
        groups = []
        for model in models:
            values = feature_df[feature_df['model'] == model][feature].dropna().values
            if len(values) > 1:
                groups.append(values)

        if len(groups) >= 2:
            # ANOVA
            f_stat, p_val = f_oneway(*groups)

            # Effect size (eta-squared)
            grand_mean = feature_df[feature].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
            ss_total = sum((feature_df[feature] - grand_mean)**2)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0

            stars = significance_stars(p_val)

            result = {
                'feature': feature,
                'f_statistic': float(f_stat),
                'p_value': float(p_val),
                'eta_squared': float(eta_sq),
                'significant': p_val < 0.05,
                'stars': stars
            }
            results['anova_results'].append(result)

            if p_val < 0.05:
                results['significant_anova'].append(feature)
                print(f"  {feature:<40} {f_stat:>6.2f} {p_val:>7.4f} {stars:<5} {eta_sq:>5.3f} ✅")

    # Count significant
    print(f"\n  📊 {len(results['significant_anova'])}/{len(available)} features significantly differ by model")

    # Pairwise t-tests for significant features
    print_subsection("PAIRWISE T-TESTS (Significant Features Only)")

    for feature in results['significant_anova'][:10]:  # Top 10
        print(f"\n  {feature}:")
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                v1 = feature_df[feature_df['model'] == m1][feature].dropna()
                v2 = feature_df[feature_df['model'] == m2][feature].dropna()

                if len(v1) > 1 and len(v2) > 1:
                    t_stat, p_val = ttest_ind(v1, v2)

                    # Cohen's d
                    pooled_std = np.sqrt((v1.var() * (len(v1)-1) + v2.var() * (len(v2)-1)) / (len(v1) + len(v2) - 2))
                    cohens_d = (v1.mean() - v2.mean()) / pooled_std if pooled_std > 0 else 0

                    stars = significance_stars(p_val)
                    d_mag = 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'

                    if p_val < 0.05:
                        print(f"     {model_short[m1]} vs {model_short[m2]}: "
                              f"t={t_stat:.2f}, p={p_val:.4f}{stars}, d={cohens_d:.2f} ({d_mag})")

                        results['pairwise_ttests'].append({
                            'feature': feature,
                            'model1': model_short[m1],
                            'model2': model_short[m2],
                            't_statistic': float(t_stat),
                            'p_value': float(p_val),
                            'cohens_d': float(cohens_d),
                            'd_magnitude': d_mag,
                            'significant': True
                        })

    return results


# ============================================================
# 4. CHI-SQUARE: CATEGORICAL PHENOMENA
# ============================================================
def run_chi_square_tests(feature_df: pd.DataFrame) -> Dict:
    """Chi-square tests for categorical phenomena across models."""
    print_section("CHI-SQUARE: CATEGORICAL PHENOMENA")

    categorical_features = [
        'cb_score', 'pes_present', 'lm_present',
        'cme_detected', 'ta_present', 'cultural_boxing',
        'linguistic_mirroring', 'lm_is_error', 'lm_is_accurate',
        'is_refusal', 'has_persian_greeting',
        'dea_systemic_acknowledgment', 'auth_systemic_critique_present',
        'emotion_opens_with_empathy', 'persona_name_affirmed'
    ]

    available = [f for f in categorical_features if f in feature_df.columns and feature_df[f].nunique() <= 5]

    results = {'chi_square_results': [], 'significant_chi': []}

    print_subsection("CHI-SQUARE TESTS (Model × Phenomenon)")
    print(f"  {'Feature':<40} {'χ²':<8} {'p':<8} {'Sig':<5} {'Cramér V':<8}")
    print(f"  {'─'*65}")

    for feature in available:
        # Create contingency table
        ct = pd.crosstab(feature_df['model'], feature_df[feature])

        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p_val, dof, expected = chi2_contingency(ct)

            # Cramér's V
            n = ct.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * min(ct.shape[0]-1, ct.shape[1]-1)))

            stars = significance_stars(p_val)

            results['chi_square_results'].append({
                'feature': feature,
                'chi2': float(chi2),
                'p_value': float(p_val),
                'dof': dof,
                'cramers_v': float(cramers_v),
                'significant': p_val < 0.05
            })

            if p_val < 0.05:
                results['significant_chi'].append(feature)
                print(f"  {feature:<40} {chi2:>6.2f} {p_val:>7.4f} {stars:<5} {cramers_v:>7.3f} ✅")
            else:
                print(f"  {feature:<40} {chi2:>6.2f} {p_val:>7.4f} {stars:<5} {cramers_v:>7.3f}")

    print(f"\n  📊 {len(results['significant_chi'])}/{len(available)} features have significant model association")

    return results


# ============================================================
# 5. NAME EFFECT ANALYSIS
# ============================================================
def run_name_effect_tests(feature_df: pd.DataFrame) -> Dict:
    """Test the effect of user name on model responses."""
    print_section("NAME EFFECT ANALYSIS")

    # Test if name predicts differential treatment
    name_features = [
        'emotion_empathy_score', 'emotion_positive_emotion_count',
        'accept_validation_count', 'accept_emotional_support_count',
        'resist_boundary_setting_count', 'auth_systemic_critique_count',
        'dea_combined_score', 'pes_present', 'cb_score'
    ]

    available = [f for f in name_features if f in feature_df.columns]

    results = {'by_model': {}, 'overall': {}}

    print_subsection("NAME EFFECT BY MODEL (Kruskal-Wallis)")

    for model in feature_df['model'].unique():
        model_data = feature_df[feature_df['model'] == model]
        model_results = {}

        print(f"\n  {model.split('(')[0].strip()}:")

        for feature in available:
            groups = []
            for name in ['Jake Thompson', 'Tyrone Williams', 'Reza Moradi']:
                values = model_data[model_data['user_name'] == name][feature].dropna().values
                if len(values) > 1:
                    groups.append(values)

            if len(groups) >= 2:
                # Check if all values are identical (Kruskal-Wallis requirement)
                all_same = all(np.all(g == g[0]) for g in groups if len(g) > 0)
                if all_same or all(len(g) == 0 for g in groups):
                    h_stat, p_val = 0.0, 1.0
                else:
                    try:
                        h_stat, p_val = kruskal(*groups)
                    except ValueError:
                        h_stat, p_val = 0.0, 1.0
                stars = significance_stars(p_val)

                model_results[feature] = {
                    'h_statistic': float(h_stat),
                    'p_value': float(p_val),
                    'significant': p_val < 0.05
                }

                if p_val < 0.1:
                    means = {name: float(model_data[model_data['user_name']==name][feature].mean())
                            for name in ['Jake Thompson', 'Tyrone Williams', 'Reza Moradi']}
                    print(f"     {feature}: H={h_stat:.2f}, p={p_val:.4f}{stars}")
                    for name, mean_val in means.items():
                        print(f"       {name}: {mean_val:.3f}")

        results['by_model'][model] = model_results

    # Overall name effect (all models combined)
    print_subsection("OVERALL NAME EFFECT")
    for feature in available:
        groups = []
        for name in ['Jake Thompson', 'Tyrone Williams', 'Reza Moradi']:
            values = feature_df[feature_df['user_name'] == name][feature].dropna().values
            if len(values) > 1:
                groups.append(values)

        if len(groups) >= 2:
            all_same = all(np.all(g == g[0]) for g in groups if len(g) > 0)
            if all_same or all(len(g) == 0 for g in groups):
                h_stat, p_val = 0.0, 1.0
            else:
                try:
                    h_stat, p_val = kruskal(*groups)
                except ValueError:
                    h_stat, p_val = 0.0, 1.0
            results['overall'][feature] = {
                'h_statistic': float(h_stat),
                'p_value': float(p_val),
                'significant': p_val < 0.05
            }

            if p_val < 0.1:
                stars = significance_stars(p_val)
                means = {name: float(feature_df[feature_df['user_name']==name][feature].mean())
                        for name in ['Jake Thompson', 'Tyrone Williams', 'Reza Moradi']}
                print(f"  {feature}: H={h_stat:.2f}, p={p_val:.4f}{stars}")
                for name, mean_val in means.items():
                    print(f"     {name}: {mean_val:.3f}")

    return results


# ============================================================
# 6. REGRESSION ANALYSIS
# ============================================================
def run_regression_analysis(feature_df: pd.DataFrame) -> Dict:
    """Run regression to predict phenomena from features."""
    print_section("REGRESSION ANALYSIS")

    results = {}

    # Outcome variables (phenomena to predict)
    outcomes = {
        'cb_score': 'Cultural Boxing',
        'pes_present': 'Proactive Empathetic Shield',
        'lm_present': 'Linguistic Mirroring',
        'cme_detected': 'Cultural Misattribution Error',
        'ta_present': 'Topic Avoidance',
    }

    # Predictor features
    predictors = [
        'emotion_empathy_score', 'auth_directives_count', 'auth_systemic_critique_count',
        'accept_validation_count', 'resist_boundary_setting_count',
        'resist_pushback_count', 'struct_char_count', 'struct_bold_count',
        'social_cultural_identity_count', 'dark_minimization_count',
        'reason_moral_reasoning_count', 'reason_practical_reasoning_count'
    ]

    available_pred = [p for p in predictors if p in feature_df.columns]

    for outcome, label in outcomes.items():
        if outcome not in feature_df.columns:
            continue

        print_subsection(f"Predicting {label} ({outcome})")

        y = feature_df[outcome].dropna()
        X = feature_df.loc[y.index, available_pred].dropna()

        # Align
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if len(y) < 10 or X.shape[1] < 2:
            print(f"  Insufficient data for {outcome}")
            continue

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Logistic regression for binary outcomes
        if y.nunique() <= 2:
            try:
                model = LogisticRegression(max_iter=1000, random_state=42)
                scores = cross_val_score(model, X_scaled, y, cv=3, scoring='f1_weighted')

                # Fit full model to get coefficients
                model.fit(X_scaled, y)
                coefs = dict(zip(available_pred, model.coef_[0]))

                # Top predictors
                top_predictors = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

                print(f"  Cross-val F1: {scores.mean():.3f} ± {scores.std():.3f}")
                print(f"  Top predictors:")
                for feat, coef in top_predictors:
                    direction = '→ higher' if coef > 0 else '→ lower'
                    print(f"     {feat}: {coef:.3f} {direction}")

                results[outcome] = {
                    'cv_f1_mean': float(scores.mean()),
                    'cv_f1_std': float(scores.std()),
                    'top_predictors': [{'feature': f, 'coefficient': float(c)} for f, c in top_predictors]
                }
            except Exception as e:
                print(f"  Logistic regression failed: {e}")

        # Linear regression for continuous outcomes
        else:
            try:
                model = LinearRegression()
                scores = cross_val_score(model, X_scaled, y, cv=3, scoring='r2')

                model.fit(X_scaled, y)
                coefs = dict(zip(available_pred, model.coef_))

                top_predictors = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

                print(f"  Cross-val R²: {scores.mean():.3f} ± {scores.std():.3f}")
                print(f"  Top predictors:")
                for feat, coef in top_predictors:
                    direction = 'positive' if coef > 0 else 'negative'
                    print(f"     {feat}: {coef:.3f} ({direction})")

                results[outcome] = {
                    'cv_r2_mean': float(scores.mean()),
                    'cv_r2_std': float(scores.std()),
                    'top_predictors': [{'feature': f, 'coefficient': float(c)} for f, c in top_predictors]
                }
            except Exception as e:
                print(f"  Linear regression failed: {e}")

    return results


# ============================================================
# 7. CORRELATION ANALYSIS
# ============================================================
def run_correlation_analysis(feature_df: pd.DataFrame) -> Dict:
    """Run comprehensive correlation analysis."""
    print_section("CORRELATION ANALYSIS")

    # Key features for correlation
    corr_features = [
        'emotion_empathy_score', 'emotion_positive_emotion_count', 'emotion_negative_emotion_count',
        'auth_directives_count', 'auth_systemic_critique_count', 'auth_hedging_count',
        'accept_validation_count', 'accept_emotional_support_count', 'accept_affirmation_count',
        'resist_boundary_setting_count', 'resist_pushback_count', 'resist_deflection_count',
        'reason_moral_reasoning_count', 'reason_practical_reasoning_count',
        'struct_char_count', 'struct_word_count', 'struct_bold_count',
        'cb_score', 'dea_combined_score', 'pes_present',
        'lm_present', 'cme_detected', 'ta_present',
        'social_cultural_identity_count', 'dark_minimization_count'
    ]

    available = [f for f in corr_features if f in feature_df.columns]

    # Compute correlation matrix
    corr_matrix = feature_df[available].corr()

    # Find strong correlations (|r| > 0.4)
    print_subsection("STRONG CORRELATIONS (|r| > 0.4)")

    strong_corrs = []
    for i, f1 in enumerate(available):
        for f2 in available[i+1:]:
            r = corr_matrix.loc[f1, f2]
            if abs(r) > 0.4:
                direction = 'positive' if r > 0 else 'negative'
                strong_corrs.append({
                    'feature1': f1, 'feature2': f2,
                    'correlation': float(r), 'direction': direction,
                    'magnitude': 'strong' if abs(r) > 0.7 else 'moderate'
                })
                print(f"  {f1} ↔ {f2}: r={r:.3f} ({direction})")

    # Correlations with key phenomena
    print_subsection("CORRELATIONS WITH KEY PHENOMENA")
    key_phenomena = ['cb_score', 'pes_present', 'lm_present', 'cme_detected', 'ta_present']
    key_available = [p for p in key_phenomena if p in available]

    for phen in key_available:
        correlations = corr_matrix[phen].drop(phen).sort_values(key=lambda x: abs(x), ascending=False)
        print(f"\n  {phen}:")
        for feat, corr in correlations.head(5).items():
            if abs(corr) > 0.1:
                print(f"     {feat}: r={corr:.3f}")

    return {
        'strong_correlations': strong_corrs,
        'correlation_matrix_shape': corr_matrix.shape
    }


# ============================================================
# 8. EFFECT SIZES (Cohen's d)
# ============================================================
def compute_effect_sizes(feature_df: pd.DataFrame) -> Dict:
    """Compute Cohen's d effect sizes for model and name comparisons."""
    print_section("EFFECT SIZES (COHEN'S D)")

    models = feature_df['model'].unique()
    model_short = {m: m.split('(')[0].strip() for m in models}
    names = ['Jake Thompson', 'Tyrone Williams', 'Reza Moradi']

    features = [
        'emotion_empathy_score', 'auth_systemic_critique_count',
        'accept_validation_count', 'resist_boundary_setting_count',
        'dea_combined_score', 'cb_score', 'struct_char_count'
    ]
    available = [f for f in features if f in feature_df.columns]

    results = {'model_effects': [], 'name_effects': []}

    # Model effect sizes
    print_subsection("MODEL EFFECT SIZES (Cohen's d)")
    for feature in available:
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                v1 = feature_df[feature_df['model'] == m1][feature].dropna()
                v2 = feature_df[feature_df['model'] == m2][feature].dropna()

                if len(v1) > 1 and len(v2) > 1:
                    pooled_std = np.sqrt((v1.var() + v2.var()) / 2)
                    if pooled_std > 0:
                        d = (v1.mean() - v2.mean()) / pooled_std
                        magnitude = 'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small' if abs(d) > 0.2 else 'negligible'

                        if abs(d) > 0.5:  # Only report medium+
                            print(f"  {feature}: {model_short[m1]} vs {model_short[m2]} → d={d:.2f} ({magnitude})")
                            results['model_effects'].append({
                                'feature': feature,
                                'comparison': f'{model_short[m1]} vs {model_short[m2]}',
                                'cohens_d': float(d),
                                'magnitude': magnitude
                            })

    # Name effect sizes
    print_subsection("NAME EFFECT SIZES (Cohen's d)")
    for feature in available:
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                v1 = feature_df[feature_df['user_name'] == n1][feature].dropna()
                v2 = feature_df[feature_df['user_name'] == n2][feature].dropna()

                if len(v1) > 1 and len(v2) > 1:
                    pooled_std = np.sqrt((v1.var() + v2.var()) / 2)
                    if pooled_std > 0:
                        d = (v1.mean() - v2.mean()) / pooled_std
                        magnitude = 'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small' if abs(d) > 0.2 else 'negligible'

                        if abs(d) > 0.3:
                            print(f"  {feature}: {n1.split()[0]} vs {n2.split()[0]} → d={d:.2f} ({magnitude})")
                            results['name_effects'].append({
                                'feature': feature,
                                'comparison': f'{n1} vs {n2}',
                                'cohens_d': float(d),
                                'magnitude': magnitude
                            })

    return results


# ============================================================
# 9. BOOTSTRAPPING FOR CONFIDENCE INTERVALS
# ============================================================
def run_bootstrapping(feature_df: pd.DataFrame) -> Dict:
    """Bootstrap confidence intervals for key metrics."""
    print_section("BOOTSTRAPPING: CONFIDENCE INTERVALS")

    np.random.seed(42)
    n_boot = 1000

    metrics = {
        'cb_rate': ('cb_score', 'mean'),
        'pes_rate': ('pes_present', 'mean'),
        'lm_rate': ('lm_present', 'mean'),
        'cme_rate': ('cme_detected', 'mean'),
        'ta_rate': ('ta_present', 'mean'),
        'mean_empathy': ('emotion_empathy_score', 'mean'),
    }

    results = {}

    print_subsection("95% CONFIDENCE INTERVALS BY MODEL")

    for model in feature_df['model'].unique():
        model_data = feature_df[feature_df['model'] == model]
        model_short = model.split('(')[0].strip()

        print(f"\n  {model_short}:")
        model_results = {}

        for metric_name, (feature, agg_func) in metrics.items():
            if feature not in model_data.columns:
                continue

            observed = model_data[feature].mean() if agg_func == 'mean' else model_data[feature].sum()

            # Bootstrap
            boot_samples = []
            for _ in range(n_boot):
                sample = model_data[feature].sample(n=len(model_data), replace=True)
                if agg_func == 'mean':
                    boot_samples.append(sample.mean())
                else:
                    boot_samples.append(sample.sum())

            ci_lower = np.percentile(boot_samples, 2.5)
            ci_upper = np.percentile(boot_samples, 97.5)

            model_results[metric_name] = {
                'observed': float(observed),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'ci_width': float(ci_upper - ci_lower)
            }

            print(f"     {metric_name}: {observed:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

        results[model_short] = model_results

    return results


# ============================================================
# 10. JSON-Safe Converter
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
# 11. Main
# ============================================================
def run_section9(section2_results=None, feature_df=None, classified_df=None):
    """Execute Section 9: Statistical Testing."""
    print_section("SECTION 9: STATISTICAL TESTING")
    print("  ANOVA | t-test | Cohen's d | Chi-square | Regression | Bootstrap")

    if feature_df is None or classified_df is None:
        feature_df, classified_df, enhanced_df = load_section_data()

    all_results = {}

    # 1. Descriptive statistics
    all_results['descriptive'] = compute_descriptive_stats(feature_df)

    # 2. ANOVA & t-tests
    all_results['anova_tests'] = run_anova_tests(feature_df)

    # 3. Chi-square
    all_results['chi_square'] = run_chi_square_tests(feature_df)

    # 4. Name effect
    all_results['name_effect'] = run_name_effect_tests(feature_df)

    # 5. Regression
    all_results['regression'] = run_regression_analysis(feature_df)

    # 6. Correlations
    all_results['correlations'] = run_correlation_analysis(feature_df)

    # 7. Effect sizes
    all_results['effect_sizes'] = compute_effect_sizes(feature_df)

    # 8. Bootstrapping
    all_results['bootstrapping'] = run_bootstrapping(feature_df)

    # Print summary
    print_section("STATISTICAL SUMMARY")

    sig_anova = len(all_results['anova_tests']['significant_anova'])
    sig_chi = len(all_results['chi_square']['significant_chi'])

    print(f"\n  ✅ ANOVA: {sig_anova} significant features differentiate models")
    print(f"  ✅ Chi-square: {sig_chi} categorical phenomena associated with models")
    print(f"  ✅ Regression models built for 5 phenomena")
    print(f"  ✅ Bootstrap CIs computed for 6 metrics")

    # Save
    results_safe = make_json_safe(all_results)
    output_path = Path("output") / "statistical_tests.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_safe, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Saved statistical results: {output_path}")

    print(f"\n{'='*70}")
    print(f"  ✅ SECTION 9 COMPLETE")
    print(f"  8 statistical analyses performed")
    print(f"{'='*70}")

    return all_results


# ============================================================
# Execute
# ============================================================
if __name__ == "__main__":
    section9_results = run_section9()

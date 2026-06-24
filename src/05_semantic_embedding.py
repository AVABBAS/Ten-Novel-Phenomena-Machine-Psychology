"""
==============================================================
SECTION 5: SEMANTIC EMBEDDING
Sentence-Transformers | Response Vectorization | Similarity Analysis
==============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle
from datetime import datetime

# Sentence Transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# For progress tracking
from tqdm import tqdm

# ============================================================
# 0. Configuration
# ============================================================
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dimensional embeddings
EMBEDDING_DIM = 384
RANDOM_SEED = 42

# ============================================================
# 1. Load Data
# ============================================================
def load_data_for_embedding():
    """Load classified responses with full text."""
    # Load the classified dataframe
    classified_path = Path("output") / "classified_responses.csv"
    if classified_path.exists():
        df = pd.read_csv(classified_path, encoding='utf-8')
        print(f"  ✅ Loaded classified data: {len(df)} rows")
    else:
        print(f"  ❌ Classified data not found at {classified_path}")
        return None

    # We need full responses for embedding
    # The preview column has first 200 chars; we need full responses
    # Try to load from section2_results or original data
    return df


def get_full_responses(df: pd.DataFrame, section2_results: Dict) -> pd.DataFrame:
    """
    Extract full response texts from experiment histories.
    Matches responses by model, chat_id, and domain.
    """
    full_responses = []

    for h in section2_results["history_data"]:
        model_name = h["model_name"]
        data = h["data"]

        chats = data.get("chats", [])
        if not chats:
            # Try Gemini old format
            for key in data.keys():
                if key.startswith("chat_") and isinstance(data[key], dict):
                    if "prompts" in data[key] or "traps" in data[key]:
                        chats.append(data[key])
            if chats:
                chats.sort(key=lambda c: c.get("chat_id", 0))

        for chat in chats:
            chat_id = chat.get("chat_id", "unknown")
            user_name = chat.get("name", chat.get("persona", ""))
            register = chat.get("register", "")

            prompts = chat.get("traps", chat.get("prompts", []))

            for prompt in prompts:
                domain = prompt.get("trap_name", prompt.get("scenario", ""))
                response_text = prompt.get("response", "")

                full_responses.append({
                    'model': model_name,
                    'chat_id': chat_id,
                    'user_name': user_name,
                    'register': register,
                    'domain': domain,
                    'full_response': response_text
                })

    full_df = pd.DataFrame(full_responses)
    print(f"  ✅ Extracted {len(full_df)} full responses")
    return full_df


# ============================================================
# 2. Generate Embeddings
# ============================================================
def generate_embeddings(texts: List[str], model_name: str = EMBEDDING_MODEL_NAME) -> Tuple[np.ndarray, SentenceTransformer]:
    """
    Generate sentence embeddings for a list of texts.

    Args:
        texts: List of response texts
        model_name: SentenceTransformer model name

    Returns:
        embeddings: numpy array of shape (n_texts, embedding_dim)
        model: The loaded SentenceTransformer model
    """
    print(f"\n  🔧 Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"  ✅ Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    print(f"  🔧 Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalization for cosine similarity
    )

    print(f"  ✅ Generated embeddings: shape {embeddings.shape}")
    return embeddings, model


# ============================================================
# 3. Similarity Analysis
# ============================================================
def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    print(f"\n  🔧 Computing similarity matrix...")
    sim_matrix = cosine_similarity(embeddings)
    print(f"  ✅ Similarity matrix: shape {sim_matrix.shape}")
    print(f"     Mean similarity: {sim_matrix.mean():.4f}")
    print(f"     Min similarity: {sim_matrix.min():.4f}")
    print(f"     Max similarity: {sim_matrix.max():.4f}")
    return sim_matrix


def compute_cross_name_similarity(df: pd.DataFrame, embeddings: np.ndarray) -> Dict:
    """
    Compute similarity statistics across different names.
    Measures whether model responses differ by user name.
    """
    print(f"\n  🔧 Computing cross-name similarities...")

    names = ['Jake Thompson', 'Tyrone Williams', 'Reza Moradi']
    results = {}

    for model in df['model'].unique():
        model_mask = df['model'] == model
        model_indices = np.where(model_mask)[0]

        model_results = {}
        for name1 in names:
            for name2 in names:
                if name1 <= name2:  # Avoid duplicates
                    mask1 = (df['model'] == model) & (df['user_name'] == name1)
                    mask2 = (df['model'] == model) & (df['user_name'] == name2)

                    idx1 = np.where(mask1)[0]
                    idx2 = np.where(mask2)[0]

                    if len(idx1) > 0 and len(idx2) > 0:
                        if name1 == name2:
                            # Within-group similarity
                            if len(idx1) > 1:
                                sim = cosine_similarity(embeddings[idx1]).mean()
                            else:
                                sim = 1.0
                        else:
                            # Cross-group similarity
                            sim = cosine_similarity(embeddings[idx1], embeddings[idx2]).mean()

                        key = f"{name1.split()[0]}_vs_{name2.split()[0]}"
                        model_results[key] = round(float(sim), 4)

        results[model] = model_results

    # Print results
    print_subsection("CROSS-NAME SIMILARITY BY MODEL")
    for model, sims in results.items():
        print(f"\n  📁 {model}:")
        for key, val in sims.items():
            print(f"     {key}: {val:.4f}")

    return results


def compute_cross_model_similarity(df: pd.DataFrame, embeddings: np.ndarray) -> Dict:
    """
    Compare response similarity across models for the same persona/domain.
    """
    print(f"\n  🔧 Computing cross-model similarities...")

    models = df['model'].unique()
    results = {}

    # Overall cross-model similarity
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            idx1 = np.where(df['model'] == model1)[0]
            idx2 = np.where(df['model'] == model2)[0]

            sim = cosine_similarity(embeddings[idx1], embeddings[idx2]).mean()
            key = f"{model1.split('(')[0].strip()}_vs_{model2.split('(')[0].strip()}"
            results[key] = round(float(sim), 4)

    # By persona
    persona_results = {}
    for name in df['user_name'].unique():
        name_mask = df['user_name'] == name
        persona_sims = {}

        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                idx1 = np.where((df['model'] == model1) & name_mask)[0]
                idx2 = np.where((df['model'] == model2) & name_mask)[0]

                if len(idx1) > 0 and len(idx2) > 0:
                    sim = cosine_similarity(embeddings[idx1], embeddings[idx2]).mean()
                    key = f"{model1.split('(')[0].strip()}_vs_{model2.split('(')[0].strip()}"
                    persona_sims[key] = round(float(sim), 4)

        persona_results[name] = persona_sims

    print_subsection("CROSS-MODEL SIMILARITY")
    print("\n  Overall:")
    for key, val in results.items():
        print(f"     {key}: {val:.4f}")

    print("\n  By Persona:")
    for name, sims in persona_results.items():
        print(f"     {name}:")
        for key, val in sims.items():
            print(f"       {key}: {val:.4f}")

    return {'overall': results, 'by_persona': persona_results}


# ============================================================
# 4. Embedding Statistics & Analysis
# ============================================================
def compute_embedding_stats(embeddings: np.ndarray, df: pd.DataFrame) -> Dict:
    """Compute statistics on the embedding space."""
    print(f"\n  🔧 Computing embedding statistics...")

    stats = {
        'embedding_shape': embeddings.shape,
        'embedding_dim': embeddings.shape[1],
        'global_mean': float(embeddings.mean()),
        'global_std': float(embeddings.std()),
        'global_min': float(embeddings.min()),
        'global_max': float(embeddings.max()),
    }

    # Per-model statistics
    model_stats = {}
    for model in df['model'].unique():
        mask = df['model'] == model
        model_emb = embeddings[mask]
        model_stats[model] = {
            'count': len(model_emb),
            'mean_norm': float(np.linalg.norm(model_emb, axis=1).mean()),
            'std_norm': float(np.linalg.norm(model_emb, axis=1).std()),
            'intra_similarity': float(cosine_similarity(model_emb).mean()) if len(model_emb) > 1 else 1.0
        }
    stats['by_model'] = model_stats

    # Per-name statistics
    name_stats = {}
    for name in df['user_name'].unique():
        mask = df['user_name'] == name
        name_emb = embeddings[mask]
        name_stats[name] = {
            'count': len(name_emb),
            'mean_norm': float(np.linalg.norm(name_emb, axis=1).mean()),
            'intra_similarity': float(cosine_similarity(name_emb).mean()) if len(name_emb) > 1 else 1.0
        }
    stats['by_name'] = name_stats

    # Per-domain statistics
    domain_stats = {}
    for domain in df['domain'].unique():
        mask = df['domain'] == domain
        domain_emb = embeddings[mask]
        domain_stats[domain] = {
            'count': len(domain_emb),
            'intra_similarity': float(cosine_similarity(domain_emb).mean()) if len(domain_emb) > 1 else 1.0
        }
    stats['by_domain'] = domain_stats

    # Print summary
    print_subsection("EMBEDDING STATISTICS")
    print(f"\n  Shape: {stats['embedding_shape']}")
    print(f"  Global Mean: {stats['global_mean']:.4f}")
    print(f"  Global Std: {stats['global_std']:.4f}")

    print("\n  Intra-Model Similarity:")
    for model, mstats in model_stats.items():
        print(f"     {model}: {mstats['intra_similarity']:.4f}")

    print("\n  Intra-Name Similarity:")
    for name, nstats in name_stats.items():
        print(f"     {name}: {nstats['intra_similarity']:.4f}")

    print("\n  Intra-Domain Similarity:")
    for domain, dstats in domain_stats.items():
        print(f"     {domain}: {dstats['intra_similarity']:.4f}")

    return stats


# ============================================================
# 5. Embedding Space Analysis: Distance Metrics
# ============================================================
def compute_distance_metrics(embeddings: np.ndarray, df: pd.DataFrame) -> Dict:
    """
    Compute distance metrics between groups to quantify differentiation.
    """
    print(f"\n  🔧 Computing distance metrics...")

    metrics = {}

    # Distance between models for same persona+domain
    models = df['model'].unique()
    names = df['user_name'].unique()

    cross_model_distances = []
    for name in names:
        for model1 in models:
            for model2 in models:
                if model1 < model2:
                    mask1 = (df['model'] == model1) & (df['user_name'] == name)
                    mask2 = (df['model'] == model2) & (df['user_name'] == name)

                    idx1 = np.where(mask1)[0]
                    idx2 = np.where(mask2)[0]

                    if len(idx1) > 0 and len(idx2) > 0:
                        # Average distance between model responses for same persona
                        sim = cosine_similarity(embeddings[idx1], embeddings[idx2]).mean()
                        cross_model_distances.append({
                            'name': name,
                            'model1': model1,
                            'model2': model2,
                            'similarity': float(sim),
                            'distance': float(1 - sim)
                        })

    metrics['cross_model_distances'] = cross_model_distances

    # Distance between names for same model+domain
    cross_name_distances = []
    for model in models:
        for name1 in names:
            for name2 in names:
                if name1 < name2:
                    mask1 = (df['model'] == model) & (df['user_name'] == name1)
                    mask2 = (df['model'] == model) & (df['user_name'] == name2)

                    idx1 = np.where(mask1)[0]
                    idx2 = np.where(mask2)[0]

                    if len(idx1) > 0 and len(idx2) > 0:
                        sim = cosine_similarity(embeddings[idx1], embeddings[idx2]).mean()
                        cross_name_distances.append({
                            'model': model,
                            'name1': name1,
                            'name2': name2,
                            'similarity': float(sim),
                            'distance': float(1 - sim)
                        })

    metrics['cross_name_distances'] = cross_name_distances

    # Print key findings
    print_subsection("CROSS-MODEL DISTANCES (same persona)")
    for d in sorted(cross_model_distances, key=lambda x: x['distance'], reverse=True)[:5]:
        print(f"  {d['name']}: {d['model1'].split('(')[0].strip()} vs {d['model2'].split('(')[0].strip()} → distance={d['distance']:.4f}")

    print_subsection("CROSS-NAME DISTANCES (same model)")
    for d in sorted(cross_name_distances, key=lambda x: x['distance'], reverse=True)[:5]:
        print(f"  {d['model'].split('(')[0].strip()}: {d['name1'].split()[0]} vs {d['name2'].split()[0]} → distance={d['distance']:.4f}")

    return metrics


# ============================================================
# 6. Similarity to Reference Texts
# ============================================================
def compute_reference_similarity(embeddings: np.ndarray, model: SentenceTransformer,
                                  df: pd.DataFrame) -> Dict:
    """
    Compute similarity of each response to key reference concepts.
    """
    print(f"\n  🔧 Computing reference similarities...")

    references = {
        'empathy': "I understand how you feel and want to help you through this difficult situation with compassion and support.",
        'professional': "Based on professional standards and best practices, here is the recommended course of action.",
        'direct': "Return it. That's the right answer.",
        'systemic': "The system has biases and structural problems that need to be acknowledged and addressed.",
        'protective': "Don't change who you are. You deserve to be treated with dignity and respect.",
        'procedural': "Follow these standard procedures and protocols to ensure compliance with regulations.",
        'cultural': "Your cultural heritage and traditions are valuable and worth sharing with others.",
        'boundary': "You have the right to set boundaries and keep your personal information private.",
        'legalistic': "Under the applicable laws and regulations, this action constitutes a violation.",
        'warm': "I'm so glad you asked! This is wonderful and I'm excited to help you with this.",
    }

    ref_embeddings = {}
    for key, text in references.items():
        ref_embeddings[key] = model.encode([text], normalize_embeddings=True)[0]

    # Compute similarity of each response to each reference
    ref_similarities = {}
    for key, ref_emb in ref_embeddings.items():
        sims = cosine_similarity(embeddings, ref_emb.reshape(1, -1)).flatten()
        ref_similarities[key] = sims

    # Add to dataframe context
    print_subsection("MEAN REFERENCE SIMILARITY BY MODEL")
    for key in references.keys():
        print(f"\n  {key}:")
        for model in df['model'].unique():
            mask = df['model'] == model
            mean_sim = ref_similarities[key][mask].mean()
            print(f"     {model.split('(')[0].strip()}: {mean_sim:.4f}")

    return ref_similarities


# ============================================================
# 7. Save Embeddings
# ============================================================
def save_embeddings(embeddings: np.ndarray, df: pd.DataFrame,
                    model: SentenceTransformer, output_dir: Path = Path("output")):
    """Save embeddings and associated metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings as numpy array
    np.save(output_dir / "response_embeddings.npy", embeddings)
    print(f"  💾 Saved embeddings: {output_dir / 'response_embeddings.npy'}")

    # Save embedding metadata
    embedding_meta = {
        'model_name': EMBEDDING_MODEL_NAME,
        'embedding_dim': embeddings.shape[1],
        'num_responses': embeddings.shape[0],
        'timestamp': datetime.now().isoformat(),
        'normalized': True
    }
    with open(output_dir / "embedding_metadata.json", 'w') as f:
        json.dump(embedding_meta, f, indent=2)
    print(f"  💾 Saved metadata: {output_dir / 'embedding_metadata.json'}")

    # Save embedding DataFrame (first 50 PCs for efficiency)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(50, embeddings.shape[1]), random_state=RANDOM_SEED)
    embeddings_pca = pca.fit_transform(embeddings)

    emb_df = df[['model', 'user_name', 'register', 'domain']].copy()
    for i in range(embeddings_pca.shape[1]):
        emb_df[f'emb_pc{i+1}'] = embeddings_pca[:, i]

    emb_df.to_csv(output_dir / "embeddings_pca50.csv", index=False)
    print(f"  💾 Saved PCA-reduced embeddings: {output_dir / 'embeddings_pca50.csv'}")
    print(f"     Explained variance (50 PCs): {pca.explained_variance_ratio_.sum():.3f}")

    return embeddings_pca


# ============================================================
# 8. Print Helpers
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
# 9. Main
# ============================================================
def run_section5(section2_results: Dict, classified_df: pd.DataFrame):
    """Execute Section 5: Semantic Embedding."""
    print_section("SECTION 5: SEMANTIC EMBEDDING")
    print(f"  Model: {EMBEDDING_MODEL_NAME} | Dim: {EMBEDDING_DIM}")

    # Get full response texts
    full_df = get_full_responses(classified_df, section2_results)

    if full_df is None or len(full_df) == 0:
        print("  ❌ No responses to embed")
        return None

    # Generate embeddings
    texts = full_df['full_response'].tolist()
    embeddings, model = generate_embeddings(texts)

    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(embeddings)

    # Cross-name similarity
    cross_name_sims = compute_cross_name_similarity(full_df, embeddings)

    # Cross-model similarity
    cross_model_sims = compute_cross_model_similarity(full_df, embeddings)

    # Embedding statistics
    emb_stats = compute_embedding_stats(embeddings, full_df)

    # Distance metrics
    dist_metrics = compute_distance_metrics(embeddings, full_df)

    # Reference similarity
    ref_sims = compute_reference_similarity(embeddings, model, full_df)

    # Save
    embeddings_pca = save_embeddings(embeddings, full_df, model)

    print(f"\n{'='*70}")
    print(f"  ✅ SECTION 5 COMPLETE")
    print(f"  {embeddings.shape[0]} responses embedded in {embeddings.shape[1]}-dim space")
    print(f"{'='*70}")

    return {
        'embeddings': embeddings,
        'model': model,
        'full_df': full_df,
        'sim_matrix': sim_matrix,
        'cross_name_sims': cross_name_sims,
        'cross_model_sims': cross_model_sims,
        'emb_stats': emb_stats,
        'dist_metrics': dist_metrics,
        'ref_sims': ref_sims,
        'embeddings_pca': embeddings_pca
    }


# ============================================================
# Execute
# ============================================================
if __name__ == "__main__":
    # Load classified data
    classified_df = load_data_for_embedding()
    if classified_df is not None:
        section5_results = run_section5(section2_results, classified_df)

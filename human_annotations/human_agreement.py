import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sentence_transformers import SentenceTransformer, util
from typing import Tuple, List, Dict, Optional

# Load the sentence transformer model once
MODEL = SentenceTransformer('all-MiniLM-L6-v2')


def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two text strings using sentence embeddings.

    Args:
        text1 (str): First text input.
        text2 (str): Second text input.

    Returns:
        float: Cosine similarity score.
    """
    embedding1 = MODEL.encode(text1, convert_to_tensor=True)
    embedding2 = MODEL.encode(text2, convert_to_tensor=True)
    return util.cos_sim(embedding1, embedding2).item()


def compute_agreement(
    df: pd.DataFrame,
    column1: str,
    column2: str,
    threshold: float = 0.9
) -> Dict[str, Optional[float]]:
    """
    Compute exact match (Cohen's Kappa) and soft agreement (semantic similarity)
    between two columns in a DataFrame.
    """
    soft_matches = [
        compute_similarity(a, b) >= threshold
        for a, b in zip(df[column1], df[column2])
    ]

    soft_agreement = (
        sum(soft_matches) / len(soft_matches) if soft_matches else None
    )
    kappa = (
        cohen_kappa_score(df[column1], df[column2])
        if len(df) > 1 else None
    )

    return {
        'columns': (column1, column2),
        'soft_agreement': round(soft_agreement, 3) if soft_agreement is not None else None,
        'kappa_exact': round(kappa, 3) if kappa is not None else None
    }



def main():
    # Define path to CSV file
    file_path = "results/classnames_e1.csv"  # Adjust as necessary

    # Load data
    df = pd.read_csv(file_path)

    # Define ontology column pairs to compare
    ontology_pairs: List[Tuple[str, str]] = [
        ('CLO_C', 'CLO_M'),
        ('CL_C', 'CL_M'),
        ('UBERON_C', 'UBERON_M'),
        ('BTO_C', 'BTO_M'),
    ]

    # Perform agreement analysis
    results = [
        compute_agreement(df, col1, col2) for col1, col2 in ontology_pairs
    ]

    # Display results
    for result in results:
        col1, col2 = result['columns']
        print(f"\nComparison: {col1} vs {col2}")
        print(f"  Soft agreement (sim ≥ 0.9): {result['soft_agreement']}")
        print(f"  Cohen's Kappa (exact match): {result['kappa_exact']}")


if __name__ == "__main__":
    main()

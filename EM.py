import pandas as pd
import re


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, and extra spaces."""
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())


def exact_match_score(dataset_path: str) -> dict:
    df = pd.read_csv(dataset_path)
    base_em = []
    fine_tuned_em = []

    for _, row in df.iterrows():
        ground_truth = normalize_answer(str(row['Ground Truth Answer']))
        base_answer = normalize_answer(str(row['Base Model Answer']))
        fine_tuned_answer = normalize_answer(str(row['Finetuned Model Answer']))

        base_em.append(1 if ground_truth == base_answer else 0)
        fine_tuned_em.append(1 if ground_truth == fine_tuned_answer else 0)

    return {
        'base_model_em': sum(base_em) / len(base_em),
        'fine_tuned_model_em': sum(fine_tuned_em) / len(fine_tuned_em)
    }


# Usage
dataset_path = "/Users/arhanwadhwani/Documents/cs578/cs578 project/eval dataset.csv"
scores = exact_match_score(dataset_path)
print(f"Base Model EM: {scores['base_model_em']:.4f}")
print(f"Fine-Tuned Model EM: {scores['fine_tuned_model_em']:.4f}")
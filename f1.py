import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
from typing import List, Tuple


def tokenize(text: str) -> List[str]:
    """Simple tokenization by splitting on whitespace and converting to lowercase."""
    return text.lower().split()


def calculate_f1(predicted: str, ground_truth: str) -> float:
    """
    Calculate F1 score between predicted and ground truth answers.
    Returns F1 score based on token overlap.
    """
    pred_tokens = set(tokenize(predicted))
    truth_tokens = set(tokenize(ground_truth))

    if not pred_tokens and not truth_tokens:
        return 1.0  # Both empty, perfect match
    if not pred_tokens or not truth_tokens:
        return 0.0  # One is empty, no match

    # Calculate precision and recall
    common_tokens = pred_tokens.intersection(truth_tokens)
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)

    # Calculate F1
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_answers(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Evaluate F1 scores for base and fine-tuned model answers.
    Returns tuple of (base_avg_f1, finetuned_avg_f1)
    """
    base_f1_scores = []
    finetuned_f1_scores = []

    for _, row in df.iterrows():
        ground_truth = row['Ground Truth Answer']
        base_answer = row['Base Model Answer']
        finetuned_answer = row['Finetuned Model Answer']

        # Calculate F1 for base model
        base_f1 = calculate_f1(base_answer, ground_truth)
        base_f1_scores.append(base_f1)

        # Calculate F1 for fine-tuned model
        finetuned_f1 = calculate_f1(finetuned_answer, ground_truth)
        finetuned_f1_scores.append(finetuned_f1)

    # Calculate average F1 scores
    base_avg_f1 = np.mean(base_f1_scores)
    finetuned_avg_f1 = np.mean(finetuned_f1_scores)

    return base_avg_f1, finetuned_avg_f1


def main():
    # Example: Load your dataset (replace with your actual file path)
    # Assuming CSV format with columns: context, question, ground_truth_answer,
    # base_model_answer, fine_tuned_model_answer
    try:
        df = pd.read_csv('/Users/arhanwadhwani/Documents/cs578/cs578 project/eval dataset.csv')

        # Verify required columns exist
        required_columns = ['Question', 'Ground Truth Answer',
                            'Base Model Answer', 'Finetuned Model Answer']
        if not all(col in df.columns for col in required_columns):
         
            raise ValueError("Dataset missing required columns")

        # Calculate F1 scores
        base_f1, finetuned_f1 = evaluate_answers(df)

        # Print results
        print(f"Evaluation Results:")
        print(f"Base Model Average F1 Score: {base_f1:.4f}")
        print(f"Fine-tuned Model Average F1 Score: {finetuned_f1:.4f}")
        print(f"Improvement: {(finetuned_f1 - base_f1):.4f}")

        # Optional: Save detailed results
        df['base_f1'] = [calculate_f1(row['Base Model Answer'], row['Ground Truth Answer'])
                         for _, row in df.iterrows()]
        df['finetuned_f1'] = [calculate_f1(row['Finetuned Model Answer'], row['Ground Truth Answer'])
                              for _, row in df.iterrows()]
        df.to_csv('evaluation_results.csv', index=False)
        print("Detailed results saved to 'evaluation_results.csv'")

    except FileNotFoundError:
        print("Error: Dataset file not found. Here's an example of how to create a sample dataset:")
        # Create sample data if file not found
        sample_data = {
            'context': ['The capital of France is Paris.', 'Python is a programming language.'],
            'question': ['What is the capital of France?', 'What is Python?'],
            'ground_truth_answer': ['Paris', 'programming language'],
            'base_model_answer': ['Paris France', 'language'],
            'fine_tuned_model_answer': ['Paris', 'programming language']
        }
        sample_df = pd.DataFrame(sample_data)
        base_f1, finetuned_f1 = evaluate_answers(sample_df)
        print(f"Sample Data Results:")
        print(f"Base Model Average F1 Score: {base_f1:.4f}")
        print(f"Fine-tuned Model Average F1 Score: {finetuned_f1:.4f}")


if __name__ == "__main__":
    main()
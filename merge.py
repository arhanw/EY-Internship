import pandas as pd

gt = pd.read_csv("/Users/arhanwadhwani/Documents/cs578/cs578 project/hclTech/merged_output.csv")
gt = gt.rename(columns={"Answer":"Ground Truth Answer"})
print(gt.columns)
base = pd.read_csv("/Users/arhanwadhwani/Documents/cs578/cs578 project/hclTech/dataset_with_answers_base_model.csv")
base = base.rename(columns={"Answer":"Base Model Answer"})
print(base.columns)
finetuned = pd.read_csv("/Users/arhanwadhwani/Documents/cs578/cs578 project/hclTech/dataset_with_answers_tuned_model.csv")
finetuned = finetuned.rename(columns={"Answer":"Finetuned Model Answer"})
print(finetuned.columns)
new_csv = pd.concat([base['Question'], gt['Ground Truth Answer'], base['Base Model Answer'], finetuned['Finetuned Model Answer']], axis=1)
print(new_csv.columns)
new_csv.to_csv("/Users/arhanwadhwani/Documents/cs578/cs578 project/eval dataset.csv")
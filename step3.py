import pandas as pd
import openpyxl

df1 = pd.read_csv('file_with_toxicity_scores.csv')
print(len(df1))
df1.dropna(subset=['prompt_toxicity', 'response_toxicity'], inplace=True)
print(len(df1))
print(df1['response_toxicity'].mean())

df2 = pd.read_csv('ppo_rank16_analysis2.csv')
print(len(df2))
df2.dropna(subset=['prompt_toxicity', 'response_toxicity'], inplace=True)
print(len(df2))
print(df2['response_toxicity'].mean())
df2.to_excel("output ppo v2.xlsx")
import pandas as pd
df = pd.read_csv('finetuning/query_model/data/test_data.csv')
df.drop(columns=['pattern'], inplace=True)
df.to_csv('finetuning/query_model/data/test_data.csv', index=False)
print(f'Columns: {list(df.columns)}, Rows: {len(df)}')

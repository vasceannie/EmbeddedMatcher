import pandas as pd

with open('target_tax.csv', 'r') as file:
    df = pd.read_csv(file, index_col=0, low_memory=False, dtype=str, header=0)
    df = df[["Flag", "Lv1NodeName", "Lv2NodeName", "Lv3NodeName", "Lv4NodeName"]]

df = df.dropna()

print(df.head())

df_staged = []

for index, row in df.iterrows():
    for i in range(-1, -5, -1):
        if row.iloc[i]:
            combined_string = row.iloc[i+2] + " " + row.iloc[i+3] + " " + row.iloc[i+4] + " " + row.iloc[i+5]
            df_staged.append([row.iloc[-1], combined_string])
            break

df_staged = pd.DataFrame(df_staged, columns=["Title", "Content"])

print(df_staged.head())

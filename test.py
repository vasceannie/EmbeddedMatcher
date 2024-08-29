import pandas as pd

def get_target_csv(csv_path):
    with open(csv_path, 'r') as file:
        df = pd.read_csv(file, index_col=0, low_memory=False, dtype=str, header=0)
        df = df[["Flag", "Lv1NodeName", "Lv2NodeName", "Lv3NodeName", "Lv4NodeName"]]

    df = df.dropna()
    return df

def stage_data(df):
    staged_data = []
    for index, row in df.iterrows():
        for i in range(-1, -5, -1):
            if pd.notna(row.iloc[i]):
                combined_string = row.iloc[i+2] + " " + row.iloc[i+3] + " " + row.iloc[i+4] + " " + row.iloc[i+5]
                staged_data.append([row.iloc[-1], combined_string])
                break
    df = pd.DataFrame(staged_data, columns=["Title", "Content"])
    return df


def main():
    df = get_target_csv("target_tax.csv")
    df = stage_data(df)
    print(df.head())

if __name__ == "__main__":
    main()

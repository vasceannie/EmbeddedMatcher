import pandas as pd
import ast
import openai as OpenAI
import tiktoken
from openai.types import Embedding, CreateEmbeddingResponse
import os
from scipy import spatial
from dotenv import load_dotenv
from torch import cosine_similarity
import torch

load_dotenv()

def get_target_csv(csv_path):
    with open(csv_path, 'r') as file:
        df = pd.read_csv(file, index_col=0, low_memory=False, dtype=str, header=0)
        df = df[["Flag", "Lv1NodeName", "Lv2NodeName", "Lv3NodeName", "Lv4NodeName"]]

    df = df.dropna()
    return df

def get_source_csv(csv_path):
    with open(csv_path, 'r') as file:
        df = pd.read_csv(file, index_col=0, low_memory=False, dtype=str, header=0)
        print("Source CSV columns:", df.columns.tolist())  # Debugging line
        df = df[["classification_code", "classification_name"]]  # Ensure these columns exist

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

def get_embedding(text, model="text-embedding-3-small"):
    client = OpenAI.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def search_embeddings(df, commodity, n=3, pprint=True):
    query_embedding = get_embedding(commodity, model='text-embedding-3-small')
    
    # Convert 'ada_embedding' from string to list if necessary
    df['ada_embedding'] = df['ada_embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Ensure tensors are used and reshape if necessary
    df['similarity'] = df['ada_embedding'].apply(lambda x: cosine_similarity(torch.tensor(x).unsqueeze(0), torch.tensor(query_embedding).unsqueeze(0)))  # Ensure tensors are 2D
    results = (
        df.sort_values('similarity', ascending=False)
        .head(n)
    )
    if pprint:
        for r in results:
            print(r)
            print()
    return results

def main():
    # df = get_target_csv("target_tax.csv")
    # df = stage_data(df)
    e_df = pd.read_csv("target_tax_embeddings.csv")
    source_df = get_source_csv("distinct_source_class.csv")
    
    print(e_df.head())
    print(source_df.head())
    
    results = []
    for index, row in source_df.iterrows():
        result = search_embeddings(e_df, row.classification_name, n=1, pprint=False)
        results.append(result)  # Store the result DataFrame
    
    combined_results = pd.concat(results, ignore_index=True)  # Combine all results into a single DataFrame
    print(combined_results)

if __name__ == "__main__":
    main()

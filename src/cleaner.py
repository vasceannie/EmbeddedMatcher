import pandas as pd
import openpyxl
from openpyxl.worksheet.datavalidation import DataValidation

# Load the data from the CSV or Excel file
file_path = 'data-1724876252370.csv'  # Change this to the correct file path

# Read the file
df = pd.read_csv(file_path, low_memory=False, dtype=str)

# Prepare the output Excel file
output_file = 'validation_output.xlsx'

# Function to create Excel file with data validation
def create_validation_file(df):
    # Create a new Excel workbook and select the active sheet
    wb = openpyxl.Workbook()
    ws = wb.active

    # Write the headers
    headers = ['UNSPSC Code', 'UNSPSC Name', 'Processed Category', 'Synonym Matches', 'Best Match', 'Processed Matches']
    ws.append(headers)

    # Write the data
    for _, row in df.iterrows():
        ws.append([
            row['classification_code'],
            row['classification_name'],
            row['processed_category'],
            row['synonym_matches'],
            '',  # Empty cell for Best Match
            ''   # Empty cell for Processed Matches
        ])

    # Add data validation for Best Match column
    for row in range(2, len(df) + 2):  # Start from row 2 (skip header)
        cell = ws.cell(row=row, column=5)  # Column E (Best Match)
        synonym_matches = df.iloc[row-2]['synonym_matches']
        
        # Handle cases where synonym_matches is not a string
        if pd.isna(synonym_matches) or not isinstance(synonym_matches, str):
            matches = []
        else:
            matches = synonym_matches.split(',')
        
        if matches:
            formula = f'=INDIRECT("D{row}")'
            dv = DataValidation(type="list", formula1=formula)
            ws.add_data_validation(dv)
            dv.add(cell)

    # Save the workbook
    wb.save(output_file)
    print(f"Validation file created: {output_file}")

# Start the script
if __name__ == "__main__":
    create_validation_file(df)
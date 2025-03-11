import re
import pandas as pd

def separate_string(input_string):
    # Extract the contents of the brackets
    bracket_content = re.search(r'\((.*?)\)', input_string).group(1)
    
    # Extract the remaining text excluding the brackets and the hyphen or minus sign
    remaining_text = re.sub(r'\(.*?\)| -', '', input_string).strip()
    
    return remaining_text, bracket_content

def process_datafile(input_filepath, output_filepath):
    # Read the data file into a DataFrame
    df = pd.read_csv(input_filepath)
    
    # Apply the separate_string function to each record
    df[['Remaining_Text', 'Bracket_Content']] = df['input_string'].apply(lambda x: pd.Series(separate_string(x)))
    
    # Save the updated DataFrame to a new file
    df.to_csv(output_filepath, index=False)

# Example usage
input_filepath = 'path_to_input_file.csv'
output_filepath = 'path_to_output_file.csv'
process_datafile(input_filepath, output_filepath)

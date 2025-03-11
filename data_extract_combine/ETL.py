import pandas as pd
import re
import zlib



def dataload():
    # load the data
    fake_df = pd.read_csv("source_data/fake.csv")
    true_df = pd.read_csv("source_data/true.csv")
    return fake_df, true_df


def classify_media(text):
    media_dict = {'video': ['video', 'watch'],
                  'image': ['image'],
                  'poll': ['poll'],
                  'twitter': ['twitter']}
    if pd.isna(text):  # Handle NaN cases safely
        return 'text'
    for key, value in media_dict.items():
        # Lowercase to ensure case insensitivity
        if any(word in text.lower() for word in value):
            return key
    return 'text'


def classify_and_combine(true_df, fake_df):
    # add a column to each dataframe to
    # indicate whether the news is fake or true
    true_df["label"] = 1
    fake_df["label"] = 0
    # combine the two dataframes
    combined_df = pd.concat([true_df, fake_df])
    combined_pre_clean = combined_df.copy()
    # rename the text colum "article_text"
    combined_df.rename(columns={"text": "article_text"}, inplace=True)
    combined_df['title'] = combined_df['title'].str.replace(r'[^\w\s]', '')
    combined_df['article_text'] = (
        combined_df['article_text'].str.replace(r'[^\w\s]', ''))
    # remove all leading and trailing spaces from title
    combined_df['title'] = combined_df['title'].str.strip()
    # remove all leading and trailing spaces from article_text
    combined_df['article_text'] = combined_df['article_text'].str.strip()
    # classify the media type of the title
    combined_df['media_type_title'] = (
        combined_df['title'].apply(classify_media))
    combined_df['media_type_article'] = (
        combined_df['article_text'].apply(classify_media))
    combined_df['media_type'] = combined_df.apply(
        lambda row: row['media_type_title'] if row['media_type_title'] != 'text' else row['media_type_article'], axis=1
    )
    # Month mapping dictionary
    month_map = {
        'january': '1', 'jan': '1',
        'february': '2', 'feb': '2',
        'march': '3', 'mar': '3',
        'april': '4', 'apr': '4',
        'may': '5',
        'june': '6', 'jun': '6',
        'july': '7', 'jul': '7',
        'august': '8', 'aug': '8',
        'september': '9', 'sep': '9',
        'october': '10', 'oct': '10',
        'november': '11', 'nov': '11',
        'december': '12', 'dec': '12'
    }

    # Remove punctuation & trim spaces
    combined_df['date'] = combined_df['date'].str.replace(r'[^\w\s]', '', regex=True).str.strip()

    # Extract month, day, and year
    combined_df[['month', 'day', 'year']] = combined_df['date'].str.extract(r'(\w+)\s+(\d+),?\s*(\d+)?')

    # Convert month names to numbers
    combined_df['month'] = combined_df['month'].str.lower().map(month_map)

    # Ensure numeric values and handle missing years (e.g., "23" → "2023")
    combined_df['year'] = combined_df['year'].fillna(pd.to_datetime('today').year).astype(str)
    combined_df['year'] = combined_df['year'].apply(lambda x: '20' + x if len(x) == 2 else x)

    # Ensure month, day, and year are strings and fill NaN values with empty strings
    combined_df[['year', 'month', 'day']] = combined_df[['year', 'month', 'day']].astype(str).fillna('')

    # Ensure we only concatenate if all components are present
    combined_df['date_str'] = combined_df.apply(
        lambda row: f"{row['year']}-{row['month']}-{row['day']}" if row['year'] and row['month'] and row['day'] else None,
        axis=1
    )

    # Convert to datetime, forcing errors to NaT
    combined_df['date_clean'] = pd.to_datetime(combined_df['date_str'], errors='coerce')

    # Drop rows where date conversion failed
    combined_df = combined_df.dropna(subset=['date_clean'])

    # Ensure the final format is YYYY-MM-DD
    combined_df['date_clean'] = combined_df['date_clean'].dt.strftime('%Y-%m-%d')

    # export combined_pre_clean to csv
    combined_pre_clean.to_csv("data/combined_pre_clean.csv", index=False)

    # drop combined_pre_clean datafile
    del combined_pre_clean
    return combined_df


# Function to extract the location, source, and remove the text
def extract_source_and_clean(text):
    # Define the regex pattern to match the source
    # (location + source in parentheses + hyphen)
    pattern = r'^[A-Za-z\s,/.]+ \([A-Za-z]+\) -'
    match = re.match(pattern, text)

    if match:
        # Extract the matched portion (location + source + hyphen)
        source = match.group(0).strip()

        # Remove the matched portion from the original
        # text to get the cleaned text
        cleaned_text = text.replace(source, '').strip()

        return source, cleaned_text
    else:
        return '', text


# Function to split the source into location and source,
# removing the parentheses around the source
def separate_string(input_string):
    # Extract the contents of the brackets
    bracket_content = re.search(r'\((.*?)\)', input_string).group(1)
    # Extract the remaining text excluding the brackets and the hyphen or minus sign
    remaining_text = re.sub(r'\(.*?\)| -', '', input_string).strip()
    return remaining_text, bracket_content


# Function to load and clean the data
def data_pipeline():
    #  Pipeline to transform the data
    fake_df, true_df = dataload()
    # Combine the dataframes
    combined_df = classify_and_combine(true_df, fake_df)
    print(combined_df.columns[combined_df.columns.duplicated()])
    # Extract the source and clean the text
    combined_df[['source', 'cleaned_text']] = (
        combined_df['article_text'].apply(
            lambda x: pd.Series(
                extract_source_and_clean(
                    (x)
                )
            )
        ))
    # Replace empty source with 'UNKNOWN (Unknown) -'
    # for ref "WASHINGTON (Reuters) -"
    combined_df['source'].replace('', 'UNKNOWN (Unknown)', inplace=True)
    combined_df['source'].fillna('UNKNOWN (Unknown)', inplace=True)
    # remove excess spaces all columns
    combined_df = (
        combined_df.apply(lambda x: x.str.strip()
                          if x.dtype == "object" else x))
    # split the source into location and source name
    combined_df[['location', 'source_name']] = combined_df['source'].apply(lambda x: pd.Series(separate_string(x)))
    # strip all extra spaces from location and source_name
    combined_df['location'] = combined_df['location'].str.strip()
    combined_df['source_name'] = combined_df['source_name'].str.strip()
    # turn any locations with a "/" in them into a list
    combined_df['location'] = combined_df['location'].str.split('/')
    # remove any leading or trailing spaces from the location
    combined_df['location'] = combined_df['location'].apply(lambda x: [i.strip() for i in x])
    combined_df['location'] = combined_df['location'].fillna('UNKNOWN')
    combined_df['source_name'] = combined_df['source_name'].fillna('Unknown')
    combined_df['title_length'] = combined_df['title'].apply(len)
    combined_df['text_length'] = combined_df['cleaned_text'].apply(len)
    # drop columns that are not needed
    combined_df.drop(['article_text', 'date', 'date_str', 'year', 'month', 'day'], axis=1, inplace=True)
    # USE ZLIBTO COMPRESS THE DATA
    combined_df['cleaned_text'] = combined_df['cleaned_text'].apply(lambda x: zlib.compress(x.encode('utf-8')))
    
    return combined_df


# Generate Combined Data and save as a csv file
combined_df = data_pipeline()
# check for any blank values
print(combined_df.isnull().sum())
# if no blank or null values in title, article_text, date, label,
# subject then export to csv
if combined_df.isnull().sum().sum() == 0:
    combined_df.to_csv("data/combined_data.csv", index=False)
    print("Data cleaned and saved as combined_data.csv in data folder")
else:
    print("There are still missing values in the data")


import pandas as pd
import re
import zlib
import numpy as np
import holidays
import datetime as dt
import pandas as pd
import zipfile
import io
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

# Download necessary NLTK resources (if you haven't already)
nltk.download('stopwords')
nltk.download('wordnet')

# Download the spaCy model (if you haven't already)
!python -m spacy download en_core_web_sm

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


def save_dataframe_to_zip(df, zip_filename, csv_filename='data.csv'):
    """Saves a pandas DataFrame to a zipped CSV file.

    Args:
        df: The pandas DataFrame to save.
        zip_filename: The name of the zip file to create.
        csv_filename: The name of the CSV file inside the zip archive.
    """

    # Create an in-memory buffer for the CSV data
    csv_buffer = io.StringIO()

    # Save the DataFrame to the buffer as a CSV
    df.to_csv(csv_buffer, index=False)  # index=False to exclude the index

    # Create a zip file and add the CSV data to it
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(csv_filename, csv_buffer.getvalue())


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
        lambda row: row['media_type_title'] 
        if row['media_type_title'] != 'text' 
        else row['media_type_article'], 
        axis=1
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
    combined_df['date'] = (
        combined_df['date'].str.replace(r'[^\w\s]',
                                        '',
                                        regex=True).str.strip())

    # Extract month, day, and year
    combined_df[['month', 'day', 'year']] = (
        combined_df['date'].str.extract(r'(\w+)\s+(\d+),?\s*(\d+)?'))

    # Convert month names to numbers
    combined_df['month'] = combined_df['month'].str.lower().map(month_map)

    # Ensure numeric values and handle missing years (e.g., "23" → "2023")
    combined_df['year'] = (
        combined_df['year'].fillna(pd.to_datetime('today').year).astype(str))
    combined_df['year'] = (
        combined_df['year'].apply(lambda x: '20' + x if len(x) == 2 else x))

    # Ensure month, day, and year are strings and
    # fill NaN values with empty strings
    combined_df[['year', 'month', 'day']] = (
        combined_df[['year', 'month', 'day']].astype(str).fillna(''))

    # Ensure we only concatenate if all components are present
    combined_df['date_str'] = combined_df.apply(
        lambda row: f"{row['year']}-{row['month']}-{row['day']}" 
            if row['year'] and row['month'] and row['day'] else None,
        axis=1
    )

    # Convert to datetime, forcing errors to NaT
    combined_df['date_clean'] = pd.to_datetime(combined_df['date_str'],
                                               errors='coerce')

    # Drop rows where date conversion failed
    combined_df = combined_df.dropna(subset=['date_clean'])

    # Ensure the final format is YYYY-MM-DD
    combined_df['date_clean'] = (
        combined_df['date_clean'].dt.strftime('%Y-%m-%d'))

    # ensure date_clean is in datetime format
    combined_df['date_clean'] = (
        pd.to_datetime(combined_df['date_clean']))

    # find earliest date and latest date then 
    # find all Us_holidays in period
    min_date = combined_df['date_clean'].min()
    max_date = combined_df['date_clean'].max()
    min_year = min_date.year
    max_year = max_date.year
    us_holidays = holidays.US(years=range(min_year, max_year+1))

    # Extract Features for use in future models
    combined_df['day_of_week'] = (
        combined_df['date_clean'].dt.day_name())
    combined_df['week_of_year'] = (
        combined_df['date_clean'].dt.isocalendar().week)
    combined_df['is_weekend'] = (
        combined_df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int))
    combined_df['is_weekday'] = (
        (~combined_df['day_of_week'].isin(['Saturday', 'Sunday'])).astype(int))
    combined_df['week_of_year_sin'] = (
        np.sin(2 * np.pi * combined_df['week_of_year'] / 52))
    combined_df['week_of_year_cos'] = (
        np.cos(2 * np.pi * combined_df['week_of_year'] / 52))
    combined_df['holiday'] = (
        combined_df['date_clean'].isin(us_holidays).astype(int))
    combined_df['day_of_month_sine'] = (
        np.sin(2 * np.pi * combined_df['date_clean'].dt.day / 31))
    combined_df['day_of_month_cos'] = (
        np.cos(2 * np.pi * combined_df['date_clean'].dt.day / 31))
    combined_df['month_sin'] = (
        np.sin(2 * np.pi * combined_df['date_clean'].dt.month / 12))
    combined_df['month_cos'] = (
        np.cos(2 * np.pi * combined_df['date_clean'].dt.month / 12))

    # create day label set it to holiday name if holiday,
    # else set it to day of week
    combined_df['day_label'] = combined_df['date_clean'].apply(
        lambda x: us_holidays.get(x) if x in us_holidays else x.day_name()
    )

    # export combined_pre_clean to csv
    save_dataframe_to_zip(combined_pre_clean, 'data/combined_pre_clean.zip')

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
    # Extract the remaining text excluding the 
    # brackets and the hyphen or minus sign
    remaining_text = re.sub(r'\(.*?\)| -', '', input_string).strip()
    return remaining_text, bracket_content


def clean_text(text):
    """Cleans the input text."""

    if not isinstance(text, str):
        return ""  # Return empty string for non-string input

    # 1. Remove HTML tags (if any)
    text = re.sub(r'<.*?>', '', text)

    # 2. Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # 3. Convert to lowercase
    text = text.lower()

    # 4. Tokenization and stop word removal
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # 5. Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # 6. Rejoin the cleaned words
    cleaned_text = ' '.join(words)

    return cleaned_text


# Example of sentiment analysis on the cleaned text:
def get_sentiment(text):
    if isinstance(text, str):
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    else:
        return None, None


def categorize_sentiment(polarity):
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'


def extract_locations(text):
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return locations


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
    combined_df[['location', 'source_name']] = (
        combined_df['source'].apply(lambda x: pd.Series(separate_string(x))))
    # strip all extra spaces from location and source_name
    combined_df['location'] = (
        combined_df['location'].str.strip())
    combined_df['source_name'] = (
        combined_df['source_name'].str.strip())
    # turn any locations with a "/" in them into a list
    combined_df['location'] = (
        combined_df['location'].str.split('/'))
    # remove any leading or trailing spaces from the location
    combined_df['location'] = (
        combined_df['location'].apply(lambda x: [i.strip() for i in x]))
    combined_df['location'] = (
        combined_df['location'].fillna('UNKNOWN'))
    combined_df['source_name'] = (
        combined_df['source_name'].fillna('Unknown'))
    combined_df['title_length'] = (
        combined_df['title'].apply(len))
    combined_df['text_length'] = (
        combined_df['cleaned_text'].apply(len))
    combined_df['nlp_text'] = (
        combined_df['cleaned_text'].apply(clean_text))
    combined_df['nlp_title'] = (
        combined_df['title'].apply(clean_text))
    combined_df['nlp_location'] = (
        combined_df['location'].apply(lambda x: [clean_text(i) for i in x]))
    combined_df[['article_polarity', 'article_subjectivity']] = (
        combined_df['nlf_text'].apply(lambda x: pd.Series(get_sentiment(x))))
    combined_df[['title_polarity', 'title_subjectivity']] = (
        combined_df['nlf_title'].apply(lambda x: pd.Series(get_sentiment(x))))
    combined_df['overall_polarity'] = (
        (combined_df['article_polarity'] + combined_df['title_polarity']) / 2)
    combined_df['sentiment_article'] = (
        combined_df['article_polarity'].apply(categorize_sentiment))
    combined_df['sentiment_title'] = (
        combined_df['title_polarity'].apply(categorize_sentiment))
    combined_df['sentiment_overall'] = (
        combined_df['overall_polarity'].apply(categorize_sentiment))
    # append nlp locations to nlp text
    combined_df['nlp_textloc'] = (
        combined_df['nlp_text'] + ' ' +
        combined_df['nlp_location'].apply(lambda x: ' '.join(x)))
    # create column with locations
    combined_df['locationsfromarticle'] = (
        combined_df['nlp_textloc'].apply(extract_locations))
    # drop columns that are not needed
    combined_df.drop(['article_text',
                      'date',
                      'date_str',
                      'media_type_title',
                      'media_type_article',
                      'nlp_textloc',
                      ], axis=1, inplace=True)
    # USE ZLIBTO COMPRESS THE DATA
    combined_df['cleaned_text'] = (
        combined_df['cleaned_text']
        .apply(lambda x: zlib.compress(x.encode('utf-8'))))
    combined_df['nlp_text'] = (
        combined_df['nlp_text']
        .apply(lambda x: zlib.compress(x.encode('utf-8'))))
    combined_df['nlp_title'] = (
        combined_df['nlp_title']
        .apply(lambda x: zlib.compress(x.encode('utf-8'))))
    combined_df['title'] = (
        combined_df['title']
        .apply(lambda x: zlib.compress(x.encode('utf-8'))))

    return combined_df


# Generate Combined Data and save as a csv file
combined_df = data_pipeline()
# check for any blank values
print(combined_df.isnull().sum())
# if no blank or null values in title, article_text, date, label,
# subject then export to csv
if combined_df.isnull().sum().sum() == 0:
    save_dataframe_to_zip(combined_df, 'data/combined_data.zip')
    print("Data cleaned and saved as combined_data.csv in data folder")
else:
    print("There are still missing values in the data")

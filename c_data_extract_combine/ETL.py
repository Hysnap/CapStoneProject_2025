import time
import pandas as pd
import re
import numpy as np
import holidays
import zipfile
import csv
from geopy.geocoders import GoogleV3
from geopy.exc import GeocoderTimedOut
from google.cloud import api_keys_v2
from google.cloud.api_keys_v2 import Key
import io
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import os
import ast
import logging
import tqdm
from flashtext import KeywordProcessor

tqdm.pandas

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler("data_pipeline.log"),
                        logging.StreamHandler()
                      ])


def restrict_api_key_server(project_id: str, key_id: str) -> Key:
    """
    Restricts the API key based on IP addresses. You can specify one or
    more IP addresses of the callers,
    for example web servers or cron jobs, that are allowed to use your API key.

    TODO(Developer): Replace the variables before running this sample.

    Args:
        project_id: Google Cloud project id.
        key_id: ID of the key to restrict. This ID is auto-created
        during key creation.
            This is different from the key string. To obtain the key_id,
            you can also use the lookup api: client.lookup_key()

    Returns:
        response: Returns the updated API Key.
    """

    # Create the API Keys client.
    client = api_keys_v2.ApiKeysClient()

    # Restrict the API key usage by specifying the IP addresses.
    # You can specify the IP addresses in IPv4 or IPv6 or a
    # subnet using CIDR notation.
    server_key_restrictions = api_keys_v2.ServerKeyRestrictions()
    server_key_restrictions.allowed_ips = ["80.189.63.110"]

    # Set the API restriction.
    # For more information on API key restriction, see:
    # https://cloud.google.com/docs/authentication/api-keys
    restrictions = api_keys_v2.Restrictions()
    restrictions.server_key_restrictions = server_key_restrictions

    key = api_keys_v2.Key()
    key.name = f"projects/{project_id}/locations/global/keys/{key_id}"
    key.restrictions = restrictions

    # Initialize request and set arguments.
    request = api_keys_v2.UpdateKeyRequest()
    request.key = key
    request.update_mask = "restrictions"

    # Make the request and wait for the operation to complete.
    response = client.update_key(request=request).result()

    print(f"Successfully updated the API key: {response.name}")
    # Use response.key_string to authenticate.
    return response


def checkdirectory():
    current_dir = os.getcwd()
    current_dir

    # check current directory contains the file README.md
    if os.path.exists("README.md"):
        print("The file README.md exists in the current directory")
    else:
        print("The file README.md does not exist in the current directory")
        print("You are in the directory: ", current_dir)
        print("Changing current directory to its parent directory")
        os.chdir(os.path.dirname(current_dir))
        print("You set a new current directory")
        current_dir = os.getcwd()
        if os.path.exists("README.md"):
            print("The file README.md exists in the current directory")
        else:
            RuntimeError("The file README.md does not exist in the"
                         " current directory, please"
                         " check the current directory")
            print("Current Directory =", current_dir)


# Download necessary NLTK resources (if you haven't already)
nltk.download('stopwords')
nltk.download('wordnet')

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize the geolocator
# Load the API key from an environment variable
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError("Google API key not found. Please set "
                       "the GOOGLE_API_KEY environment variable.")

geolocator = GoogleV3(api_key=api_key)


def save_dataframe_to_zip(df, zip_filename, csv_filename='data.csv'):
    logging.info(f"Saving Df to {zip_filename}")
    """Saves a pandas DataFrame to a zipped CSV file.

    Args:
        df: The pandas DataFrame to save.
        zip_filename: The name of the zip file to create.
        csv_filename: The name of the CSV file inside the zip archive.
    """
    # save the dataframe to a csv file
    df.to_csv(csv_filename, index=False)
    # Create an in-memory buffer for the CSV data
    csv_buffer = io.StringIO()
    # Save the DataFrame to the buffer as a CSV
    df.to_csv(csv_buffer,
              index=True,
              index_label="index",
              quoting=csv.QUOTE_NONNUMERIC
              )  # index=False to exclude the index
    # Create a zip file and add the CSV data to it
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(csv_filename, csv_buffer.getvalue())


# load the data from the csv files
def dataload():
    logging.info("Loading fake news data...")
    fake_df = pd.read_csv("2_source_data/fake.csv.zip")
    logging.info("Loading true news data...")
    true_df = pd.read_csv("2_source_data/true.csv.zip")
    logging.info("Loading test data...")
    test_data_df = pd.read_csv("2_source_data/testdata.csv.zip")
    logging.info("Loading combined misinformation data...")
    comb_misinfo_df = pd.read_csv("2_source_data/combined_misinfo.zip")
    logging.info("Appending combined misinformation data to test data...")
    test_data_df = pd.concat([test_data_df,
                              comb_misinfo_df],
                             ignore_index=True)
    logging.info("Dropping combined misinformation data from memory...")
    del comb_misinfo_df
    logging.info("Data loading completed.")
    return fake_df, true_df, test_data_df


# define media types
def classify_media(text):
    media_dict = {'video': ['video', 'watch', 'live',
                            'stream', 'youtube',
                            'vimeo', 'twitch'],
                  'audio': ['audio', 'listen', 'podcast', 'radio'],
                  'image': ['image', 'photo', 'picture', 'gif'],
                  'infographic': ['infographic'],
                  'poll': ['poll'],
                  'twitter': ['twitter', 'X', 'x', 'tweet', 'retweeted'],
                  'facebook': ['facebook', 'fb', 'like', 'share', 'comment'],
                  'instagram': ['instagram', 'ig', ],
                  'linkedin': ['linkedin', 'share'],
                  }
    # Handle NaN cases safely
    if pd.isna(text):
        return 'text'
    for key, value in media_dict.items():
        # Lowercase to ensure case insensitivity
        if any(word in text.lower() for word in value):
            return key
    # Default to 'text' if no media type is found
    return 'text'


# Data cleaning and feature extraction
def classify_and_combine(true_df, fake_df, test_data_df):
    # add a column to each dataframe to
    logging.info("Add label flags to true_df and fake_df.")
    # indicate whether the news is fake or true
    true_df["label"] = 1
    fake_df["label"] = 0
    # provide a count of rows in each dataframe
    logging.info("Add count column to true_df and fake_df.")
    true_df["count"] = true_df.index
    fake_df["count"] = fake_df.index
    test_data_df["count"] = test_data_df.index
    # log row count of files
    logging.info(f"Number of rows in true_df:  {true_df.shape[0]}")
    logging.info(f"Number of rows in fake_df:  {fake_df.shape[0]}")
    logging.info(f"Number of rows in test_data_df:  {test_data_df.shape[0]}")
    # add a column to each dataframe to indicate the source
    logging.info("Add source column to true_df and fake_df.")
    true_df["file"] = "true"
    fake_df["file"] = "fake"
    test_data_df["file"] = "test_data"
    # combine the three dataframes
    logging.info("Concatenate true_df, fake_df, and test_data_df.")
    combined_df = pd.concat([true_df, fake_df, test_data_df],
                            ignore_index=True)
    combined_pre_clean = combined_df.copy()
    # count the number of rows in the combined dataframe
    logging.info("Count the number of rows in the combined dataframe.")
    logging.info("Number of rows in combined"
                 f" dataframe: {combined_df.shape[0]}")
    # remove rows where text is identical
    logging.info("Remove rows where text is identical.")
    combined_pre_clean.drop_duplicates(subset='text', inplace=True)
    # log the number of rows in the combined dataframe
    logging.info("Number of rows in combined dataframe"
                 " after removing"
                 f" identical text: {combined_pre_clean.shape[0]}")
    logging.info("Drop rows where title or text is empty.")
    # if title is empty or fill with first sentence from text
    combined_pre_clean['title'] = (
        combined_pre_clean['title'].fillna(
            combined_pre_clean['text'].str.split('.').str[0]))
    # remove any rows where the title or text is empty
    combined_pre_clean = combined_pre_clean.dropna(subset=['title', 'text'])
    # log the number of rows in the combined dataframe
    logging.info("Number of rows in combined "
                 "dataframe after removing"
                 f" empty title or text: {combined_pre_clean.shape[0]}")
    logging.info("Drop rows where title or text is whitespace.")
    # remove any rows where the title or text is whitespace
    combined_pre_clean = (
        combined_pre_clean[combined_pre_clean['title'].
                           str.strip().astype(bool)])
    combined_pre_clean = (
        combined_pre_clean[combined_pre_clean['text'].
                           str.strip().astype(bool)])
    # log the number of rows in the combined dataframe
    logging.info("Number of rows in combined dataframe"
                 " after removing whitespace title"
                 f" or text: {combined_pre_clean.shape[0]}")
    logging.info("Title and text column cleansing")
    # rename the text colum "article_text"
    combined_pre_clean.rename(columns={"text": "article_text"}, inplace=True)
    combined_pre_clean['title'] = (
        combined_pre_clean['title'].str.replace(r'[^\w\s]', ''))
    combined_pre_clean['article_text'] = (
        combined_pre_clean['article_text'].str.replace(r'[^\w\s]', ''))
    # remove all leading and trailing spaces from title
    combined_pre_clean['title'] = combined_pre_clean['title'].str.strip()
    # remove all leading and trailing spaces from article_text
    combined_pre_clean['article_text'] = (
        combined_pre_clean['article_text'].str.strip())
    logging.info("Title and text column cleansing completed")
    logging.info("Classify media type of title and article_text")
    # classify the media type of the title
    combined_pre_clean['media_type_title'] = (
        combined_pre_clean['title'].apply(classify_media))
    combined_pre_clean['media_type_article'] = (
        combined_pre_clean['article_text'].apply(classify_media))
    combined_pre_clean['media_type'] = combined_pre_clean.apply(
        lambda row: row['media_type_title']
        if row['media_type_title'] != 'text'
        else row['media_type_article'],
        axis=1
    )
    logging.info("Classify media type of title and article_text completed")

    # extract date information from date column
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
    logging.info("Extract date information from date column")
    # set all blank date values to '1901-01-01 00:000:00'
    combined_pre_clean['date'] = (
        combined_pre_clean['date'].fillna('1901-02-01 00:00:00'))
    # Remove punctuation & trim spaces
    combined_pre_clean['date'] = (
        combined_pre_clean['date'].str.replace(r'[^\w\s]',
                                               '',
                                               regex=True).str.strip())
    # Extract month, day, and year
    combined_pre_clean[['month', 'day', 'year']] = (
        combined_pre_clean['date'].str.extract(r'(\w+)\s+(\d+),?\s*(\d+)?'))
    # Convert month names to numbers
    combined_pre_clean['month'] = (
        combined_pre_clean['month'].str.lower().map(month_map))
    # Ensure numeric values and handle missing years (e.g., "23" → "2023")
    combined_pre_clean['year'] = (
        combined_pre_clean['year'].
        fillna(pd.to_datetime('today').year).astype(str))
    combined_pre_clean['year'] = (
        combined_pre_clean['year'].
        apply(lambda x: '20' + x if len(x) == 2 else x))
    # Ensure month, day, and year are strings and
    # fill NaN values with empty strings
    combined_pre_clean[['year', 'month', 'day']] = (
        combined_pre_clean[['year', 'month', 'day']].astype(str).fillna(''))
    # Ensure we only concatenate if all components are present
    combined_pre_clean['date_str'] = combined_pre_clean.apply(
        lambda row: f"{row['year']}-{row['month']}-{row['day']}"
        if row['year'] and row['month'] and row['day']
        else None, axis=1
    )
    # Convert to datetime, forcing errors to NaT
    combined_pre_clean['date_clean'] = (
        pd.to_datetime(combined_pre_clean['date_str'],
                       errors='coerce'))
    # log rows following date extraction
    logging.info("Number of rows with NA dates:"
                 f" {combined_pre_clean['date_clean'].isna().sum()}")
    # if date is NA try and find a date in the article_text
    combined_pre_clean['date_clean'] = (
        combined_pre_clean['date_clean']
        .fillna(combined_pre_clean['article_text'].
                str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    )
    # log rows following date extraction
    logging.info("Number of rows with NA dates post articletext:"
                 f" {combined_pre_clean['date_clean'].isna().sum()}")
    # if date is NA try and find a date in the title
    combined_pre_clean['date_clean'] = (
        combined_pre_clean['date_clean']
        .fillna(combined_pre_clean['title'].
                str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
    )
    # log rows following date extraction
    logging.info("Number of rows with NA dates post title:"
                 f" {combined_pre_clean['date_clean'].isna().sum()}")
    # use NLP to extract dates from article_text
    date_formats = [
        '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
        '%Y.%m.%d', '%d.%m.%Y', '%m.%d.%Y', '%Y %m %d', '%d %m %Y', '%m %d %Y'
    ]

    def try_parsing_date(text):
        for fmt in date_formats:
            try:
                return pd.to_datetime(text, format=fmt, errors='coerce')
            except ValueError:
                continue
        return None

    combined_pre_clean['date_clean'] = (
        combined_pre_clean['date_clean'].
        fillna(combined_pre_clean['article_text'].apply(
            lambda x: try_parsing_date(' '.join(
                [word for word in x.split() if word.isdigit()]))))
    )
    combined_pre_clean['date_clean'] = (
        combined_pre_clean['date_clean'].
        fillna(combined_pre_clean['title'].apply(
            lambda x: try_parsing_date(' '.join(
                [word for word in x.split() if word.isdigit()]))))
    )
    # log count of rows with NA dates
    logging.info("Number of rows with NA dates post NLP:"
                 f" {combined_pre_clean['date_clean'].isna().sum()}")
    # force all Na dates to be 2000-02-01
    combined_pre_clean['date_clean'] = (
        combined_pre_clean['date_clean'].
        fillna(pd.to_datetime('2000-02-01')))
    # set missing month, day and year values
    combined_pre_clean[['month', 'day', 'year']] = (
        combined_pre_clean['date'].str.extract(r'(\w+)\s+(\d+),?\s*(\d+)?'))
    # Drop rows where date conversion failed
    combined_pre_clean = combined_pre_clean.dropna(subset=['date_clean'])
    # log rows following date extraction
    logging.info("Number of rows in combined dataframe"
                 f" after date extraction: {combined_pre_clean.shape[0]}")
    # Convert 'date_clean' to datetime, handling errors
    combined_pre_clean['date_clean'] = (
        pd.to_datetime(combined_pre_clean['date_clean'], errors='coerce'))
    # Now apply .dt.strftime safely
    combined_pre_clean['date_clean'] = (
        combined_pre_clean['date_clean'].dt.strftime('%Y-%m-%d'))
    # ensure date_clean is in datetime format
    combined_pre_clean['date_clean'] = (
        pd.to_datetime(combined_pre_clean['date_clean']))
    # find earliest date and latest date then
    # find all Us_holidays in period
    min_date = combined_pre_clean['date_clean'].min()
    max_date = combined_pre_clean['date_clean'].max()
    min_year = min_date.year
    max_year = max_date.year
    us_holidays = holidays.US(years=range(min_year, max_year+1))
    logging.info("create date features for ML models")
    # Extract Features for use in future models
    combined_pre_clean['day_of_week'] = (
        combined_pre_clean['date_clean'].dt.day_name())
    combined_pre_clean['week_of_year'] = (
        combined_pre_clean['date_clean'].dt.isocalendar().week)
    combined_pre_clean['is_weekend'] = (
        combined_pre_clean['day_of_week'].
        isin(['Saturday', 'Sunday']).astype(int))
    combined_pre_clean['is_weekday'] = (
        (~combined_pre_clean['day_of_week'].
         isin(['Saturday', 'Sunday'])).astype(int))
    combined_pre_clean['week_of_year_sin'] = (
        np.sin(2 * np.pi * combined_pre_clean['week_of_year'] / 52))
    combined_pre_clean['week_of_year_cos'] = (
        np.cos(2 * np.pi * combined_pre_clean['week_of_year'] / 52))
    combined_pre_clean['holiday'] = (
        combined_pre_clean['date_clean'].isin(us_holidays).astype(int))
    combined_pre_clean['day_of_month_sine'] = (
        np.sin(2 * np.pi * combined_pre_clean['date_clean'].dt.day / 31))
    combined_pre_clean['day_of_month_cos'] = (
        np.cos(2 * np.pi * combined_pre_clean['date_clean'].dt.day / 31))
    combined_pre_clean['month_sin'] = (
        np.sin(2 * np.pi * combined_pre_clean['date_clean'].dt.month / 12))
    combined_pre_clean['month_cos'] = (
        np.cos(2 * np.pi * combined_pre_clean['date_clean'].dt.month / 12))
    # create day label set it to holiday name if holiday,
    # else set it to day of week
    combined_pre_clean['day_label'] = combined_pre_clean['date_clean'].apply(
        lambda x: us_holidays.get(x) if x in us_holidays else x.day_name()
    )
    logging.info("Date information extracted from date column")
    # drop unnecessary and blank columns
    logging.info("Drop unnecessary and blank columns")
    combined_pre_clean.drop(['Column1',
                             'subject2',
                             'count',
                             'date_str',
                             'media_type_title',
                             'media_type_article',
                             'date'
                             ],
                            axis=1, inplace=True)
    # log the number of rows in the combined dataframe
    logging.info("Number of rows in combined dataframe"
                 f" after dropping columns: {combined_pre_clean.shape[0]}")
    # export combined_pre_clean to csv
    save_dataframe_to_zip(combined_pre_clean, 'data/combined_pre_clean.zip')
    logging.info("Combined data saved to combined_pre_clean.zip")
    # drop combined_pre_clean datafile
    return combined_pre_clean


# Function to extract the location, source, and remove the text
def extract_source_and_clean(text):
    # Define the regex pattern to match the source
    # (location + source in parentheses + hyphen)
    pattern = r'^[A-Za-z\s,/.]+ \([A-Za-z]+\) -'
    match = re.match(pattern, text)
    # if there is data in match then extract the source
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
    # check that passed text is a string
    if not isinstance(text, str):
        return ""  # Return empty string for non-string input
    # Remove HTML tags (if any)
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenization and stop word removal
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Rejoin the cleaned words
    cleaned_text = ' '.join(words)
    # return the cleaned text
    return cleaned_text


# Sentiment analysis on the cleaned text:
def get_sentiment(text):
    # check that passed text is a string
    if isinstance(text, str):
        # turn into a TextBlob object
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    else:
        return None, None


# Functions to categorize the polarity and subjectivity scores
def categorize_polarity(polarity):
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'


def categorize_subjectivity(subjectivity):
    if subjectivity > 0.8:
        return 'highly subjective'
    elif subjectivity > 0.6:
        return 'subjective'
    elif subjectivity > 0.4:
        return 'neutral'
    elif subjectivity > 0.2:
        return 'objective'
    else:
        return 'higly objective'


# covert a string to a list
def string_to_list(location_str):
    """Safely converts a string representation of a list to a list."""
    try:
        return ast.literal_eval(location_str)
    except (ValueError, SyntaxError, TypeError):
        return []


# Function to get geolocation information
def get_geolocation_info(location):
    usetimedelay = True
    try:
        location_info = geolocator.geocode(location, timeout=10)
        if location_info:
            if usetimedelay:
                time.sleep(1)
            return {
                'latitude': location_info.latitude,
                'longitude': location_info.longitude,
                'address': location_info.address
            }
        else:
            if usetimedelay:
                time.sleep(1)
            return {
                'latitude': None,
                'longitude': None,
                'address': None
            }
    except GeocoderTimedOut:
        print(location)
        if usetimedelay:
            time.sleep(1)
        return {
            'latitude': None,
            'longitude': None,
            'address': None
        }
    except Exception as e:
        print(f"Geocoding error: {e}")
        if usetimedelay:
            time.sleep(1)
        return {
            'latitude': None,
            'longitude': None,
            'address': None
        }


# Function to extract geolocation details
def extract_geolocation_details(address):
    if address:
        address_parts = address.split(', ')
        country = address_parts[-1] if len(address_parts) > 0 else None
        state = address_parts[-2] if len(address_parts) > 1 else None
        continent = address_parts[-3] if len(address_parts) > 2 else None
        return continent, country, state
    else:
        return None, None, None


# Function to load and clean the data
def data_pipeline(useprecombineddata=False, usepostnlpdata=False):
    logging.info("Starting data pipeline...")
    if usepostnlpdata is True:
        useprecombineddata = True
    # Pipeline to transform the data
    if useprecombineddata is False:
        logging.info("Loading data...")
        fake_df, true_df, test_data_df = dataload()

    if usepostnlpdata is False:
        logging.info("Cleaning data...")
        logging.info("Combining dataframes...")
        if useprecombineddata:
            combined_df = pd.read_csv('data/combined_pre_clean.zip')
        else:
            combined_df = classify_and_combine(true_df, fake_df, test_data_df)
        logging.info(combined_df.info())
        logging.info("Checking for duplicated columns...")
        logging.info("Duplicated columns:"
                     f" {combined_df.columns[combined_df.
                                             columns.duplicated()]}")
        logging.info("Resetting index...")
        combined_df.reset_index(drop=True, inplace=True)
        logging.info("Renaming index to article_id...")
        combined_df['article_id'] = combined_df.index
        combined_df.index.name = 'index'
        logging.info("Extracting source and cleaning text...")
        combined_df[['source',
                    'cleaned_text']] = combined_df['article_text'].apply(
            lambda x: pd.Series(extract_source_and_clean(x))
        )

        logging.info("Replacing empty sources with 'UNKNOWN (Unknown)'...")
        combined_df['source'].replace('', 'UNKNOWN (Unknown)', inplace=True)
        combined_df['source'].fillna('UNKNOWN (Unknown)', inplace=True)

        logging.info("Removing excess spaces from all columns...")
        combined_df = (
            combined_df.
            apply(lambda x: x.str.strip() if x.dtype == "object" else x))

        logging.info("Splitting source into location and source name...")
        combined_df[['location', 'source_name']] = combined_df['source'].apply(
            lambda x: pd.Series(separate_string(x))
        )
        combined_df['location'] = combined_df['location'].str.strip()
        combined_df['source_name'] = combined_df['source_name'].str.strip()

        logging.info("Processing locations...")
        combined_df['location'] = combined_df['location'].str.split('/')
        combined_df['location'] = (
            combined_df['location'].apply(lambda x: [i.strip() for i in x]))
        combined_df['location'] = combined_df['location'].fillna('UNKNOWN')
        combined_df['source_name'] = (
            combined_df['source_name'].fillna('Unknown'))

        logging.info("Calculating text and title lengths...")
        combined_df['title_length'] = combined_df['title'].apply(len)
        combined_df['text_length'] = combined_df['cleaned_text'].apply(len)

        logging.info("Cleaning text for NLP...")
        combined_df['nlp_text'] = combined_df['cleaned_text'].apply(clean_text)
        combined_df['nlp_title'] = combined_df['title'].apply(clean_text)
        combined_df['nlp_location'] = (
            combined_df['location'].
            apply(lambda x: [clean_text(i) for i in x]))
        logging.info("Saving post NLP data to CSV...")
        save_dataframe_to_zip(combined_df,  'data/combined_data_postnlp.zip',
                              'combined_data_postnlp.csv')
    else:
        try:
            logging.info("Loading post nlp data...")
            combined_df = pd.read_csv('data/combined_data_postnlp.zip')

            # Check data has loaded correctly
            logging.info(combined_df.info())
            logging.info("Checking for duplicated columns...")
            logging.info("Duplicated columns:"
                         f" {combined_df.columns[combined_df.
                             columns.duplicated()]}")
            logging.info("Resetting index...")
            combined_df.reset_index(drop=True, inplace=True)
            logging.info("Renaming index to article_id...")
            combined_df['article_id'] = combined_df.index
            combined_df.index.name = 'index'
        except (FileNotFoundError, ImportError):
            logging.error("Post NLP data not found. Generating new data...")
            usepostnlpdata = False
            combined_df = data_pipeline()

    logging.info("Performing sentiment analysis...")
    combined_df[['article_polarity', 'article_subjectivity']] = (
        combined_df['nlp_text'].apply(
            lambda x: pd.Series(get_sentiment(x))
            ))
    combined_df[['title_polarity', 'title_subjectivity']] = (
        combined_df['nlp_title'].apply(
            lambda x: pd.Series(get_sentiment(x))
            ))
    combined_df['overall_polarity'] = (
        (combined_df['article_polarity'] + combined_df['title_polarity']) / 2)
    combined_df['overall_subjectivity'] = (
        (combined_df['article_subjectivity'] +
         combined_df['title_subjectivity']) / 2)

    logging.info("Calculating contradictions and variations...")
    combined_df['contradiction_polarity'] = (
        combined_df['article_polarity'] - combined_df['title_polarity'])
    combined_df['contradiction_subjectivity'] = (
        combined_df['article_subjectivity'] -
        combined_df['title_subjectivity'])
    combined_df['polarity_variations'] = combined_df.apply(
        lambda row: row['contradiction_polarity'] / row['title_polarity']
        if row['title_polarity'] != 0 else 0, axis=1)
    combined_df['subjectivity_variations'] = (combined_df.apply(
        lambda row: row['contradiction_subjectivity'] /
        row['title_subjectivity']
        if row['title_subjectivity'] != 0 else 0, axis=1))

    logging.info("Categorizing sentiments...")
    combined_df['sentiment_article'] = (
        combined_df['article_polarity'].apply(categorize_polarity) +
        " " + combined_df['article_subjectivity'].
        apply(categorize_subjectivity))
    combined_df['sentiment_title'] = (
        combined_df['title_polarity'].apply(categorize_polarity) +
        " " + combined_df['title_subjectivity'].apply(categorize_subjectivity))
    combined_df['sentiment_overall'] = (
        combined_df['overall_polarity'].apply(categorize_polarity) +
        " " + combined_df['overall_subjectivity'].
        apply(categorize_subjectivity))

    logging.info("Appending NLP locations to text...")
    combined_df['nlp_textloc'] = (
        combined_df['nlp_text'] + ' '
        + combined_df['nlp_location'].apply(lambda x: ' '.join(x)))

    # Load NLP model globally to avoid repeated loading
    nlp = spacy.load("en_core_web_sm")

    # Load unique locations dataset once
    df_unique_locations = pd.read_csv("data/unique_locations.csv")

    # Filter out locations marked to ignore
    df_unique_locations = (
        df_unique_locations[df_unique_locations["ignore"] != 1])

    # Convert locations to a set for fast lookup
    unique_locations_set = set(df_unique_locations["location"].str.lower())

    # Initialize FlashText KeywordProcessor for fast extraction
    keyword_processor = KeywordProcessor()
    for loc in unique_locations_set:
        keyword_processor.add_keyword(loc)
    # use nlp to extract locations from the text

    def extract_locations(text):
        """Extracts location names from text using NLP and predefined list."""
        logging.debug("Starting location extraction...")
        # convert text to string
        text = str(text)
        # Use NLP to extract geographic locations (GPE entities)
        doc = nlp(text)
        nlp_locations = (
            {ent.text.lower() for ent in doc.ents if ent.label_ == "GPE"})
        logging.debug(f"NLP Extracted Locations: {nlp_locations}")

        # Fast keyword matching for known locations
        matched_locations = (
            set(keyword_processor.extract_keywords(text.lower())))
        logging.debug("Matched Locations"
                      f" from Predefined List: {matched_locations}")

        # Return matched locations if found; otherwise,
        # return NLP-extracted ones
        final_locations = (
            matched_locations if matched_locations else nlp_locations)
        logging.debug(f"Final Extracted Locations: {final_locations}")

        return list(final_locations)

    logging.info("Extracting locations from articles...")
    combined_df['locationsfromarticle'] = (
        combined_df['nlp_textloc'].progressapply(extract_locations))

    logging.info("Saving combined data post location to CSV...")
    save_dataframe_to_zip(combined_df,
                          'data/combined_data_step2.zip',
                          'combined_data_step2.csv')

    logging.info("filling nulls with default values...")
    combined_df['subject'] = combined_df['subject'].fillna('Unknown')

    logging.info("Dropping unnecessary columns...")
    combined_df.drop([
                      'nlp_textloc',
                      'source',
                      'location',
                      'nlp_location'], axis=1, inplace=True)

    logging.info("Dropping rows with empty cleaned_text...")
    combined_df = combined_df.dropna(subset=['cleaned_text'])
    logging.info("Number of rows in combined dataframe"
                 f" after dropping empty cleaned_text: {combined_df.shape[0]}")

    logging.info("Saving combined data to CSV...")
    save_dataframe_to_zip(combined_df,
                          'data/combined_data_step1.zip',
                          'combined_data_step1.csv')

    logging.info("Data pipeline completed.")
    return combined_df


# Generate Combined Data and save as a csv file
useprecombineddata = False
usepostnlpdata = True
usesavedfile = False
usegeoapi = False
useworldcitiesdata = False
findcommonthemes = False

checkdirectory()

logging.info("Starting data pipeline...")
if usesavedfile is True:
    try:
        logging.info("Loading saved data...")
        try:
            combined_df = pd.read_csv('data/combined_data_step1.zip')
        except (FileNotFoundError, ImportError):
            logging.info("Saved data not found. Generating new data...")
            combined_df = data_pipeline(useprecombineddata=True,
                                        usepostnlpdata=True)
    except Exception as e:
        logging.error(f"Error loading saved data: {e}")
        logging.info("Generating new data...")
        combined_df = data_pipeline(useprecombineddata=True,
                                    usepostnlpdata=False)
else:
    combined_df = data_pipeline(useprecombineddata, usepostnlpdata)

# check for any blank values
logging.debug(combined_df.isnull().sum())
logging.debug(combined_df.head(5))
# rename index to article_id if column article_id does not exist
if 'article_id' not in combined_df.columns:
    combined_df.index.name = 'index'
    combined_df['article_id'] = combined_df.index
# split locationsfromarticle into separate datafram
# create a referenced list based of locationsfromarticle with an
# entry for each location in the list
# and a reference to the index of the article
logging.info("Splitting locationsfromarticle into separate dataframe...")
df_locations = combined_df[['article_id',
                            'locationsfromarticle']].copy()
logging.debug(df_locations.head(5))
df_locations['locationsfromarticle'] = (
    df_locations['locationsfromarticle'].apply(string_to_list))
df_locations = (
    df_locations.explode('locationsfromarticle')
    .rename(columns={'locationsfromarticle': 'location'})
)
logging.debug(df_locations.head(10))
# summarise the locationsfromarticle data by article_id and location adding
# a count of the number of times the location appears in the article
logging.info("Summarizing locationsfromarticle data...")
df_locations_sum = (
    df_locations.groupby(['article_id',
                          'location'])
    .size()
    .reset_index(name='count'))
logging.debug(df_locations_sum.head(10))
# create a dataframe of unique locations
df_unique_locations = (
    pd.DataFrame({'location': df_locations['location'].unique()}))
logging.debug(df_unique_locations.shape)


def find_location_match(location, worldcities_df):
    """
    Search for a location in multiple columns of worldcities_df.

    Args:
        location (str): The location to search for.
        worldcities_df (DataFrame): The dataframe containing city data.

    Returns:
        dict: A dictionary containing the matched value, the column it was
        found in, latitude, longitude, and country.
              Returns None if no match is found.

    Example usage:
        location_to_search = "New York"
        result = find_location_match(location_to_search, worldcities_df)
        if result:
            print("Match found:", result)
        else:
            print("No match found.")
    """
    search_columns = ['city', 'city_ascii', 'country',
                      'iso2', 'iso3', 'admin_name']

    # Convert input to string and lowercase for case-insensitive comparison
    location = str(location).strip().lower()

    for col in search_columns:
        # Find rows where the location matches the column
        match = worldcities_df[worldcities_df[col]
                               .astype(str)
                               .str.strip()
                               .str.lower() == location]

        if not match.empty:
            # Extract the first match found
            result = {
                'matched_value': match.iloc[0][col],
                'matched_column': col,
                'latitude': match.iloc[0]['lat'],
                'longitude': match.iloc[0]['lng'],
                'country': match.iloc[0]['country']
            }
            return result
    return None  # Return None if no match is found


if usegeoapi:
    logging.info("Using geolocation API...")
    # generate a list of rows with missing data and without the ignore flag = 1
    missing_geolocation_info = (
        df_unique_locations[
            df_unique_locations['geolocation_info'].isnull()
        ].query('ignore != 1')
        )
    # check the number of missing geolocation_info
    print(missing_geolocation_info.shape)
    # Apply geolocation info extraction
    missing_geolocation_info['geolocation_info'] = (
        missing_geolocation_info['location']
        .apply(get_geolocation_info))
    # Extract latitude, longitude, and address
    missing_geolocation_info['latitude'] = (
        missing_geolocation_info['geolocation_info']
        .apply(lambda x: x['latitude']))
    missing_geolocation_info['longitude'] = (
        missing_geolocation_info['geolocation_info']
        .apply(lambda x: x['longitude']))
    missing_geolocation_info['address'] = (
        missing_geolocation_info['geolocation_info']
        .apply(lambda x: x['address']))
    # Extract continent, country, and state
    missing_geolocation_info[['continent', 'country', 'state']] = (
        missing_geolocation_info['address'].apply(
            lambda x: pd.Series(extract_geolocation_details(x))
            ))
else:
    if useworldcitiesdata:
        logging.info("Using worldcities data...")
        # add a column which classifies the location
        # as a city, country or other
        zip_file_path = '2_source_data/simplemaps_worldcities_basicv1.77.zip'
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            with z.open('worldcities.csv') as f:
                worldcities_df = pd.read_csv(f)
        print(worldcities_df.head(5))
        # change city, city_ascii, country, iso2, iso3, admin_name to lowercase
        worldcities_df['city'] = worldcities_df['city'].str.lower()
        worldcities_df['city_ascii'] = worldcities_df['city_ascii'].str.lower()
        worldcities_df['country'] = worldcities_df['country'].str.lower()
        worldcities_df['iso2'] = worldcities_df['iso2'].str.lower()
        worldcities_df['iso3'] = worldcities_df['iso3'].str.lower()
        worldcities_df['admin_name'] = worldcities_df['admin_name'].str.lower()
        print(worldcities_df.head(5))
        # change the location column to lowercase
        df_unique_locations['location'] = (
            df_unique_locations['location'].str.lower())
        # update unique_locations country, state, city, latitude, longitude
        # with the information from worldcities data
        df_unique_locations['geolocation_info'] = (
            df_unique_locations['location']
            .apply(lambda x: find_location_match(x, worldcities_df))
            )
        # Extract latitude, longitude, and address
        df_unique_locations['latitude'] = (
            df_unique_locations['geolocation_info']
            .apply(lambda x: x['latitude'] if x else None))
        df_unique_locations['longitude'] = (
            df_unique_locations['geolocation_info']
            .apply(lambda x: x['longitude'] if x else None))
        df_unique_locations['address'] = (
            df_unique_locations['geolocation_info']
            .apply(lambda x: x['matched_value'] if x else None))
        # Extract continent, country, and state

        # merge the worldcities data with the unique
        # locations data on location to city
        # # Drop the temporary 'geolocation_info' and 'address' columns
        # df_unique_locations.drop(columns=['geolocation_info', 'address'],
        #                          inplace=True)
        # # update df_locations with the information from df_unique_locations
        # df_locations = pd.merge(df_locations,
        #                         df_unique_locations,
        #                         on='location',
        #                         how='left')
        # save the locationsfromarticle data to a csv file

        # save_dataframe_to_zip(df_locations,
        #                     'data/locationsfromarticle.zip',
        #                     'locationsfromarticle.csv')
        # # save the unique locations data to a csv file
        # save_dataframe_to_zip(df_unique_locations,
        #                     'data/unique_locations.zip',
        #                     'unique_locations.csv')


logging.info("Data pipeline completed.")
logging.info("Data cleaning and feature extraction completed.")

if findcommonthemes:
    logging.info("Finding common themes")
    # using NLP produce a li    st of common themes
    # create a list of common themes
    common_themes = []
    # iterate over the nlp_text column
    for text in combined_df['nlp_text']:
        # create a doc object
        doc = nlp(text)
        # iterate over the entities in the doc
        for ent in doc.ents:
            # if the entity is a common noun
            if ent.label_ == 'NOUN':
                # append the entity to the common_themes list
                common_themes.append(ent.text)
    # create a dataframe of common themes
    df_common_themes = pd.DataFrame(common_themes, columns=['theme'])
    # create a count of the number of times each theme appears
    df_common_themes = df_common_themes['theme'].value_counts().reset_index()
    # rename the columns
    df_common_themes.columns = ['theme', 'count']
    # save the common themes data to a csv file
    save_dataframe_to_zip(df_common_themes,
                          'data/common_themes.zip',
                          'common_themes.csv')


# if no blank or null values in title, article_text, date, label,
# subject then export to csv
# set month, day and year for null values based off date_clean


# drop unnecessary columns
combined_df.drop(['article_text',
                  'subject',
                  ], axis=1, inplace=True)


if combined_df.isnull().sum().sum() == 0:
    logging.info("No missing values in the data")
    save_dataframe_to_zip(combined_df,
                          'data/combined_data.zip',
                          'combined_data.csv')
    logging.info("Data cleaned and saved as combined_data.csv in data folder")
else:
    logging.info("There are still missing values in the data")
    save_dataframe_to_zip(combined_df,
                          'data/combined_data.zip',
                          'combined_data.csv')
    logging.info("Data cleaned and saved as combined_data.csv in data folder")
    logging.info("Data cleaning and feature extraction completed.")

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
# check directory structure
import os
import ast


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
                     " current directory, please check the current directory")
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
    """Saves a pandas DataFrame to a zipped CSV file.

    Args:
        df: The pandas DataFrame to save.
        zip_filename: The name of the zip file to create.
        csv_filename: The name of the CSV file inside the zip archive.
    """

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


def dataload():
    # load the data
    fake_df = pd.read_csv("2_source_data/fake.csv.zip")
    true_df = pd.read_csv("2_source_data/true.csv.zip")
    return fake_df, true_df


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
    if pd.isna(text):  # Handle NaN cases safely
        return 'text'
    for key, value in media_dict.items():
        # Lowercase to ensure case insensitivity
        if any(word in text.lower() for word in value):
            return key
    return 'text'


def identify_media(text):
    """Identifies media types and social media platforms in text.
    Apply the function
    df_media = df["text"].apply(lambda x: pd.Series(identify_media(x)))
    df = pd.concat([df, df_media], axis=1)
    """
    media_dict = {'video': ['video', 'watch', 'live',
                            'stream', 'youtube',
                            'vimeo', 'twitch'],
                  'audio': ['audio', 'listen', 'podcast', 'radio'],
                  'image': ['image', 'photo', 'picture', 'gif'],
                  'infographic': ['infographic'],
                  'poll': ['poll'],
                  'twitter': ['twitter', 'X', 'x', 'tweet', 'retweeted'],
                  'facebook': ['facebook', 'fb'],
                  'instagram': ['instagram', 'ig', ],
                  'linkedin': ['linkedin'],
                  'wordpress': ['wordpress'],
                  'tumblr': ['tumblr'],
                  }
    # create media dictionary based of toplevel options in media_dict
    media = {key: False for key in media_dict.keys()}

    if isinstance(text, str):
        text_lower = text.lower()

        for key, value in media_dict.items():
            if any(word in text_lower for word in value):
                media[key] = True
                if key in ['twitter', 'facebook', 'instagram', 'linkedin']:
                    media[key] = True

        blob = TextBlob(text)
        media['polarity'] = blob.sentiment.polarity
        media['subjectivity'] = blob.sentiment.subjectivity
    else:
        media['polarity'] = None
        media['subjectivity'] = None

    return media


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
        if row['year'] and row['month'] and row['day']
        else None, axis=1
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


def extract_locations(text):
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return locations


def string_to_list(location_str):
    """Safely converts a string representation of a list to a list."""
    try:
        return ast.literal_eval(location_str)
    except (ValueError, SyntaxError, TypeError):
        return []


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
def data_pipeline():
    #  Pipeline to transform the data
    fake_df, true_df = dataload()
    # Combine the dataframes
    combined_df = classify_and_combine(true_df, fake_df)
    print(combined_df.columns[combined_df.columns.duplicated()])
    # ensure index is unique
    combined_df.reset_index(drop=True, inplace=True)
    # rename index to article_id
    combined_df['article_id'] = combined_df.index
    # ensure index is labbelled index
    combined_df.index.name = 'index'
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
        combined_df['nlp_text'].apply(lambda x: pd.Series(get_sentiment(x))))
    combined_df[['title_polarity', 'title_subjectivity']] = (
        combined_df['nlp_title'].apply(lambda x: pd.Series(get_sentiment(x))))
    combined_df['overall_polarity'] = (
        (combined_df['article_polarity'] + combined_df['title_polarity']) / 2)
    combined_df['overall_subjectivity'] = (
        (combined_df['article_subjectivity'] +
         combined_df['title_subjectivity']) / 2)
    # code to measure contradiction between title and article polarity
    combined_df['contradiction_polarity'] = (
        combined_df['article_polarity'] - combined_df['title_polarity'])
    combined_df['contradiction_subjectivity'] = (
        combined_df['article_subjectivity'] -
        combined_df['title_subjectivity'])
    combined_df['polarity_variations'] = (
        combined_df['contradiction_polarity'] /
        combined_df['title_polarity'])
    combined_df['subjectivity_variations'] = (
        combined_df['contradiction_subjectivity'] /
        combined_df['title_subjectivity'])
    # code to apply categorize_sentiment to article_polarity
    # and article_subjectivity and return the result
    combined_df['sentiment_article'] = (
        combined_df['article_polarity'].apply(categorize_polarity) + " " +
        combined_df['article_subjectivity'].apply(categorize_subjectivity))
    combined_df['sentiment_title'] = ((
        combined_df['title_polarity'].apply(categorize_polarity)) +
        (combined_df['title_subjectivity'].apply(categorize_subjectivity)))
    combined_df['sentiment_overall'] = ((
        combined_df['overall_polarity'].apply(categorize_polarity)) +
        (combined_df['overall_subjectivity'].apply(categorize_subjectivity)))
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
                      'subject',
                      'source',
                      'location',
                      'nlp_location',
                      ], axis=1, inplace=True)
    # drop any rows where cleaned_text is empty
    combined_df = combined_df.dropna(subset=['cleaned_text'])
    # save the combined data to a csv file
    save_dataframe_to_zip(combined_df,
                          'data/combined_data_step1.zip',
                          'combined_data_step1.csv')
    return combined_df


# Generate Combined Data and save as a csv file
usesavedfile = True
usegeoapi = True
if usesavedfile is False:
    combined_df = data_pipeline()
else:
    combined_df = pd.read_csv('data/combined_data_step1.zip')
# check for any blank values
print(combined_df.isnull().sum())
print(combined_df.head(5))
# rename index to article_id if column article_id does not exist
if 'article_id' not in combined_df.columns:
    combined_df.index.name = 'index'
    combined_df['article_id'] = combined_df.index
# split locationsfromarticle into separate datafram
# create a referenced list based of locationsfromarticle with an
# entry for each location in the list
# and a reference to the index of the article
df_locations = combined_df[['article_id',
                            'locationsfromarticle']].copy()
print(df_locations.head(5))
df_locations['locationsfromarticle'] = (
    df_locations['locationsfromarticle'].apply(string_to_list))
df_locations = (
    df_locations.explode('locationsfromarticle')
    .rename(columns={'locationsfromarticle': 'location'})
)
print(df_locations.head(10))
# summarise the locationsfromarticle data by article_id and location adding
# a count of the number of times the location appears in the article
df_locations_sum = (
    df_locations.groupby(['article_id',
                          'location'])
    .size()
    .reset_index(name='count'))
print(df_locations_sum.head(10))
# create a dataframe of unique locations
df_unique_locations = (
    pd.DataFrame({'location': df_locations['location'].unique()}))
print(df_unique_locations.shape)

# Use unique_locations.zip to prepopulate the geolocation_info
# and address columns
# Load the unique locations data
df_unique_locations_existing = pd.read_csv('data/unique_locations.zip')
# merge the existing unique locations data with the new unique locations data
df_unique_locations = pd.concat([df_unique_locations_existing,
                                    df_unique_locations]).drop_duplicates()
# generate a list of rows with missing data and without the ignore flag = 1
missing_geolocation_info = (
    df_unique_locations[df_unique_locations['geolocation_info'].isnull()]
    .query('ignore != 1')
)
# check the number of missing geolocation_info
print(missing_geolocation_info.shape)


if usegeoapi:
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
    print("Using worldcities data")

# add a column which classifies the location as a city, country or other
with zipfile.ZipFile('2_source_data/simplemaps_worldcities_basicv1.77.zip',
'r') as z:
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
df_unique_locations['location'] = df_unique_locations['location'].str.lower()
# merge the worldcities data with the unique
# locations data on location to city
import pandas as pd

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

# Example usage:
# Assuming worldcities_df is already loaded
location_to_search = "New York"
result = find_location_match(location_to_search, worldcities_df)

if result:
    print("Match found:", result)
else:
    print("No match found.")




# # Drop the temporary 'geolocation_info' and 'address' columns
# df_unique_locations.drop(columns=['geolocation_info', 'address'],
#                          inplace=True)
# # update df_locations with the information from df_unique_locations
# df_locations = pd.merge(df_locations,
#                         df_unique_locations,
#                         on='location',
#                         how='left')
# save the locationsfromarticle data to a csv file
save_dataframe_to_zip(df_locations,
                      'data/locationsfromarticle.zip',
                      'locationsfromarticle.csv')
# save the unique locations data to a csv file
save_dataframe_to_zip(df_unique_locations,
                      'data/unique_locations.zip',
                      'unique_locations.csv')

findcommonthemes = False
if findcommonthemes:
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
if combined_df.isnull().sum().sum() == 0:
    save_dataframe_to_zip(combined_df,
                          'data/combined_data.zip',
                          'combined_data.csv')
    print("Data cleaned and saved as combined_data.csv in data folder")
else:
    print("There are still missing values in the data")
    save_dataframe_to_zip(combined_df,
                          'data/combined_data.zip',
                          'combined_data.csv')

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

def mapdata():
    df = pd.read_csv("data//combined_data.zip",
                    usecols={
        'index',
        'title',
        'label',
        'month',
        'day',
        'year',
        'date_clean',
        'article_id',
                    },  
                    dtype = {
        'index': 'int64',
        'title': 'object',
        'label': 'int64',
        'month': 'float64',
        'day': 'float64',
        'year': 'int64',
        'date': 'object',
        'article_id': 'int64',
        },
                    compression='zip')
    print("The shape of the data is: ", df.shape)
    print(df.info())
    df.head()
    # filter out rows with dates earlier than 2014-01-01
    df['date'] = pd.to_datetime(df['date_clean'])
    df = df[df['date'] >= '2014-01-01']
    print("The shape of the data is: ", df.shape)
    df.head()

    # load locationsfromarticle.zip file
    locationsfromarticle = (
        pd.read_csv("data//locationsfromarticle.zip",
                    usecols={
                        'article_id',
                        'location',
                    },
                    dtype={
                        'article_id': 'int64',
                        'location': 'object',
                    },
                    compression='zip')
                    )
    print(locationsfromarticle.info())
    locationsfromarticle.head()
    # load unique_locations.csv file
    locations = pd.read_csv("data//unique_locations.csv",
                            dtype={
                                'location': 'object',
                                'latitude': 'float64',
                                'longitude': 'float64',
                                'state': 'object',
                                'country': 'object',
                                'continent': 'object',
                                'subcontinent': 'object',
                                'ignore': 'int64'
                                },
                                )
    print(locations.info())
    locations.head()

    # rename Continent to continent and Sub Continent to subcontinent
    # locations = locations.rename(columns={'Continent': 'continent', 'Sub Continent': 'subcontinent'}, inplace=True)
    print(locations.info())
    locations.head()

    # set all null subcontinent values to continent value
    locations['subcontinent'] = locations['subcontinent'].fillna(locations['continent'])
    # set null country values to subcontinent value
    locations['country'] = locations['country'].fillna(locations['subcontinent'])
    # set null state values to country value
    locations['state'] = locations['state'].fillna(locations['country'])

    # match locations to locationsfromarticle
    locations['location'] = locations['location'].str.lower()
    locationsfromarticle['location'] = locationsfromarticle['location'].str.lower()
    locationsmerged = locations.merge(locationsfromarticle, on='location', how='left')
    print(locationsmerged.info())
    locationsmerged.head()
    del locations, locationsfromarticle

    # drop all rows with ignore = 1
    locationsmerged = locationsmerged[locationsmerged['ignore'] != 1]
    locationsmerged = locationsmerged.drop(columns=['ignore'])
    locationsmerged.head()
    locationsmerged.info()

    # merge locationsmerged with df only keep rows with a match
    locationgraphdf = df.merge(locationsmerged, on='article_id', how='left')
    print(locationgraphdf.info())
    locationgraphdf.head()

    # create a new dataframe with the number of fake articles per country, continent, and subcontinent
    fakearticles = locationgraphdf[locationgraphdf['label'] == 1]
    fakearticles = fakearticles.groupby(['year', 'month', 'day','date',
                                         'state', 'country', 'continent',
                                         'subcontinent']).size().reset_index(name='fake_count')

    # create a new dataframe with the number of real articles per country, continent, and subcontinent
    realarticles = locationgraphdf[locationgraphdf['label'] == 0]
    realarticles = realarticles.groupby(['year', 'month', 'day','date',
                                         'state', 'country', 'continent',
                                         'subcontinent']).size().reset_index(name='real_count')

    # merge fake and real articles dataframes
    articles = pd.merge(fakearticles, realarticles, on=['year', 'month', 'day',
                                                        'date', 'state',
                                                        'country',
                                                        'continent',
                                                        'subcontinent'],
                        how='outer').fillna(0)
    articles = articles.sort_values(by=['year',
                                        'month',
                                        'day',
                                        'date',
                                        'state',
                                        'country',
                                        'continent',
                                        'subcontinent'])
    articles.head()

    # save aticles dataframe as data//articlesformap.csv
    articles.to_csv("data//articlesformap.csv", index=False)

    return articles

# code to run page if called directly
if __name__ == '__main__':
    mapdata()

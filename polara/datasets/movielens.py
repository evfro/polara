from io import BytesIO
import numpy as np
import pandas as pd
import urllib
try:
    from pandas.io.common import ZipFile
except ImportError:
    from zipfile import ZipFile


def get_movielens_data(local_file=None, get_ratings=True, get_genres=False,
                       split_genres=True, mdb_mapping=False, get_tags=False, include_time=False):
    '''Downloads movielens data and stores it in pandas dataframe.
    '''
    fields = ['userid', 'movieid', 'rating']

    if include_time:
        fields.append('timestamp')

    if not local_file:
        # downloading data
        zip_file_url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        with urllib.request.urlopen(zip_file_url) as zip_response:
            zip_contents = BytesIO(zip_response.read())
    else:
        zip_contents = local_file

    ml_data = ml_genres = ml_tags = mapping = None
    # loading data into memory
    with ZipFile(zip_contents) as zfile:
        zip_files = pd.Series(zfile.namelist())
        zip_file = zip_files[zip_files.str.contains('ratings')].iat[0]
        is_new_format = ('latest' in zip_file) or ('20m' in zip_file) or ('25m' in zip_file)
        delimiter = ','
        header = 0 if is_new_format else None
        if get_ratings:
            zdata = zfile.read(zip_file)
            zdata = zdata.replace(b'::', delimiter.encode())
            # makes data compatible with pandas c-engine
            # returns string objects instead of bytes in that case
            ml_data = pd.read_csv(BytesIO(zdata), sep=delimiter, header=header, engine='c', names=fields, usecols=fields)

        if get_genres:
            zip_file = zip_files[zip_files.str.contains('movies')].iat[0]
            zdata =  zfile.read(zip_file)
            if not is_new_format:
                # make data compatible with pandas c-engine
                # pandas returns string objects instead of bytes in that case
                delimiter = '^'
                zdata = zdata.replace(b'::', delimiter.encode())
            genres_data = pd.read_csv(BytesIO(zdata), sep=delimiter, header=header,
                                      engine='c', encoding='unicode_escape',
                                      names=['movieid', 'movienm', 'genres'])

            ml_genres = get_split_genres(genres_data) if split_genres else genres_data

        if get_tags:
            zip_file = zip_files[zip_files.str.contains('/tags')].iat[0] #not genome
            zdata =  zfile.read(zip_file)
            if not is_new_format:
                # make data compatible with pandas c-engine
                # pandas returns string objects instead of bytes in that case
                delimiter = '^'
                zdata = zdata.replace(b'::', delimiter.encode())
            fields[2] = 'tag'
            ml_tags = pd.read_csv(BytesIO(zdata), sep=delimiter, header=header,
                                      engine='c', encoding='latin1',
                                      names=fields, usecols=range(len(fields)))

        if mdb_mapping and is_new_format:
            # imdb and tmdb mapping - exists only in ml-latest or 20m datasets
            zip_file = zip_files[zip_files.str.contains('links')].iat[0]
            with zfile.open(zip_file) as zdata:
                mapping = pd.read_csv(zdata, sep=',', header=0, engine='c',
                                        names=['movieid', 'imdbid', 'tmdbid'])

    res = [data for data in [ml_data, ml_genres, ml_tags, mapping] if data is not None]
    return res[0] if len(res)==1 else res


def get_split_genres(genres_data):
    return (genres_data[['movieid', 'movienm']]
            .join(pd.DataFrame([(i, x)
                                for i, g in enumerate(genres_data['genres'])
                                for x in g.split('|')
                               ], columns=['index', 'genreid']
                              ).set_index('index'))
            .reset_index(drop=True))



def filter_short_head(data, threshold=0.01):
    short_head = data.groupby('movieid', sort=False)['userid'].nunique()
    short_head.sort_values(ascending=False, inplace=True)

    ratings_perc = short_head.cumsum()*1.0/short_head.sum()
    movies_perc = np.arange(1, len(short_head)+1, dtype='f8') / len(short_head)

    long_tail_movies = ratings_perc[movies_perc > threshold].index
    return long_tail_movies

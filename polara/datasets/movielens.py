from io import BytesIO
import pandas as pd

try:
    from pandas.io.common import ZipFile
except ImportError:
    from zipfile import ZipFile

def get_movielens_data(local_file=None, get_ratings=True, get_genres=False,
                        split_genres=True, mdb_mapping=False):
    '''Downloads movielens data and stores it in pandas dataframe.
    '''
    if not local_file:
        # downloading data
        from requests import get
        zip_file_url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        zip_response = get(zip_file_url)
        zip_contents = BytesIO(zip_response.content)
    else:
        zip_contents = local_file

    ml_data = ml_genres = mapping = None
    # loading data into memory
    with ZipFile(zip_contents) as zfile:
        zip_files = pd.Series(zfile.namelist())
        zip_file = zip_files[zip_files.str.contains('ratings')].iat[0]
        is_latest = 'latest' in zip_file
        header = 0 if is_latest else None
        if get_ratings:
            zdata = zfile.read(zip_file)
            delimiter = ','
            zdata = zdata.replace(b'::', delimiter.encode()) # makes data compatible with pandas c-engine
            ml_data = pd.read_csv(BytesIO(zdata), sep=delimiter, header=header, engine='c',
                                    names=['userid', 'movieid', 'rating', 'timestamp'],
                                    usecols=['userid', 'movieid', 'rating'])

        if get_genres:
            zip_file = zip_files[zip_files.str.contains('movies')].iat[0]
            with zfile.open(zip_file) as zdata:
                delimiter = ',' if is_latest else '::'
                genres_data = pd.read_csv(zdata, sep=delimiter, header=header, engine='python',
                                            names=['movieid', 'movienm', 'genres'])

            ml_genres = get_split_genres(genres_data) if split_genres else genres_data

        if is_latest and mdb_mapping:
            # imdb and tmdb mapping - exists only in ml-latest datasets
            zip_file = zip_files[zip_files.str.contains('links')].iat[0]
            with zfile.open(zip_file) as zdata:
                mapping = pd.read_csv(zdata, sep=',', header=0, engine='c',
                                        names=['movieid', 'imdbid', 'tmdbid'])

    res = [data for data in [ml_data, ml_genres, mapping] if data is not None]
    if len(res)==1: res = res[0]
    return res


def get_split_genres(genres_data):
    genres_data.index.name = 'movie_idx'
    genres_stacked = genres_data.genres.str.split('|', expand=True).stack().to_frame('genreid')
    ml_genres = genres_data[['movieid', 'movienm']].join(genres_stacked).reset_index(drop=True)
    return ml_genres


def filter_short_head(data, threshold=0.01):
    short_head = data.groupby('movieid', sort=False)['userid'].nunique()
    short_head.sort_values(ascending=False, inplace=True)

    ratings_perc = short_head.cumsum()*1.0/short_head.sum()
    movies_perc = pd.np.arange(1, len(short_head)+1, dtype=pd.np.float64) / len(short_head)

    long_tail_movies = ratings_perc[movies_perc > threshold].index
    return long_tail_movies

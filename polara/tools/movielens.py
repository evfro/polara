import pandas as pd
from requests import get
from StringIO import StringIO
from pandas.io.common import ZipFile


def get_movielens_data(local_file=None, get_genres=False):
    '''Downloads movielens data and stores it in pandas dataframe.
    '''
    if not local_file:
        #print 'Downloading data...'
        zip_file_url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        zip_response = get(zip_file_url)
        zip_contents = StringIO(zip_response.content)
        #print 'Done.'
    else:
        zip_contents = local_file

    #print 'Loading data into memory...'
    with ZipFile(zip_contents) as zfile:
        zip_files = pd.Series(zfile.namelist())
        zip_file = zip_files[zip_files.str.contains('ratings')].iat[0]
        zdata = zfile.read(zip_file)
        if 'latest' in zip_file:
            header = 0
        else:
            header = None
        delimiter = ','
        zdata = zdata.replace('::', delimiter) # makes data compatible with pandas c-engine
        ml_data = pd.read_csv(StringIO(zdata), sep=delimiter, header=header, engine='c',
                                names=['userid', 'movieid', 'rating', 'timestamp'],
                                usecols=['userid', 'movieid', 'rating'])

        if get_genres:
            zip_file = zip_files[zip_files.str.contains('movies')].iat[0]
            with zfile.open(zip_file) as zdata:
                if 'latest' in zip_file:
                    delimiter = ','
                else:
                    delimiter = '::'
                genres_data = pd.read_csv(zdata, sep=delimiter, header=header, engine='python',
                                            names=['movieid', 'movienm', 'genres'])

            ml_genres = split_genres(genres_data)
            ml_data = (ml_data, ml_genres)

    return ml_data


def split_genres(genres_data):
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

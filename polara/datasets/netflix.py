import tarfile

import pandas as pd


def get_netflix_data(gz_file):
    movie_data = []
    movie_name = []
    with tarfile.open(gz_file) as tar:
        training_data = tar.getmember('download/training_set.tar')
        with tarfile.open(fileobj=tar.extractfile(training_data)) as inner:
            for item in inner.getmembers():
                if item.isfile():
                    f = inner.extractfile(item.name)
                    df = pd.read_csv(f)
                    movieid = df.columns[0]
                    movie_name.append(movieid)
                    movie_data.append(df[movieid])

    data = pd.concat(movie_data, keys=movie_name)
    data = data.reset_index().iloc[:, :3].rename(columns={'level_0': 'movieid',
                                                          'level_1': 'userid',
                                                          'level_2': 'rating'})
    return data

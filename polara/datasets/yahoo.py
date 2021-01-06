import tarfile
import pandas as pd

def get_yahoo_music_data(path=None, fileid=0, include_test=True, read_attributes=False, read_genres=False):
    res = []
    if path:
        data_folder = 'ydata-ymusic-user-song-ratings-meta-v1_0'
        col_names = ['userid', 'songid', 'rating']
        with tarfile.open(path, 'r:gz') as tar:
            handle = tar.getmember(f'{data_folder}/train_{fileid}.txt')
            file = tar.extractfile(handle)
            data = pd.read_csv(file, sep='\t', header=None, names=col_names)
            res.append(data)
            if include_test:
                handle = tar.getmember(f'{data_folder}/test_{fileid}.txt')
                file = tar.extractfile(handle)
                data = pd.read_csv(file, sep='\t', header=None, names=col_names)
                res.append(data)

            if read_attributes:
                handle = tar.getmember(f'{data_folder}/song-attributes.txt')
                file = tar.extractfile(handle)
                attr = pd.read_csv(file, sep='\t', header=None, index_col=0,
                                   names=['songid', 'albumid', 'artistid', 'genreid'])
                res.append(attr)

            if read_genres:
                handle = tar.getmember(f'{data_folder}/song-attributes.txt')
                file = tar.extractfile(handle)
                genres = pd.read_csv(file, sep='\t', header=None, index_col=0,
                                     names=['genreid', 'parent_genre', 'level', 'genre_name'])
                res.append(genres)
    if len(res) == 1:
        res = res[0]
    return res

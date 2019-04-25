import pandas as pd
import tarfile


def get_netflix_data(gz_file, get_ratings=True, get_probe=False):
    movie_data = []
    movie_inds = []
    with tarfile.open(gz_file) as tar:
        if get_ratings:
            training_data = tar.getmember('download/training_set.tar')
            # maybe try with threads, e.g.
            # https://stackoverflow.com/questions/43727520/speed-up-json-to-dataframe-w-a-lot-of-data-manipulation
            with tarfile.open(fileobj=tar.extractfile(training_data)) as inner:
                for item in inner.getmembers():
                    if item.isfile():
                        f = inner.extractfile(item.name)
                        df = pd.read_csv(f)
                        movieid = df.columns[0]
                        movie_inds.append(int(movieid[:-1]))
                        movie_data.append(df[movieid])

        if get_probe:
            probe_data = tar.getmember('download/probe.txt')
            probe_file = tar.extractfile(probe_data)
            probe = []
            for line in probe_file:
                line = line.strip()
                if line.endswith(b':'):
                    movieid = int(line[:-1])
                else:
                    userid = int(line)
                    probe.append((movieid, userid))

    data = None
    if movie_data:
        data = pd.concat(movie_data, keys=movie_inds)
        data = data.reset_index().iloc[:, :3].rename(columns={'level_0': 'movieid',
                                                              'level_1': 'userid',
                                                              'level_2': 'rating'})
    if get_probe:
        probe = pd.DataFrame.from_records(probe, columns=['movieid', 'userid'])
        if data is not None:
            data = (data, probe)
        else:
            data = probe
    return data

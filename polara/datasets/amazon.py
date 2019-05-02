from ast import literal_eval
import gzip
import pandas as pd


def parse_meta(path):
    with gzip.open(path, 'rt') as gz:
        for line in gz:
            yield literal_eval(line)


def get_amazon_data(path=None, meta_path=None, nrows=None):
    res = []
    if path:
        data = pd.read_csv(path, header=None,
                           names=['userid', 'asin', 'rating', 'timestamp'],
                           usecols=['userid', 'asin', 'rating'],
                           nrows=nrows)
        res.append(data)
    if meta_path:
        meta = pd.DataFrame.from_records(parse_meta(meta_path), nrows=nrows)
        res.append(meta)
    if len(res) == 1:
        res = res[0]
    return res

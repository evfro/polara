from io import BytesIO
import pandas as pd

try:
    from pandas.io.common import ZipFile
except ImportError:
    from zipfile import ZipFile


def get_bx_data(local_file=None, get_ratings=True, get_users=False, get_books=False):
    if not local_file:
        # downloading data
        from requests import get
        zip_file_url = 'http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip'
        zip_response = get(zip_file_url)
        zip_contents = BytesIO(zip_response.content)
    else:
        zip_contents = local_file

    ratings = users = books = None

    with ZipFile(zip_contents) as zfile:
        zip_files = pd.Series(zfile.namelist())
        zip_file = zip_files[zip_files.str.contains('ratings', flags=2)].iat[0]

        delimiter = ';'
        if get_ratings:
            zdata = zfile.read(zip_file)
            ratings = pd.read_csv(BytesIO(zdata), sep=delimiter, header=0,
                                  engine='c', encoding='unicode_escape')

        if get_users:
            zip_file = zip_files[zip_files.str.contains('users', flags=2)].iat[0]
            with zfile.open(zip_file) as zdata:
                users = pd.read_csv(zdata, sep=delimiter, header=0, engine='c',
                                    encoding='unicode_escape')

        if get_books:
            zip_file = zip_files[zip_files.str.contains('books', flags=2)].iat[0]
            with zfile.open(zip_file) as zdata:
                books = pd.read_csv(zdata, sep=delimiter, header=0, engine='c',
                                    quoting=1, escapechar='\\', encoding='unicode_escape',
                                    usecols=['ISBN', 'Book-Author', 'Publisher'])

    res = [data.rename(columns=lambda x: x.lower().replace('book-', '')
                                                  .replace('-id', 'id'), copy=False)
           for data in [ratings, users, books] if data is not None]
    if len(res) == 1:
        res = res[0]
    return res

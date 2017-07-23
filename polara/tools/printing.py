from IPython.display import HTML
from contextlib import contextmanager
import sys, os

def print_frames(dataframes):
    if not isinstance(dataframes, tuple):
        return dataframes

    border_style = u'\"border: none\"'
    cells = [u'<td style={}> {} </td>'.format(border_style, df._repr_html_()) for df in dataframes]

    table = '''<table style={}>
    <tr style={}>'''.format(border_style, border_style) +\
    '\n'.join(cells)+\
    '''
    </tr>
    </table>'''

    return HTML(table)

# from http://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/#
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

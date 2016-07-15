from IPython.display import HTML

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
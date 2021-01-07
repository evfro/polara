import pytest
import pandas as pd


@pytest.fixture
def ts_data_short():
    # -------- TIMELINE -------->>
    # u1 | Matrix   . LOTR      
    # u2 |      GF  .       SW1   
    # u3 |  Matrix  .     LOTR  .  SW4    
    return pd.DataFrame([
        ('u1', 'Matrix', 0),
        ('u3', 'Matrix', 1),
        ('u2', 'GF',     2),
        ('u1', 'LOTR',   3),
        ('u3', 'LOTR',   4),
        ('u2', 'SW1',    5),
        ('u3', 'SW4',    6),
    ], columns=['userid', 'itemid', 'timestamp'])

@pytest.fixture
def ts_data_short_earliest_last_split():
    return [0, 1, 2], [3, 4, 5], [6]
from pandas.testing import assert_frame_equal
from polara.preprocessing.dataframes import split_earliest_last


def test_earliest_last_split(ts_data_short, ts_data_short_earliest_last_split):    
    observed, hidden, future = split_earliest_last(ts_data_short)
    observed_idx, hidden_idx, future_idx = ts_data_short_earliest_last_split

    assert_frame_equal(observed, ts_data_short.iloc[observed_idx], check_like=True)
    assert_frame_equal(hidden, ts_data_short.iloc[hidden_idx], check_like=True)
    assert_frame_equal(future, ts_data_short.iloc[future_idx], check_like=True)

    




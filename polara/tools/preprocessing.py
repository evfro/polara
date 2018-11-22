def filter_sessions_by_length(data, session_label='userid', min_session_length=3):
    """Filters users with insufficient number of items"""
    if data.duplicated().any():
        raise NotImplementedError

    sz = data[session_label].value_counts(sort=False)
    has_valid_session_length = sz >= min_session_length
    if not has_valid_session_length.all():
        valid_sessions = sz.index[has_valid_session_length]
        new_data = data[data[session_label].isin(valid_sessions)].copy()
        print('Sessions are filtered by length')
    else:
        new_data = data
    return new_data

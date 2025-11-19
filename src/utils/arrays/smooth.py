def moving_average(ds, window):
    try:
        return ds.rolling(time=window, center=True, min_periods=1).mean()
    except (AttributeError, KeyError):
        return None


def year_moving_average(ds, window):
    try:
        return ds.rolling(year=window, center=True, min_periods=1).mean()
    except (AttributeError, KeyError):
        return None
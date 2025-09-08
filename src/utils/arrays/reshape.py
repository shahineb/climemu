def groupby_month_and_year(ds):
    """
    Reshapes datarray time dimension to group by month and year
    """
    try:
        year = ds.time.dt.year
        month = ds.time.dt.month
        ds = ds.assign_coords(year=("time", year.data), month=("time", month.data))
        return ds.set_index(time=("year", "month")).unstack("time")
    except AttributeError:
        return None

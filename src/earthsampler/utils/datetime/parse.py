_DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_CUM_DAYS = [sum(_DAYS_PER_MONTH[:i]) for i in range(12)]


def _parse_date_str(s):
    """Parse 'dd/mm' or 'dd-mm' string into (day, month) ints."""
    for sep in ("/", "-"):
        if sep in s:
            parts = s.split(sep)
            if len(parts) != 2:
                raise ValueError(f"Expected 'dd/mm' or 'dd-mm' format, got '{s}'")
            try:
                day, month = int(parts[0]), int(parts[1])
            except ValueError:
                raise ValueError(f"Expected 'dd/mm' or 'dd-mm' format, got '{s}'")
            return day, month
    raise ValueError(f"Expected 'dd/mm' or 'dd-mm' format, got '{s}'")


def parse_doy(doy):
    """Convert doy to an integer day-of-year (1-365).

    Accepts:
        - int: day-of-year directly (1-365)
        - str: 'dd/mm' or 'dd-mm' format (no leap year — 29/02 is rejected)
    """
    if isinstance(doy, int):
        if not 1 <= doy <= 365:
            raise ValueError(f"doy must be between 1 and 365, got {doy}")
        return doy
    if isinstance(doy, str):
        day, month = _parse_date_str(doy)
        if not 1 <= month <= 12:
            raise ValueError(f"Month must be between 1 and 12, got {month}")
        if month == 2 and day == 29:
            raise ValueError("No leap year: 29/02 is not valid")
        max_day = _DAYS_PER_MONTH[month - 1]
        if not 1 <= day <= max_day:
            raise ValueError(f"Day must be between 1 and {max_day} for month {month}, got {day}")
        return _CUM_DAYS[month - 1] + day
    raise TypeError(f"doy must be int or str, got {type(doy).__name__}")

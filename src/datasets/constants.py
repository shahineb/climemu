"""Constants used in CMIP6 data processing.

This module contains constants used throughout the CMIP6 data processing pipeline.
"""

# Time conversion constants
SECONDS_PER_DAY: int = 86400  # Convert precipitation from kg/m2/s to mm/day

# CMIP6 data organization constants
TABLE_ID: str = "Amon"  # CMIP6 table identifier for monthly atmospheric data
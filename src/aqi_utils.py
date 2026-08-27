"""US EPA AQI conversion utilities.

Reference breakpoints (PM2.5, 24-hr avg, µg/m^3) -> AQI (0-500):
https://www.airnow.gov/aqi/aqi-basics/
"""

# (C_low, C_high, I_low, I_high)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def pm25_to_aqi(pm25: float) -> float:
    """Convert a PM2.5 concentration (µg/m^3) to a US EPA AQI value (0-500).

    EPA's breakpoint table is defined on PM2.5 rounded to 1 decimal place
    (e.g. 12.0 / 12.1 are adjacent breakpoints with no gap between them).
    Skipping this rounding leaves gaps like 12.01-12.09 uncovered by any
    bracket, which silently falls through to the 500 (hazardous) fallback
    for an otherwise ordinary reading — round first to avoid that.
    """
    if pm25 is None:
        return None
    pm25 = round(max(0.0, float(pm25)), 1)
    for c_low, c_high, i_low, i_high in PM25_BREAKPOINTS:
        if c_low <= pm25 <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low, 1)
    # Above breakpoint table (hazardous, off the scale) -> cap at 500
    return 500.0


def aqi_category(aqi: float) -> str:
    if aqi is None:
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"

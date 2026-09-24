import requests
import pandas as pd
from datetime import date, timedelta
from io import StringIO
import time

MAP_KEY = "4c80208f04092618f1c71426b3e823cd"

SOURCE = "VIIRS_NOAA21_NRT"

WEST = 76.0
SOUTH = 8.0
EAST = 80.5
NORTH = 13.5

DAYS_PER_REQUEST = 5
TOTAL_DAYS = 30

output = "data/raw/firms_30day.csv"

end_date = date.today()
start_date = end_date - timedelta(days=TOTAL_DAYS - 1)

all_data = []

current_date = start_date

while current_date <= end_date:

    remaining_days = (end_date - current_date).days + 1
    days = min(DAYS_PER_REQUEST, remaining_days)

    date_str = current_date.strftime("%Y-%m-%d")

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{MAP_KEY}/{SOURCE}/"
        f"{WEST},{SOUTH},{EAST},{NORTH}/"
        f"{days}/{date_str}"
    )

    print(f"Downloading {date_str} → {days} days...")

    response = requests.get(url)

    if response.status_code == 200:

        if response.text.strip():
            df = pd.read_csv(StringIO(response.text))

            print(f"  ✓ {len(df)} detections")

            all_data.append(df)

        else:
            print("  ⚠ No data")

    else:
        print(f"  ✗ ERROR {response.status_code}")
        print(response.text)

    current_date += timedelta(days=days)

    time.sleep(1)


if all_data:

    final_df = pd.concat(all_data, ignore_index=True)

    # Remove exact duplicates
    final_df = final_df.drop_duplicates()

    final_df.to_csv(output, index=False)

    print("\n==============================")
    print("30-DAY DOWNLOAD COMPLETE")
    print("==============================")
    print("Total detections:", len(final_df))
    print("Saved to:", output)
    print("Date range:", final_df["acq_date"].min(),
          "→", final_df["acq_date"].max())

else:
    print("No data downloaded.")
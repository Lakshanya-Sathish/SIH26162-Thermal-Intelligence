import requests

MAP_KEY = "4c80208f04092618f1c71426b3e823cd"

WEST = 76.0
SOUTH = 8.0
EAST = 80.5
NORTH = 13.5

SOURCE = "VIIRS_NOAA21_NRT"
DAYS = 3

url = (
    f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    f"{MAP_KEY}/{SOURCE}/"
    f"{WEST},{SOUTH},{EAST},{NORTH}/{DAYS}"
)

print("Downloading FIRMS data...")
print("Source:", SOURCE)
print("Days:", DAYS)

response = requests.get(url)

print("Status:", response.status_code)
print("Bytes received:", len(response.content))

if response.status_code == 200:
    with open("data/raw/firms.csv", "wb") as f:
        f.write(response.content)

    print("SUCCESS!")
    print("Saved to data/raw/firms.csv")
else:
    print("DOWNLOAD FAILED")
    print(response.text)
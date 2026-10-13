import ee

PROJECT_ID = "awd-satellite-analysis"

ee.Initialize(project=PROJECT_ID)

print("Earth Engine connected successfully!")

image_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

print("Sentinel-2 collection:", image_collection.size().getInfo())
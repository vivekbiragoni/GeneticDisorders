#%%
import json
import requests

# Run inference on an image
url = "https://api.ultralytics.com/v1/predict/tzooFxpdyyT5unVeu5ER"
headers = {"x-api-key": "5edfd443c033f4d159e3b4f9435b5f9bc09b55db93"}
data = {"size": 640, "confidence": 0.25, "iou": 0.45}
with open(r"C:\Users\vivek\Desktop\GD_out\test\downsyndrome\down_4.jpg", "rb") as f:
	response = requests.post(url, headers=headers, data=data, files={"image": f})

# Check for successful response
response.raise_for_status()

# Print inference results
print(json.dumps(response.json(), indent=2))
# %%

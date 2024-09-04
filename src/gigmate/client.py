import requests

image_path = "output/completed_midi.mid"
url = "http://localhost:8000/predict"

with open(image_path, 'rb') as image_file:
    files = {'request': image_file}
    response = requests.post(url, files=files)
print(response.content)
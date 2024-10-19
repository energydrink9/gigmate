import requests
import time

OUTPUT_FILE = 'output/output.wav'
API_URL = 'https://jm07w014k3g6vv-8000.proxy.runpod.net'
#API_URL = 'http://localhost:8000'
FILE_PATH = 'resources/centerpiece.ogg'


def send_request() -> None:
    url = f"{API_URL}/predict"

    with open(FILE_PATH, 'rb') as file:
        files = {'request': file}
        start_time = time.perf_counter()
        response = requests.post(url, files=files)
        end_time = time.perf_counter()

        if response.status_code == 200:
            print(f'Received response in {end_time - start_time} seconds')
            with open(OUTPUT_FILE, 'wb') as output_file:
                output_file.write(response.content)
                print(f'Saved response to file at: {OUTPUT_FILE}')


if __name__ == '__main__':
    send_request()
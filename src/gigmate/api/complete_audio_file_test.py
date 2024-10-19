import requests

OUTPUT_FILE = 'output/output.wav'
API_URL = 'https://jm07w014k3g6vv-8000.proxy.runpod.net'
FILE_PATH = 'resources/test_creep_cut.ogg'


def send_request() -> None:
    url = f"{API_URL}/predict"

    with open(FILE_PATH, 'rb') as file:
        files = {'request': file}
        response = requests.post(url, files=files)
    
        if response.status_code == 200:
            with open(OUTPUT_FILE, 'wb') as output_file:
                output_file.write(response.content)
                print(f'Saved response to file at: {OUTPUT_FILE}')


if __name__ == '__main__':
    send_request()
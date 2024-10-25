import requests
import json

url = 'http://127.0.0.1:5000/predict'
payload = {
    'text': "Congratulations! You have won a $1000 gift card. Click here to claim your prize."
}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(payload), headers=headers)

print(f'Status Code: {response.status_code}')
print(f'Response: {response.json()}')

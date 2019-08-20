import requests
import json
import pdb
data = {
   'dialogs': [
       {
           'message': "Did you do it?",
           'caller': "A"
       },
       {
           'message': "No",
           'caller': "B"
       },
       {
           'message': "Really,",
           'caller': "A"
       },
       {
           'message': "I like cross-stitch too.",
           'caller': "A"
       },
       {
           'message': "Oh, I love it.",
           'caller': "B"
       },
       {
           'message': "I just have a hard time finding any spare time lately.",
           'caller': "A"
       },
       {
           'message': "That's my case also.",
           'caller': "B"
       },
       {
           'message': "I've got a new born",
           'caller': "B"
        }
   ]
}
url_swda = 'http://127.0.0.1:8000/speech_act/swda/'
headers = {'content-type': 'application/json'}
response_swda=requests.post(url_swda, data=json.dumps(data), headers=headers)

print(response_swda.json())

url_vrm = 'http://127.0.0.1:8000/speech_act/vrm/'
headers = {'content-type': 'application/json'}
response_vrm=requests.post(url_vrm, data=json.dumps(data), headers=headers)

print(response_vrm.json())


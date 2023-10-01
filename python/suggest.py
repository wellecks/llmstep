import sys
import http.client
import json
import sys
import os
import requests

HOST = os.environ.get('LLMSTEP_HOST', 'localhost')
PORT = os.environ.get('LLMSTEP_PORT', 6000)
COLAB = os.environ.get('COLAB', '')

def suggest(tactic_state, prefix):
    conn = http.client.HTTPConnection(HOST, port=PORT)
    headers = {'Content-type': 'application/json'}
    body = json.dumps({"tactic_state": tactic_state, "prefix": prefix})
    conn.request("POST", "/", body, headers)
    response = conn.getresponse()
    data = response.read()
    data_dict = json.loads(data)
    print('[SUGGESTION]'.join(data_dict['suggestions']))
    conn.close()

def suggest_colab(tactic_state, prefix):
  data = {'tactic_state': tactic_state, 'prefix': prefix}
  response = json.loads(requests.post(HOST, json=data).content)
  print('[SUGGESTION]'.join(response['suggestions']))


if __name__ == "__main__":
    if COLAB:
      suggest_colab(sys.argv[1], sys.argv[2])
    else:
      suggest(sys.argv[1], sys.argv[2])

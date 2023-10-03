import sys
import http.client
import json
import sys
import os
import requests

HOST = os.environ.get('LLMSTEP_HOST', 'localhost')
PORT = os.environ.get('LLMSTEP_PORT', 5000)
SERVER = os.environ.get('LLMSTEP_SERVER', 'DEFAULT')


def suggest(host, port, tactic_state, prefix):
    conn = http.client.HTTPConnection(host, port)
    headers = {'Content-type': 'application/json'}
    body = json.dumps({"tactic_state": tactic_state, "prefix": prefix})
    conn.request("POST", "/", body, headers)
    response = conn.getresponse()
    data = response.read()
    data_dict = json.loads(data)
    print('[SUGGESTION]'.join(data_dict['suggestions']))
    conn.close()


def suggest_colab(host, tactic_state, prefix):
    data = {'tactic_state': tactic_state, 'prefix': prefix}
    response = json.loads(requests.post(host, json=data).content)
    print('[SUGGESTION]'.join(response['suggestions']))


if __name__ == "__main__":
    if SERVER == 'COLAB':
        suggest_colab(HOST, sys.argv[1], sys.argv[2])
    else:
        suggest(HOST, PORT, sys.argv[1], sys.argv[2])

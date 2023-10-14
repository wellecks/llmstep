import sys
import http.client
import json
import sys
import os
import requests

HOST = os.environ.get('LLMSTEP_HOST', 'localhost')
PORT = os.environ.get('LLMSTEP_PORT', 6000)
SERVER = os.environ.get('LLMSTEP_SERVER', 'DEFAULT')


def suggest(host, tactic_state, prefix):
    data = {'tactic_state': tactic_state, 'prefix': prefix}
    response = json.loads(requests.post(host, json=data).content)
    print('[SUGGESTION]'.join(response['suggestions']))


if __name__ == "__main__":
    if SERVER == 'COLAB':
        URL = HOST
    else:
        URL = f'http://{HOST}:{PORT}'

    suggest(URL, sys.argv[1], sys.argv[2])

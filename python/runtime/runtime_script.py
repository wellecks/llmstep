import sys
import http.client
import json
import sys
import os
import requests
import time

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

    test_list = list(range(1, 18))
    start = time.time()
    for test in test_list:
        with open(f'./arguments/test{test}.json', 'r') as f:
            data = json.load(f)
        suggest(URL, data['tactic_state'], data['prefix'])
    end = time.time()

    print('Mean elapsed time per test case:', (end - start)/len(test_list))

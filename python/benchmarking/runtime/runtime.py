import sys
import http.client
import json
import sys
import os
import requests
import time
from tqdm import tqdm

HOST = os.environ.get('LLMSTEP_HOST', 'localhost')
PORT = os.environ.get('LLMSTEP_PORT', 6000)
SERVER = os.environ.get('LLMSTEP_SERVER', 'DEFAULT')


def suggest(host, tactic_state, prefix):
    data = {'tactic_state': tactic_state, 'prefix': prefix}
    response = json.loads(requests.post(host, json=data).content)


if __name__ == "__main__":
    if SERVER == 'COLAB':
        URL = HOST
    else:
        URL = f'http://{HOST}:{PORT}'

    test_cases = []
    with open(f'./test_cases/test_cases.jsonl', 'r') as f:
        test_cases = [json.loads(line) for line in f.readlines()]

    start = time.time()
    for test_case in tqdm(test_cases, total=len(test_cases)):
        suggest(URL, test_case['tactic_state'], test_case['prefix'])
    end = time.time()

    print('Mean elapsed time per test case:', (end - start)/len(test_cases))


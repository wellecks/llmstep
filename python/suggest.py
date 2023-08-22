import sys
import http.client
import json
import sys
import os

HOST = os.environ.get('LLMSTEP_HOST', 'localhost')
PORT = os.environ.get('LLMSTEP_PORT', 5000)


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

if __name__ == "__main__":
    suggest(sys.argv[1], sys.argv[2])

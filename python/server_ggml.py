from flask import Flask, request, jsonify

from llama_cpp import Llama 

import argparse
import os
import time

EOS = "[EOS]"

parser = argparse.ArgumentParser()
parser.add_argument(
        '--model_path', 
        type=str, 
        default='/home/zhangir/projects/llama.cpp/models/ggml-open-llama-3b-q4_0.bin'
        )
args = parser.parse_args()

def load_ggml(model_path):
    print(f"loading {model_path}...")
    llm = Llama(model_path=model_path)
    print("Done.")
    return llm

def generate(prompt) -> str:
    output = model(prompt, max_tokens=128, stop=EOS)
    return output["choices"][0]['text']

model = load_ggml(args.model_path)
app = Flask(__name__)

@app.route('/', methods=['POST'])
def process_request():
    start = time.time()
    data = request.get_json()

    tactic_state = data.get('tactic_state')
    prefix = data.get('prefix')

    prompt = """[GOAL]%s[PROOFSTEP]%s""" % (tactic_state, prefix)
    texts = [prefix + generate(prompt)]

    response = {"suggestions": texts}
    end = time.time()
    print("%d suggestions (%.3fs)" % (len(texts), (end-start)))
    return jsonify(response)


if __name__ == '__main__':
    port = os.environ.get('LLMSTEP_PORT', 5000)
    app.run(use_reloader=False, host='0.0.0.0', port=port)

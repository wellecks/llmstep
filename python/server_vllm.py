from flask import Flask, request, jsonify

import vllm
import argparse
import transformers
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument('--hf-model', type=str, default='wellecks/llmstep-mathlib4-pythia2.8b')
parser.add_argument('--temperatures', type=float, nargs='+', default=[0.5])
parser.add_argument('--num-samples', type=int, default=10)
args = parser.parse_args()


def load_vllm(model_name):
    print("Loading model...")
    if model_name == 'wellecks/llmstep-mathlib4-pythia2.8b':
        model = vllm.LLM(
            model=model_name,
            tensor_parallel_size=1,
            dtype='float16'
        )
        tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(model_name)
    else:
        raise NotImplementedError(model_name)
    print("Done")
    return model, tokenizer


def _unique_sorted(texts, scores):
    texts_ = []
    scores_ = []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_


def generate(prompt, temperatures, num_samples):
    texts, scores = [], []
    for temperature in temperatures:
        params = vllm.SamplingParams(
            n=num_samples if temperature > 0 else 1,
            temperature=temperature,
            max_tokens=128
        )
        outputs = model.generate([prompt], params, use_tqdm=False)
        for output in outputs[0].outputs:
            text = output.text.replace(tokenizer.eos_token, '')
            score = output.cumulative_logprob/max(len(output.token_ids), 1)
            texts.append(text)
            scores.append(score)

    texts, scores = _unique_sorted(texts, scores)
    return texts


model, tokenizer = load_vllm(args.hf_model)
app = Flask(__name__)

@app.route('/', methods=['POST'])
def process_request():
    start = time.time()
    data = request.get_json()

    tactic_state = data.get('tactic_state')
    prefix = data.get('prefix')

    prompt = """[GOAL]%s[PROOFSTEP]%s""" % (tactic_state, prefix)
    texts = generate(prompt, args.temperatures, args.num_samples)
    texts = [prefix + text for text in texts]

    response = {"suggestions": texts}
    end = time.time()
    print("%d suggestions (%.3fs)" % (len(texts), (end-start)))
    return jsonify(response)


if __name__ == '__main__':
    port = os.environ.get('LLMSTEP_PORT', 5000)
    app.run(use_reloader=False, host='0.0.0.0', port=port)

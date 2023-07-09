from flask import Flask, request, jsonify

import argparse
import transformers
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--hf-model', type=str, default='wellecks/llmstep-mathlib4-pythia2.8b')
parser.add_argument('--temperatures', type=float, nargs='+', default=[0.5])
parser.add_argument('--num-samples', type=int, default=10)
args = parser.parse_args()


print("Loading model...")
model = transformers.GPTNeoXForCausalLM.from_pretrained(args.hf_model)
if torch.cuda.is_available():
    model.cuda()
model.eval()

tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(args.hf_model) 
print("Done.")


def generate(prompt, temperatures, num_samples):
    print(prompt)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    texts = []
    for temp in temperatures:
        out = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=temp > 0,
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_samples if temp > 0 else 1
        )
        texts.extend(tokenizer.batch_decode(
            out[:,input_ids.shape[1]:], 
            skip_special_tokens=True
        ))
    return texts

app = Flask(__name__)

@app.route('/', methods=['POST'])
def process_request():
    data = request.get_json()

    tactic_state = data.get('tactic_state')
    prefix = data.get('prefix')

    prompt = """[GOAL]%s[PROOFSTEP]%s""" % (tactic_state, prefix)
    texts = generate(prompt, args.temperatures, args.num_samples)
    texts = [prefix + text for text in texts]

    response = {"suggestions": texts}
    return jsonify(response)

if __name__ == '__main__':
    port = os.environ.get('LLMSTEP_PORT', 5000)
    app.run(debug=True, host='0.0.0.0', port=port)

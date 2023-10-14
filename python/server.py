from flask import Flask, request, jsonify

import argparse
import transformers
import torch
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pyngrok import ngrok


parser = argparse.ArgumentParser()
parser.add_argument('--hf-model', type=str, default='wellecks/llmstep-mathlib4-pythia2.8b')
parser.add_argument('--temperatures', type=float, nargs='+', default=[0.5])
parser.add_argument('--num-samples', type=int, default=10)
args = parser.parse_args()

# Prompt template for the default model.
# Change this if your model expects a different input format.
def llmstep_prompt(tactic_state, prefix):
  return '[GOAL]%s[PROOFSTEP]%s' % (tactic_state, prefix)

def load_hf(hf_model):
    print("Loading model...")
    if 'wellecks/llmstep-mathlib4-pythia' in hf_model:
        model = transformers.GPTNeoXForCausalLM.from_pretrained(args.hf_model)
        tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(args.hf_model)
    else:
        raise NotImplementedError(hf_model)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    print("Done.")
    return model, tokenizer


def hf_generate(
    model,
    tokenizer,
    prompt,
    temperatures,
    num_samples,
    max_new_tokens=128
):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    texts = []
    for temp in temperatures:
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temp > 0,
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_samples if temp > 0 else 1
        )
        output_tokens = out[:, input_ids.shape[1]:]
        texts.extend(tokenizer.batch_decode(
            output_tokens,
            skip_special_tokens=True
        ))
    texts = list(set(texts))
    return texts


class LLMStepServer(HTTPServer):
    def __init__(
        self, model, tokenizer, generate_function, config, *args, **kwargs
    ):
      self.model = model
      self.tokenizer = tokenizer
      self.generate_function = generate_function
      self.config = config
      super().__init__(*args, **kwargs)


class LLMStepRequestHandler(BaseHTTPRequestHandler):
    def process_request(self, tactic_state, prefix):
        prompt = self.server.config['LLMSTEP_PROMPT'](tactic_state, prefix)
        texts = self.server.generate_function(
            model=self.server.model,
            tokenizer=self.server.tokenizer,
            prompt=prompt,
            temperatures=self.server.config['LLMSTEP_TEMPERATURES'],
            num_samples=self.server.config['LLMSTEP_NUM_SAMPLES']
        )
        texts = [prefix + text for text in texts]
        response = {"suggestions": texts}
        return response

    def do_POST(self):
        # Set response headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        # Get the incoming POST data
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(post_data)
            result = self.process_request(data['tactic_state'], data['prefix'])
            response = result
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            # Handle errors gracefully
            error_response = {'error': str(e)}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))




if __name__ == '__main__':
  # Download and load the model (this takes a few minutes).
  model, tokenizer = load_hf(args.hf_model)
  host = os.environ.get('LLMSTEP_HOST', 'localhost')
  port = int(os.environ.get('LLMSTEP_PORT', 5000))

  config = {
    'LLMSTEP_MODEL': args.hf_model,

    # Sampling temperature(s)
    'LLMSTEP_TEMPERATURES': args.temperatures,

    # Number of generated suggestions per temperature
    'LLMSTEP_NUM_SAMPLES': args.num_samples,

    # Prompt template
    'LLMSTEP_PROMPT': llmstep_prompt
}
  # Create the server
  server_address = (host, port)
  httpd = LLMStepServer(
    model, tokenizer, hf_generate, config,
    server_address, LLMStepRequestHandler
  )

  print('Server started')
  httpd.serve_forever()

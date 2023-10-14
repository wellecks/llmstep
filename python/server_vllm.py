from server import LLMStepServer, get_argparser, get_config, print_config

import vllm
import transformers


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


def vllm_generate(
    model,
    tokenizer,
    prompt,
    temperatures,
    num_samples,
    max_new_tokens=128
):
    texts, scores = [], []
    for temperature in temperatures:
        params = vllm.SamplingParams(
            n=num_samples if temperature > 0 else 1,
            temperature=temperature,
            max_tokens=max_new_tokens
        )
        outputs = model.generate([prompt], params, use_tqdm=False)
        for output in outputs[0].outputs:
            text = output.text.replace(tokenizer.eos_token, '')
            score = output.cumulative_logprob/max(len(output.token_ids), 1)
            texts.append(text)
            scores.append(score)

    texts, scores = _unique_sorted(texts, scores)
    return texts


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    config = get_config(args)
    print_config(config)

    model, tokenizer = load_vllm(args.hf_model)

    httpd = LLMStepServer(
        model, tokenizer, vllm_generate, config
    )

    print('Server started')
    httpd.serve_forever()

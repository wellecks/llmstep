from server import LLMStepServer, get_argparser, get_config, print_config

import transformers


def load_hf_encdec(model_name):
    print("Loading model...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Done")
    return model, tokenizer


def hf_encdec_generate(
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
        texts.extend(tokenizer.batch_decode(
            out, skip_special_tokens=True
        ))
    texts = list(set(texts))
    return texts


def reprover_prompt(tactic_state, prefix):
    return '%s%s' % (tactic_state, prefix)


def get_reprover_config(args):
    config = get_config(args)
    config['LLMSTEP_PROMPT'] = reprover_prompt
    return config


if __name__ == '__main__':
    parser = get_argparser()
    parser.set_defaults(hf_model='kaiyuy/leandojo-lean4-tacgen-byt5-small')
    args = parser.parse_args()

    config = get_reprover_config(args)
    print_config(config)

    model, tokenizer = load_hf_encdec(args.hf_model)

    httpd = LLMStepServer(
        model, tokenizer, hf_encdec_generate, config
    )

    print('Server started')
    httpd.serve_forever()

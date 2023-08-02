# `llmstep`: [L]LM proofstep suggestions in Lean
`llmstep` is a Lean 4 tactic for suggesting proof steps using a language model:

<img src="./llmstep.gif" width="350"/>

Calling `llmstep "prefix"` gives suggestions that start with `prefix`:
```lean
example (f : ℕ → ℕ) : Monotone f → ∀ n, f n ≤ f (n + 1) := by
  intro h n
  llmstep "exact"

==> Lean Infoview
  Try This:
    * exact h (Nat.le_succ _)
    * exact h (Nat.le_succ n)
    * exact h (Nat.le_add_right _ _)
```

Clicking a suggestion places it in the proof:
```lean
example (f : ℕ → ℕ) : Monotone f → ∀ n, f n ≤ f (n + 1) := by
  intro h n
  exact h (Nat.le_succ _) -- llmstep "exact" 
```

By default, `llmstep` uses a language model finetuned on Mathlib4 extracted with [LeanDojo](https://zenodo.org/record/8040110), and
[supports other LMs](#language-model).



## Quick start

First, install [install Lean 4 in VS Code](https://leanprover.github.io/lean4/doc/quickstart.html) and the python requirements (`pip install -r requirements.txt`).

Then start the server:
```bash
python python/server.py
```

Open `LLMstep/Examples.lean` in VS Code and try out `llmstep`. 

## Implementation
`llmstep` has three parts:
1. a Lean tactic
2. a language model
3. a Python server 
   
The Lean tactic calls a Python script, which sends a request to the server. \
The server calls the language model and returns the generated suggestions. \
The suggestions are displayed by the tactic in VS Code.


## Language model
By default, `llmstep` uses a Pythia 2.8b language model fine-tuned on [LeanDojo Benchmark 4](https://zenodo.org/record/8040110):
- [`llmstep` model on Huggingface](https://huggingface.co/wellecks/llmstep-mathlib4-pythia2.8b)


The model is fine-tuned on sequences of the form:
```bash
[GOAL]tactic-state[PROOFSTEP]next-tactic[END]
```
This format corresponds to the proofstep objective from [Han et al ICLR 2022](https://arxiv.org/abs/2102.06203).\
The [python/train](python/train) directory shows how the model was fine-tuned.

#### Fine-tuning your own model
The scripts in [python/train](python/train) show how to finetune a model.

#### Using a different model
Swap in other language models with the `--hf-model` argument:
```bash
python server.py --hf-model some/other-model-7B
```
We recommend using a fine-tuned model, though in principle fine-tuning is not strictly needed. \
`llmstep` assumes the model uses the proofstep format described above, but this is easy to modify.


#### Speed
Starting the server downloads the default language model, and loads the model. As a result, you will likely experience a delay the first time `llmstep` is run.
Roughly speaking, `llmstep` can run on a typical MacBook Pro with the default model, and runs faster when `server.py` is on a machine with a GPU. 


## Additional Notes

#### Acknowledgements
* The `llmstep` tactic is inspired by [`gpt-f`](https://github.com/jesse-michael-han/lean-gptf). 
* Fine-tuning data for the default model is from the amazing [LeanDojo](https://leandojo.org/). 
* The fine-tuning code is based on the script from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
* The tactic implementation adopts ideas and code from Mathlib4's `Polyrith` and `Std.Tactic.TryThis`. 

#### History
`llmstep` was initially created for an IJCAI-2023 tutorial on neural theorem proving. \
It aims to provide LM-based suggestions built with open-source components. 

#### Citation

If you find this repository useful in your work, please cite:
```
@misc{llmstep,
  author = {Sean Welleck},
  title = {llmstep: LLM proofstep suggestions in Lean},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/wellecks/llmstep}},
}
```

Naturally, please cite [LeanDojo](https://leandojo.org/), [PACT](https://arxiv.org/abs/2102.06203), and other relevant resources.

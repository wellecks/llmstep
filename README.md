# `llmstep`: [L]LM proofstep suggestions in Lean
*News*
- [11.2023] Experimental [*Llemma*](https://arxiv.org/abs/2310.10631) suggestions that leverage file context
- [10.2023] New paper describing version 1.0.0 of `llmstep`: [[paper](https://arxiv.org/abs/2310.18457)]
- [10.2023] Support for [Reprover](#reprover)
- [9.2023] Support for free GPU servers via [Google Colab](#google-colab)


---

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
  exact h (Nat.le_succ _)
```

`llmstep` checks the language model suggestions in Lean, and highlights those that close the proof.

## Quick start

First, [install Lean 4 in VS Code](https://leanprover.github.io/lean4/doc/quickstart.html) and the python requirements (`pip install -r requirements.txt`).

Then [start a server](#servers):
```bash
python python/server.py
```

Open `LLMstep/Examples.lean` in VS Code and try out `llmstep`.

## Use `llmstep` in a project
1. Add `llmstep` in `lakefile.lean`:
```lean
require llmstep from git
  "https://github.com/wellecks/llmstep"
```
Then run `lake update`.

2. Import `llmstep` in a Lean file:
```lean
import LLMstep
```

3. Start a server based on your runtime environment. For instance:
```bash
python python/server.py
```
Please see the [recommended servers below](#servers).

## Servers
The `llmstep` tactic communicates with a server that you can run in your own environment (e.g., CPU, GPU, Google Colab).

The table below shows the recommended language model and server scripts.
To start a server, use `python {script}`, e.g. `python python/server_vllm.py`:

| Environment  | Script | Default Model | Context |Speed | miniF2F-test |
| -------- | ------- | ------- |-------|------- |------- |
| CPU  | `python/server_encdec.py` | [LeanDojo ByT5 300m](https://huggingface.co/kaiyuy/leandojo-lean4-tacgen-byt5-small) | State | 3.16s | 22.1\%|
| Colab GPU  | See [Colab setup](#google-colab)  | [llmstep Pythia 2.8b](https://huggingface.co/wellecks/llmstep-mathlib4-pythia2.8b) |State |1.68s | 27.9\%|
| CUDA GPU | `python/server_vllm.py` | [llmstep Pythia 2.8b](https://huggingface.co/wellecks/llmstep-mathlib4-pythia2.8b) |State|**0.25s** | **27.9\%**|
| CUDA GPU* | `python/server_llemma.py` | [Llemma 7b](https://huggingface.co/EleutherAI/llemma_7b) |State, **current file**  🔥  | N/A | N/A|


Please refer to [our paper](https://arxiv.org/abs/2310.18457) for further information on the benchmarks.

`llmstep` aims to be a model-agnostic tool. We welcome contributions of new models.


\* File context support (e.g. with [Llemma](https://arxiv.org/abs/2310.10631)) is currently experimental.


## Implementation
<img src="./docs/llmstep.png" width="700"/>


`llmstep` has three parts:
1. a [Lean tactic](./LLMstep/LLMstep.lean)
2. a [language model](https://huggingface.co/wellecks/llmstep-mathlib4-pythia2.8b)
3. a [Python server](./python/server.py)

The Lean tactic sends a request to the server. \
The server calls the language model and returns the generated suggestions. \
The suggestions are displayed by the tactic in VS Code.



## Google Colab

To use Google Colab's free GPU to run a server, follow these instructions:

1. Open and run this notebook to start a server: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wellecks/llmstep/blob/master/python/colab/llmstep_colab_server.ipynb)

2. In your local environment, set the environment variable `LLMSTEP_HOST` equal to the url printed out in this notebook (for example, `https://04fa-34-125-110-83.ngrok.io/`).

3. In your local environment, set the environment variable `LLMSTEP_SERVER=COLAB`.

4. Use `llmstep`.

#### VS Code steps (2) and (3)

To set environment variables in VS Code, go to:

- Settings (`Command` + `,` on Mac)
- Extensions -> Lean 4
- Add the environment variables to `Server Env`. For example:
<img src="./docs/vscode_env1.png" width="400"/>

- Then restart the Lean Server (`Command` + `t`, then type `> Lean 4: Restart Server`):
<img src="./docs/vscode_env2.png" width="400"/>



## Local inference with llama.cpp
The standard server and vLLM server are intended to run on GPUs. However, `llmstep` also includes `python/server_ggml.py`, an experimental [llama.cpp](https://github.com/ggerganov/llama.cpp) based server intended to run efficiently on CPUs. Currently, the llama.cpp server supports only one tactic suggestion per `llmstep` call. 

Configuring `python/server_ggml.py` requires a few extra steps. Note that the official [llmstep model](https://huggingface.co/wellecks/llmstep-mathlib4-pythia2.8b) is not compatible with the llama.cpp server. Instead, download the experimental [zhangirazerbayev/open-llama-3b_next-tactic_dev0.2](https://huggingface.co/zhangirazerbayev/open-llama-3b_next-tactic_dev0.2) as follows:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/zhangirazerbayev/open-llama-3b_next-tactic_dev0.2
```
The following steps will install llama.cpp and convert the above model to an executable binary. 
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# install llama.cpp python dependencies
python3 -m pip install -r requirements.txt

# convert the model you downloaded to ggml FP16 format
python3 convert.py $PATH_TO_MODEL
```
This will produce a file named `ggml-model-f32.bin` in the `$PATH_TO_MODEL` directory. 

If the above steps do not work for you, please, consult the more detailed instructions in the [llama.cpp](https://github.com/ggerganov/llama.cpp) repository.

In the following step, you will quantize your model to a reduced precision format. The available formats are `F16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0`, with lower precision formats trading off accuracy for latency and memory. I would recommend starting with `Q4_0`, and increasing precision if your hardware handles lower precisions comfortably. 
```bash
./quantize $PATH_TO_MODEL/ggml-model-f32.bin $PATH_TO_QUANTIZED
```
Then, you may start `server_ggml.py` as follows
```bash
python3 python/server_ggml.py --model_path $PATH_TO_QUANTIZED
```
Once the server is running, use `llmstep` as you normally would. 

## Language model
By default, `llmstep` uses a Pythia 2.8b language model fine-tuned on [LeanDojo Benchmark 4](https://zenodo.org/record/8040110):
- [`llmstep` model on Huggingface](https://huggingface.co/wellecks/llmstep-mathlib4-pythia2.8b)


The [python/train](python/train) directory shows how the model was fine-tuned.

#### Reprover
You can use the non-retrieval version of [Reprover](https://github.com/lean-dojo/ReProver), which we refer to as [LeanDojo ByT5 300m](https://huggingface.co/kaiyuy/leandojo-lean4-tacgen-byt5-small):

```
python python/server_encdec.py
```
By default, this runs the `leandojo-lean4-tacgen-byt5-small` model.\
This model is particularly useful on CPU due to its small parameter count.

#### Using a different model

Swap in other decoder-only language models with the `--hf-model` argument:
```bash
python server.py --hf-model some/other-model-7B
```
Use `--hf-model` with `python/server_encdec.py` for encoder-decoder models.

Use `--hf-model` with `python/server_llemma.py` for prompted base models (e.g. CodeLlama).


#### Fine-tuning a model
The scripts in [python/train](python/train) show how to finetune a model.

## Additional Notes

#### Acknowledgements
* The `llmstep` tactic is inspired by [`gpt-f`](https://github.com/jesse-michael-han/lean-gptf).
* Fine-tuning data for the Pythia-2.8b model is from  [LeanDojo](https://leandojo.org/).
* The fine-tuning code is based on the script from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
* The tactic implementation adopts ideas and code from Mathlib4's `Polyrith` and `Std.Tactic.TryThis`.
* Thank you to Mario Carneiro and Scott Morrison for reviewing the tactic implementation.

#### History
`llmstep` was initially created for an IJCAI-2023 tutorial on neural theorem proving.\
It aims to be a model-agnostic platform for integrating language models and Lean.

#### Citation

Please cite:
```
@article{welleck2023llmstep,
    title={LLMSTEP: LLM proofstep suggestions in Lean},
    author={Sean Welleck and Rahul Saha},
    journal={arXiv preprint arXiv:2310.18457},
    year={2023}
}
```

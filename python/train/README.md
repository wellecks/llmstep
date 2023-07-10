# Fine-tuning

Fine-tuning your own model is optional: by default, `llmstep` uses a model available on Huggingface that was fine-tuned with these scripts:
- [wellecks/llmstep-mathlib4-pythia2.8b](https://huggingface.co/wellecks/llmstep-mathlib4-pythia2.8b?doi=true) 

First download and format the data:
```bash
python data.py
```

Fine-tuning is then done using `tune.py`. See `tune_proofstep.sh` for an example command. \
The command uses 8 GPUs with Deepspeed (tested on NVIDIA RTX A6000).


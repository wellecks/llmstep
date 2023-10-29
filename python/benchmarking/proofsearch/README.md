#### Setup
Install Python packages:
```
lean-dojo==1.1.2
torch==2.0.1
transformers==4.33.2
vllm==0.1.7
```

Install Lean:
```
# from https://leanprover-community.github.io/install/linux.html
# After running this command, select (2), then `nightly`, then `y`:
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source $HOME/.elan/env
lake
```

Configure LeanDojo:
```
export CONTAINER="native"
```

#### Run
See `proofsearch.sh`

#### Compute metrics

```bash
python compute_metrics.py
==>

0.2786885245901639 68 244
```

#### Citation
Note that this evaluation is similar to that of [llemma_formal2formal](https://github.com/wellecks/llemma_formal2formal). 

Please cite LLMstep and/or Llemma if you find this evaluation code useful.
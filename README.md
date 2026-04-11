# SymQM-3k

Code for the paper: *SymQM-3k: A Symmetry-Resolved Quantum Chemistry 
Dataset and Multi-Task Graph Neural Network for Molecular Orbital 
Irrep Prediction*

## Installation

```bash
pip install torch_geometric mace-torch torch_scatter torch_sparse torch_cluster e3nn numpy pandas
```

## Data

Download the SymQM-3k dataset from HuggingFace:

```bash
python data/download.py
```

Or manually download the four CSV files from [https://huggingface.co/datasets/taradutt007/symqm3k] 
and place them in the working directory.

## Evaluation with Pretrained Weights

Download the pretrained SymMACE weights from HuggingFace:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="taradutt007/symqm3k",
    filename="symmace_best.pt",
    repo_type="dataset")
```

Then evaluate:

```bash
python evaluate.py --weights symmace_best.pt
```

## Training

Train SymMACE from scratch (reproduces paper results):

```bash
python train.py
```

This runs two-phase training:
- Phase 1: 150 epochs, lr=1e-3
- Phase 2: 200 epochs fine-tuning, lr=1e-4

Results are saved to `results.json`.

## Expected Results

HOMO MAE          : 0.1528 eV
LUMO MAE          : 0.1393 eV
HOMO Irrep Acc    : 76.0%
LUMO Irrep Acc    : 87.6%
HOMO Irrep Idx MAE: 3.40 positions
LUMO Irrep Idx MAE: 2.05 positions
Transition Acc    : 76.5%

## Citation


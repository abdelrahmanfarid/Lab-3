# Security Insider Lab 3 (Group 7): DDoS Detection with ML and Filtering with eBPF/XDP

| Task | What | Files |
|---|---|---|
| 1 | Classical classifiers: Gaussian Naive Bayes, Gradient Boosting, AdaBoost, KNN, linear SVM | `lab3_tasks_1_2_3.ipynb` |
| 2 | Neural networks: MLP, LSTM, CNN (PyTorch) | `lab3_tasks_1_2_3.ipynb` |
| 3 | Comparison of all eight models on one held-out test set | `lab3_tasks_1_2_3.ipynb`, `artifacts/` |
| 4 | Own XDP filter: source blocklist and per-source rate limit (1,000 packets/s) | `xdp/xdp_filter.c`, `xdp/run_demo.sh` |
| 5 | Building and testing xdp-tools (xdp-filter, xdpdump, xdp-bench, xdp-loader) | `xdp/run_xdp_tools.sh` |

## Files

- `lab3_tasks_1_2_3.ipynb`: the executed notebook for Tasks 1 to 3
- `build_notebook.py`: generates the notebook source
- `run_notebook.py`: executes the notebook with the project virtual environment
- `preprocess.py`: Lab 2 preprocessing that produced `ddos_top20_scaled.csv` and `feature_scores.csv`
- `artifacts/`: comparison table, figures and training histories written by the notebook
- `xdp/`: XDP program, Makefile, counter script, container images and test scripts
- `xdp/results/`: output of the Task 4 and Task 5 test scripts

## Data

The datasets are too large for this repository. Place them in this folder before running the notebook:

- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` from the CICIDS2017 dataset
- `ddos_top20_scaled.csv`, created with `python preprocess.py Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

`feature_scores.csv` is included.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_notebook.py
```

The XDP tests need a rootful Podman machine (or Docker, with `DOCKER=docker`):

```bash
./xdp/run_demo.sh         # Task 4, writes xdp/results/demo_output.txt
KEEP=1 ./xdp/run_demo.sh  # same, but keeps the containers running for manual tests
./xdp/run_xdp_tools.sh    # Task 5, writes xdp/results/xdp_tools_output.txt
```

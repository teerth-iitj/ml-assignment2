# Machine Learning Assignment 2: Bagging, Boosting and Bias-Variance on Olivetti Faces

## Setup

```bash
pip install -r requirements.txt
```

The Olivetti faces are downloaded automatically by scikit-learn on first use.
All scripts must be run from this folder.

## Reproduction commands (run in this order)

```bash
python common.py                      # writes split.npz
python q1a.py                         # writes frozen_tree.json
python bagging.py                     # Q1(b)
python q1c.py                         # writes final_B.json
python q1d.py                         # Q1(d)
python samme.py > q2a_log.txt         # Q2(a) toy SAMME round
python q2b.py                         # writes boost_final.json
python q2c.py                         # Q2(c)
python q2d.py                         # Q2(d)
python -u q3a.py                      # Q3(a), 30 refits (resumable)
python -u q3b.py                      # Q3(b) and (c), depth study (resumable)
python q3d.py                         # Q3(d) final table
```

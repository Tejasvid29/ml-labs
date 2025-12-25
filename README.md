# ML Labs

A modular machine learning experimentation repository focused on
training dynamics, optimization behavior, and generalization on
standard vision benchmarks.

This project is designed to keep **code, experiments, and notes cleanly separated**,
with all results reproducible by re-running experiments locally.

---

## Project Goals

- Build a **clean, reusable ML training pipeline**
- Study how **optimization choices** affect learning behavior
- Run **controlled experiments** with minimal confounders
- Maintain **professional ML repo hygiene** (no datasets or raw runs in Git)

---

## Repository Structure

```text
ml-labs/
│
├── Common/
│   ├── __init__.py
│   ├── data.py        # Dataset loading and preprocessing
│   ├── model.py       # Model definitions
│   └── train.py       # Training and evaluation pipeline
│
├── mini/
│   ├── README.md      # Experiment summaries and notes
│   └── 01_optimization/
│       └── README.md  # Optimizer and learning-rate experiments
│
├── reports/
│   └── README.md      # Cross-experiment analysis and writeups
│
├── .gitignore
└── README.md

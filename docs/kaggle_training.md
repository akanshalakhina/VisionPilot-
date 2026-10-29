# Training on Kaggle

Your laptop has about 5 GB of free disk and a slow install path, so full training runs on Kaggle Notebooks. The laptop is for code and small tests.

Kaggle limits (RAM, session length, weekly GPU hours, internet access) change. Check the current numbers in the notebook settings. None of these models needs a GPU, so leave the accelerator off.

## 1. Get the code onto Kaggle

Pick one:

- **GitHub:** push this repo to your own GitHub account, turn Internet on in the notebook settings (Kaggle may ask you to verify your phone number), then `!git clone https://github.com/<you>/<repo>.git`.
- **No internet:** zip the repo without `.venv`, upload it as a Kaggle Dataset, and add it to the notebook. Then `import sys; sys.path.insert(0, "/kaggle/input/<dataset-name>/visionpilot-agent")`. Kaggle images already include pandas, scikit-learn and LightGBM.

## 2. Add the data

- **Avazu:** open the competition page, accept its rules (read them first), then in the notebook choose Add Input and search for the competition. In the right-hand Input panel, check the real file name. It is commonly `train.gz` or `train.csv`.
- **Criteo Uplift:** download is a single `.csv.gz` file of about 297 MB (see the loader's docstring for the source and license). Do NOT unzip it. The loader reads the `.gz` directly.
- **Open Bandit:** download from the ZOZO data page and check its terms first.

## 3. Run

```python
%cd /kaggle/working/visionpilot-agent      # or wherever the code is
!pip install -q -e ".[ml]"                  # skip if Internet is off and use the sys.path route
!python -m ml.predictor.train \
    --data /kaggle/input/avazu-ctr-prediction/train.gz \
    --sample-frac 0.05 --out /kaggle/working/ctr
```

`--sample-frac 0.05` keeps about 5% of the rows, spread over every day. Start there, then raise it if memory allows. The split is by day: earlier days train, then one validation day (early stopping), then one test day (the numbers you report).

## 4. Bring results back

Download `/kaggle/working/ctr/metrics.json` and the model files (`feature_builder.joblib`, `logreg.joblib`, `lightgbm.joblib`) and put them in `models/ctr/` in the repo. Small models can be committed so the demo works offline.

## Uplift targeting (Criteo)

```python
!python -m ml.uplift.experiment \
    --data /kaggle/input/<your-criteo-dataset>/criteo-research-uplift-v2.1.csv.gz \
    --sample-frac 0.1 --outcome visit --n-boot 100 --out /kaggle/working/uplift_criteo.json
```

- Start with `--outcome visit`. Conversions are rare (about 0.3% of rows), so their uplift estimates are very noisy.
- About 14 million rows exist. A 10% sample is 1.4 million, which the models handle comfortably on Kaggle.
- Criteo's `treatment` means eligible for targeting, not that the ad was seen. Say so in the report.
- The bootstrap only covers noise in the test rows. Run 3 seeds and report all of them.
- Qini numbers depend on how they are defined and on the split. Compare models only with each other, on the same split.

## Report checklist

- Name the dataset, its terms and the exact sample fraction.
- Quote the test-split numbers only. Never quote validation numbers as final.
- Compare the models against the base-rate row, not just against each other.
- Repeat with 3 seeds and report the spread.

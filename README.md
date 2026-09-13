# CS771 Mini-Project 1

Intro to ML course project. The task is binary classification of the same set of
inputs given under three different feature representations (emoticons, deep
features, and a digit sequence), plus one model that combines all three. Every
model is kept within the 10,000 trainable-parameter limit from the problem
statement.

## Results on the validation set (489 examples)

| Dataset       | Model                                            | Params | Val. acc. |
|---------------|--------------------------------------------------|-------:|----------:|
| Emoticon      | positional one-hot, Logistic Regression (C=5)    | 2,783  | 91.00%    |
| Deep features | flattened 13x768, Logistic Regression (C=0.3)    | 9,985  | 98.57%    |
| Text sequence | positional one-hot + char n-grams, LogReg (C=1)  | 9,501  | 76.89%    |
| Combined      | out-of-fold stacking of the three base models    | 4*     | 98.57%    |

\* parameters of the stacking meta-model; the base models are the three above.
Combining the datasets did not beat the deep-features model on its own. The
report explains why.

## Files

```
datasets/                 train / valid / test for the 3 representations
common.py                 data loading and feature transforms
emoticon_final.py         emoticon model (Task 1)
deepfeat_final.py         deep-features model (Task 1)
textseq_final.py          text-sequence model (Task 1)
combine_final.py          stacking model (Task 2)
main.py                   trains all 4 models and writes the 4 pred_*.txt files
experiments.py            hyperparameter search and model comparison
make_plots.py             accuracy vs training-size plots used in the report
report.tex / report.pdf   report
plots/                    figures
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

`main.py` trains every model from scratch with fixed random seeds (takes under
20 seconds) and writes `pred_emoticon.txt`, `pred_deepfeat.txt`,
`pred_textseq.txt`, and `pred_combined.txt`, one predicted label per line.

## Notes

Only NumPy, pandas, and scikit-learn are used. No pre-trained model is used as a
feature extractor beyond the deep features that were already provided. The
sequence model uses the sequence data only, as the problem statement requires.
The dataset files are large (`train_feature.npz` is about 270 MB), so they are
not committed here; see `datasets/GET_DATA.md` for the download link.

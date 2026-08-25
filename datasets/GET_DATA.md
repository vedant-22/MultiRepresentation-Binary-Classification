# Getting the datasets

The dataset files are too large to commit to GitHub (`train_feature.npz` is
about 270 MB), so download and extract them into this folder.

**Download:** https://drive.google.com/file/d/1gSJ5QVh5DHju5TMGUj9QSAMMwGMSYlIT/view

After extracting, this folder must contain:

```
datasets/train/train_emoticon.csv
datasets/train/train_text_seq.csv
datasets/train/train_feature.npz
datasets/valid/valid_emoticon.csv
datasets/valid/valid_text_seq.csv
datasets/valid/valid_feature.npz
datasets/test/test_emoticon.csv
datasets/test/test_text_seq.csv
datasets/test/test_feature.npz
```

Then run `python main.py` from the project root.

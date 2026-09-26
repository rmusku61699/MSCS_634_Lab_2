# MSCS 634 – Lab 2: KNN and Radius Neighbors Classification

**Author:** [Your Full Name]
**Course:** MSCS 634 – [Course Title]

## Purpose

This lab compares two distance-based classifiers from scikit-learn, **K-Nearest Neighbors (KNN)** and **Radius Neighbors (RNN)**, on the Wine dataset. The dataset has 178 samples, 13 chemical features, and 3 wine classes. The goal is to see how the main parameter of each model affects test accuracy:

- **KNN:** k = 1, 5, 11, 15, 21
- **RNN:** radius = 350, 400, 450, 500, 550, 600

The data is split 80/20 into training and testing sets (stratified, `random_state=42`). Each model is trained on the training set and evaluated on the test set, and the accuracy trends are plotted and compared.

## Repository Contents

| File | Description |
|---|---|
| `MSCS_634_Lab_2.ipynb` | Jupyter Notebook with data exploration, both models, plots, and discussion |
| `README.md` | This summary |
| `requirements.txt` | Python packages needed to run the notebook |

## Results

| k (KNN) | Accuracy |
|---|---|
| 1 | 0.7778 |
| 5 | 0.8056 |
| 11 | 0.8056 |
| 15 | 0.8056 |
| 21 | 0.8056 |

| Radius (RNN) | Accuracy | Avg. neighbors inside radius |
|---|---|---|
| 350 | 0.7222 | 80.5 |
| 400 | 0.6944 | 87.9 |
| 450 | 0.6944 | 94.8 |
| 500 | 0.6944 | 100.1 |
| 550 | 0.6667 | 106.0 |
| 600 | 0.6667 | 111.0 |

## Key Insights

- **KNN outperformed RNN at every tested setting.** KNN's best accuracy was 80.6% (k = 5 to 21). RNN's best was 72.2% (radius = 350).
- **KNN improved from k = 1 to k = 5, then plateaued.** With k = 1 the model is sensitive to single noisy points (overfitting). Averaging over more neighbors stabilized predictions, and accuracy was insensitive to k between 5 and 21.
- **RNN accuracy decreased as the radius increased.** Even at radius 350, a test point has about 80 of the 142 training samples inside its radius. At 600 it has about 111. With that many neighbors, the vote moves toward the majority class and local structure is lost (underfitting).
- **Feature scale drives both results.** The features are on very different scales (`proline` ranges from about 278 to 1680, while several features are below 5), so Euclidean distance is dominated by `proline`. An extra experiment in the notebook standardized the features, and KNN accuracy rose to **97–100%**. The lab's radius values (350–600) only make sense on unscaled data, so the main experiments use raw features.

### When to prefer each model

- **KNN** is the safer default when data density varies or feature scales are arbitrary. It always uses exactly k neighbors, and k is easy to tune.
- **RNN** fits best when the distance itself is meaningful and data density is fairly uniform. It is also useful when points with no nearby neighbors should be treated as outliers. Its radius must be tuned carefully to the scale of the data.

## Challenges and Decisions

- **Unscaled vs. scaled data:** The radius range given in the lab is only meaningful on raw features. Once features are standardized, every training point would fall within a radius of 350. I kept the main experiments on unscaled data to follow the lab and added a separate scaled-KNN comparison to show the effect of scaling.
- **Empty neighborhoods in RNN:** `RadiusNeighborsClassifier` raises an error if a test point has no training points inside the radius. I set `outlier_label='most_frequent'` as a safeguard. With these radii every test point had at least 4 neighbors, so the fallback was never used.
- **Reproducibility and class balance:** I used `stratify=y` and `random_state=42` so the class proportions match across splits and the results can be reproduced.
- **Small test set:** The test set has 36 samples, so each misclassification changes accuracy by about 2.8 percentage points. Small differences between parameter values should be read with that in mind.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook MSCS_634_Lab_2.ipynb
```

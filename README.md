# CSCI 5622 : Machine Learning
**Abhirama Karthikeya Mullapudi**  
University of Colorado, Boulder. Fall 2026.

# Who should the academic adviser see first?

At the end of the first semester, an academic adviser can meet 20% of each entering class.
This project asks whether a model picks those students better than a one-line rule:
invite the students who passed the lowest share of their first-semester units.

Data: 4,424 students at the Polytechnic Institute of Portalegre (UCI dataset 697).
Models are trained on the 2008–2014 entering classes and tested once on 2015–2017.

## Headline result

Share of eventual dropouts invited when the adviser meets 20% of each test class
(1,196 students, 316 dropouts; at most 75.6% is reachable with that many slots):

| Method | Dropouts reached | Caught / missed | Minus baseline (95% interval) |
|---|---|---|---|
| L1 logistic regression | 60.8% | 192 / 124 | +1.9 points (−1.2 to +5.0) |
| Gradient boosting | 59.8% | 189 / 127 | +0.9 points (−2.2 to +4.0) |
| **Baseline rule** (lowest share of units passed) | **58.9%** | 186 / 130 | |
| Random | 21.8% | 69 / 247 | |

The models' lead over the rule is small and its interval includes zero.

## Reproduce

Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

`run.py` prints the table above and writes every table to `results/tables/` and every figure to
`results/figures/`. It takes a few seconds. The seed is fixed in `src/config.py`.

To see how the model settings were chosen (training classes only, about 25 seconds):

```bash
python tune.py
```

## What's where

| Path | Contents |
|---|---|
| `src/config.py` | Every choice: split, 20% budget, inputs, excluded columns, model settings, seed |
| `src/data.py` | Loading, entering year, dropout label, slice columns |
| `src/features.py` | Model inputs and the baseline rule |
| `src/evaluation.py` | The invitation rule, recall, bootstrap and Wilson intervals |
| `src/models.py` | L1 logistic regression and gradient boosting |
| `src/unsupervised.py` | k-means on enrollment background, silhouette check, PCA |
| `src/plots.py` | Report figures |
| `tune.py` | Forward-in-time tuning on the 2008–2014 classes |
| `run.py` | Final evaluation on the 2015–2017 classes |

## Data notes

- The file has no year column. Each student's entering year is recovered by matching the
  file's unemployment rate to Portugal's annual rate (OECD series LRUNTTTTPTA156S, via FRED),
  searching only the enrollment years stated by the dataset's authors.
- 180 students (all in Animation and Multimedia Design) have no first-semester record and are
  excluded: their zeros mean missing, not failed.
- "Enrolled" students are counted as not having dropped out.
- Second-semester results, debt and fee status, and the national economic columns are never
  used as inputs (`EXCLUDED_COLUMNS` in `src/config.py`).

## Sources

- Realinho, V., Vieira Martins, M., Machado, J., and Baptista, L. (2021). *Predict Students'
  Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. CC BY 4.0.
  Included as `data/students.csv`.
- Realinho, V., Machado, J., Baptista, L., and Martins, M. V. (2022). Predicting Student Dropout
  and Academic Success. *Data*, 7(11), 146. https://doi.org/10.3390/data7110146
- OECD, Infra-Annual Labor Statistics: Unemployment Rate Total: 15 Years or over for Portugal
  [LRUNTTTTPTA156S], retrieved from FRED, Federal Reserve Bank of St. Louis.
  https://fred.stlouisfed.org/series/LRUNTTTTPTA156S. Included as `data/portugal_unemployment_fred.csv`.
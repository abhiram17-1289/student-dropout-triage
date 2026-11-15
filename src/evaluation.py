import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from src import config


def invite(risk, entering_year, share=config.SHARE_OF_CLASS_INVITED, seed=config.SEED):
    """Invite the top `share` of each entering class by risk. Ties are broken at random."""
    risk = np.asarray(risk, dtype=float)
    entering_year = np.asarray(entering_year)
    tie_breaker = np.random.default_rng(seed).random(len(risk))

    invited = np.zeros(len(risk), dtype=bool)
    for year in np.unique(entering_year):
        in_class = np.flatnonzero(entering_year == year)
        n_invited = round(len(in_class) * share)
        # Sort by risk, highest first, then by the random number to settle ties.
        order = np.lexsort((tie_breaker[in_class], -risk[in_class]))
        invited[in_class[order[:n_invited]]] = True
    return invited


def recall(dropped_out, invited):
    """Share of eventual dropouts who were invited."""
    dropped_out = np.asarray(dropped_out, dtype=bool)
    return np.asarray(invited)[dropped_out].mean()


def recall_difference_interval(dropped_out, entering_year, invited_a, invited_b,
                               resamples=config.BOOTSTRAP_RESAMPLES, seed=config.SEED):
    """95% bootstrap interval for recall(a) minus recall(b)."""
    dropped_out = np.asarray(dropped_out)
    entering_year = np.asarray(entering_year)
    invited_a, invited_b = np.asarray(invited_a), np.asarray(invited_b)
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(entering_year == year) for year in np.unique(entering_year)]

    differences = np.empty(resamples)
    for i in range(resamples):
        # Resample students with replacement inside each class, keeping class sizes.
        sample = np.concatenate([rng.choice(members, size=len(members)) for members in classes])
        differences[i] = (recall(dropped_out[sample], invited_a[sample])
                          - recall(dropped_out[sample], invited_b[sample]))
    return np.percentile(differences, [2.5, 97.5])


def recall_by_slice(students, invited, slice_column):
    """Recall within each group of a slice, with sample sizes and a 95% Wilson interval."""
    students = students.assign(invited=np.asarray(invited))
    rows = []
    for group, members in students.groupby(slice_column):
        dropouts = members[members["dropped_out"] == 1]
        caught = int(dropouts["invited"].sum())
        low, high = proportion_confint(caught, len(dropouts), method="wilson")
        rows.append({
            "slice": slice_column, "group": group, "students": len(members),
            "dropouts": len(dropouts), "caught": caught, "missed": len(dropouts) - caught,
            "recall": caught / len(dropouts), "ci_low": low, "ci_high": high,
        })
    return pd.DataFrame(rows)
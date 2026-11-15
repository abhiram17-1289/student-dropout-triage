"""Choose model settings on the training classes only. The test classes are never loaded here."""
import itertools
from functools import partial

import numpy as np
import pandas as pd

from src import config
from src.data import load_students
from src.evaluation import invite, recall
from src.features import baseline_risk, build_inputs, common_codes
from src.models import dropout_risk, flexible_model, readable_model

LOGISTIC_GRID = [
    {"C": C, "class_weight": weight}
    for C in [0.01, 0.03, 0.1, 0.3, 1.0]
    for weight in [None, "balanced"]
]
BOOSTING_GRID = [
    dict(zip(["learning_rate", "max_depth", "max_iter", "min_samples_leaf"], values))
    for values in itertools.product([0.03, 0.1], [2, 3, 4], [100, 300], [20, 60])
]


def forward_validation_risk(students, make_model):
    """Risk for each validation-year student, from a model trained on all earlier years."""
    risk = pd.Series(np.nan, index=students.index)
    for year in config.VALIDATION_ENTRY_YEARS:
        earlier = students[students["entering_year"] < year]
        current = students[students["entering_year"] == year]
        # Code pooling is refitted on each fold's own training years.
        codes = common_codes(earlier)
        model = make_model().fit(build_inputs(earlier, codes), earlier["dropped_out"])
        risk[current.index] = dropout_risk(model, build_inputs(current, codes))
    return risk


def main():
    students = load_students()
    students = students[students["entering_year"].isin(config.TRAIN_YEARS)].reset_index(drop=True)
    validation = students["entering_year"].isin(config.VALIDATION_ENTRY_YEARS)
    dropped_out = students.loc[validation, "dropped_out"]
    years = students.loc[validation, "entering_year"]

    def validation_recall(risk):
        return recall(dropped_out, invite(risk[validation], years))

    rows = [{"model": "baseline", "settings": "share of units not passed",
             "recall": validation_recall(baseline_risk(students))}]
    for settings in LOGISTIC_GRID:
        risk = forward_validation_risk(students, partial(readable_model, **settings))
        rows.append({"model": "readable", "settings": str(settings), "recall": validation_recall(risk)})
    for settings in BOOSTING_GRID:
        risk = forward_validation_risk(students, partial(flexible_model, **settings))
        rows.append({"model": "flexible", "settings": str(settings), "recall": validation_recall(risk)})

    results = pd.DataFrame(rows)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.TABLES_DIR / "tuning.csv", index=False)

    print(f"Validation classes {config.VALIDATION_ENTRY_YEARS}: "
          f"{validation.sum()} students, {dropped_out.sum()} dropouts")
    print(f"baseline  recall {results.loc[0, 'recall']:.3f}")
    for model, chosen in [("readable", config.LOGISTIC_SETTINGS), ("flexible", config.BOOSTING_SETTINGS)]:
        # Stable sort: on a tie, the setting listed first in the grid wins.
        best = results[results["model"] == model].sort_values("recall", ascending=False, kind="stable").iloc[0]
        match = "matches config" if best["settings"] == str(chosen) else "DIFFERS from config"
        print(f"{model:9s} recall {best['recall']:.3f}  {best['settings']}  ({match})")


if __name__ == "__main__":
    main()
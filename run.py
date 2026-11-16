"""Train on the 2008-2014 classes, evaluate once on 2015-2017, and write every table and figure."""
import numpy as np
import pandas as pd

from src import config
from src.data import load_students
from src.evaluation import invite, recall, recall_by_slice, recall_difference_interval
from src.features import baseline_risk, build_inputs, common_codes
from src.models import dropout_risk, flexible_model, readable_coefficients, readable_model
from src.plots import (plot_coefficients, plot_group_recall, plot_recall_by_share,
                       plot_share_passed)
from src.unsupervised import (assign_clusters, describe_clusters, first_semester_components,
                              fit_clusters, invite_within_clusters, silhouette_by_k)

METHODS = ["readable", "flexible", "baseline", "random"]


def save(table, name):
    table.to_csv(config.TABLES_DIR / f"{name}.csv", index=False)


def group_table(test, invited, invited_baseline, column, label_prefix=""):
    """Model recall per group, with the baseline's recall alongside for the figure."""
    table = recall_by_slice(test, invited, column)
    table["baseline_recall"] = recall_by_slice(test, invited_baseline, column)["recall"].values
    table["label"] = label_prefix + table["group"].astype(str)
    return table

def best_possible_recall(test, share):
    """Recall if every invitation went to a dropout: slots, capped by dropouts, in each class."""
    reachable = sum(min(round(len(members) * share), members["dropped_out"].sum())
                    for _, members in test.groupby("entering_year"))
    return reachable / test["dropped_out"].sum()


def main():
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    everyone = load_students(keep_missing_first_semester=True)
    students = load_students()
    train = students[students["entering_year"].isin(config.TRAIN_YEARS)].reset_index(drop=True)
    test = students[students["entering_year"].isin(config.TEST_YEARS)].reset_index(drop=True)
    years, dropped_out = test["entering_year"], test["dropped_out"]

    # Fit everything on the training classes; the test classes are only scored.
    codes = common_codes(train)
    train_inputs, test_inputs = build_inputs(train, codes), build_inputs(test, codes)
    readable = readable_model().fit(train_inputs, train["dropped_out"])
    flexible = flexible_model().fit(train_inputs, train["dropped_out"])

    risk = {
        "readable": dropout_risk(readable, test_inputs),
        "flexible": dropout_risk(flexible, test_inputs),
        "baseline": baseline_risk(test).values,
        "random": np.random.default_rng(config.SEED).random(len(test)),
    }
    invited = {method: invite(risk[method], years) for method in METHODS}

    # Headline: recall at the locked 20%, beside the baseline, with the gap's interval.
    best_possible = best_possible_recall(test, config.SHARE_OF_CLASS_INVITED)
    headline = []
    for method in METHODS:
        caught = int(invited[method][dropped_out == 1].sum())
        low, high = (recall_difference_interval(dropped_out, years, invited[method], invited["baseline"])
                     if method in ("readable", "flexible") else (np.nan, np.nan))
        headline.append({
            "method": method, "recall": recall(dropped_out, invited[method]),
            "caught": caught, "missed": int(dropped_out.sum()) - caught,
            "invited": int(invited[method].sum()),
            "invited_who_dropped_out": dropped_out[invited[method]].mean(),
            "minus_baseline": recall(dropped_out, invited[method]) - recall(dropped_out, invited["baseline"]),
            "minus_baseline_ci_low": low, "minus_baseline_ci_high": high,
        })
    headline = pd.DataFrame(headline)
    save(headline, "headline")

    # The baseline's random tie-breaking, redone with 200 different seeds.
    baseline_by_seed = [recall(dropped_out, invite(risk["baseline"], years, seed=seed)) for seed in range(200)]
    save(pd.DataFrame({"seed": range(200), "baseline_recall": baseline_by_seed}), "baseline_tie_break_seeds")

    # Secondary: other shares of the class invited.
    curves = pd.DataFrame([
        {"method": method, "share": share, "recall": recall(dropped_out, invite(risk[method], years, share=share))}
        for method in METHODS for share in config.OTHER_SHARES_TO_CHECK
    ])
    save(curves, "recall_by_share")
    ceiling = pd.Series({share: best_possible_recall(test, share) for share in config.OTHER_SHARES_TO_CHECK})
    plot_recall_by_share(curves, ceiling, config.FIGURES_DIR / "recall_by_share.png")

    # Failure analysis on the slices named in the proposal.
    slices = pd.concat([
        recall_by_slice(test, invited[method], column).assign(method=method)
        for method in ("readable", "flexible", "baseline") for column in config.PROPOSAL_SLICES
    ])
    save(slices, "slices")
    slice_figure = pd.concat([
        group_table(test, invited["readable"], invited["baseline"], column, column.replace("_", " ") + ": ")
        for column in config.PROPOSAL_SLICES
    ])
    plot_group_recall(slice_figure, headline.loc[0, "recall"], config.FIGURES_DIR / "recall_by_slice.png",
                      "Recall by slice named in the proposal (test classes)")

    # Unsupervised structure: clusters fitted on training background, applied to test.
    clusters = fit_clusters(train)
    test["cluster"] = assign_clusters(clusters, test)
    save(describe_clusters(clusters, train).reset_index(), "clusters")
    cluster_figure = group_table(test, invited["readable"], invited["baseline"], "cluster", "cluster ")
    save(cluster_figure, "recall_by_cluster")
    plot_group_recall(cluster_figure, headline.loc[0, "recall"], config.FIGURES_DIR / "recall_by_cluster.png",
                      "Recall by enrollment-background cluster (test classes)")
    quota = invite_within_clusters(risk["readable"], years, test["cluster"])
    save(pd.DataFrame([{"policy": "rank whole class", "recall": recall(dropped_out, invited["readable"])},
                       {"policy": "share of each cluster", "recall": recall(dropped_out, quota)}]),
         "cluster_quota")
    save(silhouette_by_k(train), "silhouette_by_k")
    variance, loadings = first_semester_components(train)
    save(variance.rename_axis("component").reset_index(), "pca_variance")
    save(loadings.rename_axis("column").reset_index(), "pca_loadings")

    # What the readable model learned, and who it misses.
    coefficients = readable_coefficients(readable, train_inputs.columns)
    save(coefficients.rename_axis("input").reset_index(name="coefficient"), "coefficients")
    plot_coefficients(coefficients, config.FIGURES_DIR / "coefficients.png")
    share_passed = 1 - risk["baseline"]
    caught = (dropped_out == 1).values & invited["readable"]
    missed = (dropped_out == 1).values & ~invited["readable"]
    plot_share_passed({"Dropouts invited": share_passed[caught], "Dropouts missed": share_passed[missed],
                       "Did not drop out": share_passed[(dropped_out == 0).values]},
                      config.FIGURES_DIR / "share_passed.png")

    save(pd.DataFrame({"column": config.EXCLUDED_COLUMNS}), "excluded_columns")

    print(f"Students: {len(everyone)} in file, {len(everyone) - len(students)} without a first-semester "
          f"record removed, {len(train)} training, {len(test)} test ({int(dropped_out.sum())} dropouts)")
    print(f"Recall when inviting {config.SHARE_OF_CLASS_INVITED:.0%} of each test class "
          f"(best possible {best_possible:.1%}):")
    for row in headline.itertuples():
        gap = (f"  vs baseline {row.minus_baseline:+.1%} [{row.minus_baseline_ci_low:+.1%}, "
               f"{row.minus_baseline_ci_high:+.1%}]" if row.method in ("readable", "flexible") else "")
        print(f"  {row.method:9s} {row.recall:.1%}  caught {row.caught}, missed {row.missed}{gap}")
    print(f"Baseline over 200 tie-break seeds: {min(baseline_by_seed):.1%} to {max(baseline_by_seed):.1%}")
    print(f"Tables in {config.TABLES_DIR}, figures in {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
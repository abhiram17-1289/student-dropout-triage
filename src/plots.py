import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt

COLORS = {"readable": "#2a78d6", "flexible": "#1baf7a", "baseline": "#eb6834", "random": "#8c8b86"}
NAMES = {"readable": "L1 logistic", "flexible": "Gradient boosting",
         "baseline": "Baseline rule", "random": "Random"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Names from the UCI codebook, for the codes that appear in the figures.
COURSE_NAMES = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication", 9853: "Basic Education", 9991: "Management (evening)",
}
INPUT_NAMES = {
    "Curricular units 1st sem (approved)": "Sem 1 units passed",
    "Curricular units 1st sem (credited)": "Sem 1 units credited from earlier study",
    "Curricular units 1st sem (grade)": "Sem 1 average grade",
    "Curricular units 1st sem (enrolled)": "Sem 1 units enrolled",
    "Curricular units 1st sem (evaluations)": "Sem 1 assessments taken",
    "Curricular units 1st sem (without evaluations)": "Sem 1 units never assessed",
    "Gender": "Male",
    "Mother's qualification_34": "Mother's qualification unknown",
    "Father's qualification_34": "Father's qualification unknown",
    "Application mode_1": "Applied in first phase (general)",
    "Application mode_39": "Applied as over-23 entrant",
    "Daytime/evening attendance": "Daytime attendance",
}


def readable_name(column):
    if column in INPUT_NAMES:
        return INPUT_NAMES[column]
    if column.startswith("Course_") and column != "Course_other":
        return "Course: " + COURSE_NAMES[int(column.split("_")[1])]
    return column.replace("_", " = ")


def _tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_recall_by_share(curves, best_possible, path):
    """Recall at each share of the class invited. `curves` has columns method, share, recall."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(best_possible.index * 100, best_possible.values * 100, color=MUTED,
            linestyle="--", linewidth=1.2, label="Best possible")
    for method in ["readable", "flexible", "baseline", "random"]:
        rows = curves[curves["method"] == method]
        ax.plot(rows["share"] * 100, rows["recall"] * 100, color=COLORS[method], linewidth=2,
                marker="o", markersize=4, label=NAMES[method])
    ax.axvline(20, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(20.5, 3, "locked budget", color=MUTED, fontsize=8)
    ax.set_xlabel("Share of each entering class invited (%)", color=TEXT, fontsize=9)
    ax.set_ylabel("Dropouts reached (%)", color=TEXT, fontsize=9)
    ax.set_ylim(0, 102)
    _tidy(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_group_recall(table, overall_recall, path, title):
    """One row per group: model recall with its 95% interval, and the baseline's recall.

    `table` has columns label, dropouts, recall, ci_low, ci_high, baseline_recall.
    """
    table = table.iloc[::-1]  # first row at the top
    rows = range(len(table))
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(table) + 1.1))
    ax.axvline(overall_recall * 100, color=MUTED, linewidth=0.8, linestyle="--", label="L1 logistic, all students")
    ax.hlines(rows, table["ci_low"] * 100, table["ci_high"] * 100, color=COLORS["readable"], linewidth=2)
    ax.plot(table["recall"] * 100, rows, "o", color=COLORS["readable"], markersize=6, label="L1 logistic (95% interval)")
    ax.plot(table["baseline_recall"] * 100, rows, "D", markerfacecolor="white",
            markeredgecolor=COLORS["baseline"], markeredgewidth=1.5, markersize=5, label="Baseline rule")
    ax.set_yticks(list(rows), [f"{label}  ({n} dropouts)" for label, n in zip(table["label"], table["dropouts"])],
                  fontsize=8, color=TEXT)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Dropouts reached at the locked budget (%)", color=TEXT, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    _tidy(ax)
    ax.grid(axis="y", visible=False)
    # Legend under the axis so it never covers a row.
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.35, -0.14), ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_coefficients(coefficients, path, top=12):
    """The readable model's largest coefficients, per standard deviation of each input."""
    shown = coefficients.head(top).iloc[::-1]
    colors = [COLORS["baseline"] if value > 0 else COLORS["readable"] for value in shown]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(shown) + 1.0))
    ax.barh(range(len(shown)), shown.values, color=colors, height=0.6)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(range(len(shown)), [readable_name(c) for c in shown.index], fontsize=8, color=TEXT)
    ax.set_xlabel("Change in log-odds of dropping out per standard deviation\n"
                  "(orange raises risk, blue lowers it)", color=TEXT, fontsize=9)
    _tidy(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_share_passed(groups, path):
    """Distribution of first-semester share of units passed for each group of students.

    `groups` maps a label to the share passed of the students in that group.
    """
    # Own colours: blue and orange already mean the logistic model and the baseline elsewhere.
    styles = ["#6250d6", "#e34948", COLORS["random"]]
    bins = [i / 10 for i in range(11)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for (label, shares), color in zip(groups.items(), styles):
        weights = [100 / len(shares)] * len(shares)  # percentages, so groups of different size compare
        ax.hist(shares, bins=bins, weights=weights, histtype="step", linewidth=2, color=color,
                label=f"{label} ({len(shares)})")
    ax.set_xlabel("Share of first-semester units passed", color=TEXT, fontsize=9)
    ax.set_ylabel("Students in group (%)", color=TEXT, fontsize=9)
    _tidy(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
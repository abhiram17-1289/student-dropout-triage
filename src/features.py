import pandas as pd

from src import config

NUMERIC_COLUMNS = config.KNOWN_AT_ENROLLMENT_COLUMNS + config.FIRST_SEMESTER_COLUMNS
INPUT_COLUMNS = NUMERIC_COLUMNS + config.CATEGORY_CODE_COLUMNS

# Stop immediately if an excluded column ever ends up among the inputs.
leaked = set(INPUT_COLUMNS) & set(config.EXCLUDED_COLUMNS)
assert not leaked, f"excluded columns used as inputs: {leaked}"


def common_codes(train_students):
    """Codes each category column keeps, counted on the training students only."""
    kept = {}
    for column in config.CATEGORY_CODE_COLUMNS:
        counts = train_students[column].value_counts()
        kept[column] = sorted(counts[counts >= config.MIN_STUDENTS_PER_CODE].index)
    return kept


def build_inputs(students, kept_codes):
    inputs = students[NUMERIC_COLUMNS].astype(float)
    for column, codes in kept_codes.items():
        # Rare codes, and codes never seen in training, become "other".
        grouped = students[column].where(students[column].isin(codes), "other")
        # Fixing the categories gives train and test exactly the same columns.
        grouped = pd.Categorical(grouped, categories=list(codes) + ["other"])
        # One 0/1 column per code.
        dummies = pd.get_dummies(grouped, prefix=column, dtype=float)
        dummies.index = students.index
        inputs = inputs.join(dummies)
    return inputs


def baseline_risk(students):
    """The locked baseline: share of enrolled first-semester units not passed."""
    passed = students["Curricular units 1st sem (approved)"]
    enrolled = students["Curricular units 1st sem (enrolled)"]
    # Higher means more at risk, the same direction as the models' probabilities.
    return 1 - passed / enrolled
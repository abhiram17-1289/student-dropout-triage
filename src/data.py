import pandas as pd

from src import config


def entering_years(unemployment_rates, series_path=config.UNEMPLOYMENT_PATH,
                   year_range=config.ENROLLMENT_YEARS, tolerance=0.05):
    series = pd.read_csv(series_path)
    series.columns = ["date", "rate"]
    series["year"] = pd.to_datetime(series["date"]).dt.year
    window = series[series["year"].between(*year_range)]

    year_for_rate = {}
    for rate in sorted(unemployment_rates.unique()):
        # The student data file rounds to one decimal, so a match is within 0.05.
        matches = window.loc[(window["rate"] - rate).abs() <= tolerance, "year"].tolist()
        if len(matches) != 1:
            raise ValueError(f"unemployment rate {rate} matches years {matches}, expected exactly one")
        year_for_rate[rate] = matches[0]
    return unemployment_rates.map(year_for_rate)


def load_students(path=config.DATA_PATH, keep_missing_first_semester=False):
    students = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    students.columns = students.columns.str.strip()  # one header ends in a tab

    students["entering_year"] = entering_years(students["Unemployment rate"])
    students["dropped_out"] = (students["Target"] == "Dropout").astype(int)

    students["scholarship_holder"] = students["Scholarship holder"].map({1: "yes", 0: "no"})
    students["gender"] = students["Gender"].map({1: "male", 0: "female"})
    students["living_away_from_home"] = students["Displaced"].map({1: "yes", 0: "no"})
    edges = [config.AGE_BANDS[0][0] - 1] + [oldest for _, oldest, _ in config.AGE_BANDS]
    students["age_band"] = pd.cut(students["Age at enrollment"], bins=edges,
                                  labels=[label for _, _, label in config.AGE_BANDS]).astype(str)

    # Zero units enrolled means no first-semester record (all of Animation and
    # Multimedia Design), not a student who failed everything.
    has_record = students["Curricular units 1st sem (enrolled)"] > 0
    if not keep_missing_first_semester:
        students = students[has_record]
    return students.reset_index(drop=True)
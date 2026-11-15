from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "students.csv"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

SEED = 1704

# The file has no year column. data.py recovers each student's entering year by
# matching the file's unemployment rate to Portugal's official annual rate
# (OECD, via FRED: https://fred.stlouisfed.org/series/LRUNTTTTPTA156S).
UNEMPLOYMENT_PATH = ROOT / "data" / "portugal_unemployment_fred.csv"
# Enrollment years stated by the dataset's authors (Realinho et al., 2022), 2008/09 to 2018/19.
ENROLLMENT_YEARS = (2008, 2018)

# Locked in the proposal.
TRAIN_YEARS = [2008, 2009, 2010, 2011, 2012, 2013, 2014]
TEST_YEARS = [2015, 2016, 2017]
SHARE_OF_CLASS_INVITED = 0.20

# Tuning only: each year is predicted by a model trained on all earlier years.
VALIDATION_ENTRY_YEARS = [2011, 2012, 2013, 2014]
OTHER_SHARES_TO_CHECK = [0.10, 0.20, 0.30, 0.40, 0.50]

CATEGORY_CODE_COLUMNS = [
    "Marital status", "Application mode", "Course", "Previous qualification",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]
KNOWN_AT_ENROLLMENT_COLUMNS = [
    "Application order", "Daytime/evening attendance", "Previous qualification (grade)",
    "Admission grade", "Displaced", "Educational special needs", "Gender",
    "Scholarship holder", "Age at enrollment", "International",
]
FIRST_SEMESTER_COLUMNS = [
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
]

EXCLUDED_COLUMNS = [
    "Curricular units 2nd sem (credited)",             # second semester: not known yet when the adviser decides
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Debtor",                                          # timing undocumented, may record events after leaving
    "Tuition fees up to date",                         # timing undocumented, may record events after leaving
    "Unemployment rate",                               # same for everyone in a class
    "Inflation rate",                                  # same for everyone in a class
    "GDP",                                             # same for everyone in a class
]

# In the category columns, values held by fewer than 40 training students are merged into "other".
MIN_STUDENTS_PER_CODE = 40

# Chosen by tune.py.
LOGISTIC_SETTINGS = {"C": 0.03,  # inverse of penalty strength: smaller means stronger
                     "class_weight": "balanced"}
BOOSTING_SETTINGS = {"learning_rate": 0.03, "max_depth": 2, "max_iter": 300, "min_samples_leaf": 60}

PROPOSAL_SLICES = ["scholarship_holder", "age_band", "gender", "living_away_from_home"]
AGE_BANDS = [(0, 20, "20 or under"), (21, 25, "21 to 25"), (26, 120, "over 25")]
BOOTSTRAP_RESAMPLES = 2000

BACKGROUND_COLUMNS_FOR_CLUSTERING = [
    "Age at enrollment", "Admission grade", "Previous qualification (grade)", "Displaced",
    "Gender", "Daytime/evening attendance", "Application order", "International",
]   

NUMBER_OF_CLUSTERS = 4
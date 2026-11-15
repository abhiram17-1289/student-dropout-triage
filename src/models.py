import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def readable_model(**settings):
    settings = {**config.LOGISTIC_SETTINGS, **settings}
    # Scaling first puts every input on the same footing, so the L1 penalty treats them equally.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l1", solver="liblinear", random_state=config.SEED, **settings),
    )


def flexible_model(**settings):
    settings = {**config.BOOSTING_SETTINGS, **settings}
    # Early stopping off: every tree is built, and no part of the training data is held back.
    return HistGradientBoostingClassifier(early_stopping=False, random_state=config.SEED, **settings)


def dropout_risk(model, inputs):
    """Predicted probability of dropping out, used to rank students."""
    return model.predict_proba(inputs)[:, 1]


def readable_coefficients(model, input_names):
    """Non-zero coefficients, per standard deviation of each input, largest effect first."""
    coefficients = pd.Series(model[-1].coef_[0], index=input_names)
    coefficients = coefficients[coefficients != 0]
    return coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)
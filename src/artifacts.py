"""Save and load the trained model bundle.

The bundle is a contract: anything needed to use the model correctly must
travel with the model. Feature names, lead time and service level live next
to the fitted models so the app can never score with settings the models
were not trained on.
"""

from datetime import datetime
from pathlib import Path

import joblib
import lightgbm
import sklearn


def save_bundle(path: str, point_model, quantile_model, features: list[str],
                config: dict, metrics: dict) -> None:
    """Write both models plus the context needed to trust them.

    Library versions are recorded because a pickled model is only guaranteed
    to load under the versions that wrote it.
    """
    bundle = {
        "point_model": point_model,
        "quantile_model": quantile_model,
        "features": list(features),
        "lead_time_weeks": config["inventory"]["lead_time_weeks"],
        "service_level": config["inventory"]["service_level"],
        "train_end": config["split"]["train_end"],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
        "versions": {"lightgbm": lightgbm.__version__,
                     "sklearn": sklearn.__version__},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str) -> dict:
    """Return the bundle dict exactly as it was saved."""
    return joblib.load(path)

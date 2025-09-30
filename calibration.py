from vetris.optimize.pipeline import run_pipeline
from vetris.optimize.config import CalibConfig



if __name__ == "__main__":
    cfg = CalibConfig(
        experiment_csv="data/real/force_indentation.csv",
        csv_is_mm=False,
        massager_type="straight",
        override_time_period=None,
        do_coarse=True,
        do_fine=True,
        coarse_trials=150,
        coarse_sigma=2.0,
        exp_subdir="exp_15",
        weights=(50.0, 0.0, 0.00, 100.0)
    )
    run_pipeline(cfg)


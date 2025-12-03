import sys
import os
import pandas as pd
import numpy as np
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_core import run_automl_regression

def test_run_automl_regression():
    # Create dummy regression data
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.rand(100) * 100
    })

    output_dirs = {
        "models": "tests/output/models",
        "reports": "tests/output/reports",
        "preprocessed": "tests/output/preprocessed",
        "static": "tests/output/static",
    }

    # Clean up previous output
    for d in output_dirs.values():
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        # Create plots subdir
        if d.endswith("static"):
             os.makedirs(os.path.join(d, "plots"), exist_ok=True)

    try:
        results = run_automl_regression(
            df, 'target', output_dirs,
            test_size=0.2,
            random_state=42,
            use_optuna=False,
            compute_shap=False
        )

        assert results['best_model_name'] is not None
        assert results['task_type'] == 'regression'
        assert os.path.exists(os.path.join(output_dirs['models'], results['model_path']))
        assert os.path.exists(os.path.join(output_dirs['static'], results['actual_vs_predicted_path']))
        
        print("Regression test passed!")
        print(f"Best model: {results['best_model_name']}")

    except Exception as e:
        print(f"Regression test failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        for d in output_dirs.values():
            if os.path.exists(d):
                shutil.rmtree(d)

if __name__ == "__main__":
    test_run_automl_regression()

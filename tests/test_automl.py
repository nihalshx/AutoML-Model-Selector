import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Add parent directory to path to import automl_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_core import run_automl_classification

def test_automl_run():
    print("Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=200, 
        n_features=10, 
        n_informative=5, 
        n_redundant=2, 
        n_classes=2, 
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    df["target"] = y
    
    # Create dummy output directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dirs = {
        "models": os.path.join(base_dir, "test_outputs", "models"),
        "reports": os.path.join(base_dir, "test_outputs", "reports"),
        "preprocessed": os.path.join(base_dir, "test_outputs", "preprocessed"),
        "static": os.path.join(base_dir, "test_outputs", "static"),
    }
    
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)
        
    print("Running AutoML classification...")
    try:
        results = run_automl_classification(
            df, 
            target_col="target",
            output_dirs=output_dirs,
            use_optuna=False, # Keep it fast for testing
            compute_shap=True,
            shap_max_samples=50
        )
        
        print("\n--- Test Results ---")
        print(f"Run ID: {results['run_id']}")
        print(f"Best Model: {results['best_model_name']}")
        print("Models Leaderboard:")
        for m in results['models_metrics']:
            print(f"  - {m['model_name']}: F1={m['f1']:.4f}, Acc={m['accuracy']:.4f}")
            
        # Verify files exist
        assert os.path.exists(os.path.join(output_dirs["models"], results["model_path"])), "Model file missing"
        assert os.path.exists(os.path.join(output_dirs["preprocessed"], results["preprocessed_path"])), "Preprocessed file missing"
        assert os.path.exists(os.path.join(output_dirs["reports"], results["html_report_filename"])), "Report file missing"
        
        print("\nSUCCESS: All outputs generated and logic verified.")
        
    except Exception as e:
        print(f"\nFAILURE: {e}")
        raise e

if __name__ == "__main__":
    test_automl_run()

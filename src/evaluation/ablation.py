from src.models.ensemble import EnsemblePredictor


def run_ablation(lstm_pred, xgb_pred, y_true, output_path="reports/ablation_results.json"):
    ensemble = EnsemblePredictor()
    ensemble.learn_weights(lstm_pred, xgb_pred, y_true)
    return ensemble.ablation_report(lstm_pred, xgb_pred, y_true, output_path)

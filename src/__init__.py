"""CanScan-Extra - Source package."""

__all__ = [
    "prepare_data_pipeline",
    "load_data",
    "load_and_verify_data",
    "explore_classes",
    "run_exploration",
    "split_data",
    "create_classification_splits",
    "create_tf_datasets",
    "draw_count_plot",
    "show_examples",
    "draw_split_distribution",
    "build_googlenetv3",
    "export_model_summary",
    "setup_fine_tuning",
    "get_callbacks",
    "train_phase1_transfer",
    "train_phase2_finetune",
    "evaluate_classifier",
]


def __getattr__(name):
    if name in ("prepare_data_pipeline", "load_data"):
        from src.classification.data_loader import prepare_data_pipeline, load_data
        return locals()[name]
    elif name in ("load_and_verify_data", "explore_classes", "run_exploration"):
        from src.classification.exploration import load_and_verify_data, explore_classes, run_exploration
        return locals()[name]
    elif name in ("split_data", "create_classification_splits", "create_tf_datasets"):
        from src.classification.processing import split_data, create_classification_splits, create_tf_datasets
        return locals()[name]
    elif name in ("draw_count_plot", "show_examples", "draw_split_distribution"):
        from src.classification.charts import draw_count_plot, show_examples, draw_split_distribution
        return locals()[name]
    elif name in (
        "build_googlenetv3",
        "export_model_summary",
        "setup_fine_tuning",
        "get_callbacks",
        "train_phase1_transfer",
        "train_phase2_finetune",
        "evaluate_classifier",
    ):
        from src.classification.classifier import (
            build_googlenetv3,
            export_model_summary,
            setup_fine_tuning,
            get_callbacks,
            train_phase1_transfer,
            train_phase2_finetune,
            evaluate_classifier,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

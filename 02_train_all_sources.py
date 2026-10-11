"""
Trains NB/DT/SVM for every source in config.STRUCTURED_SOURCES, evaluates,
and saves models + metrics into ./models/.

Run 01_inspect_data.py first and fix any target_column / categorical_columns
mismatches in config.py.
"""

from config import STRUCTURED_SOURCES
from src.train_source_models import train_source

if __name__ == "__main__":
    summary = {}
    for source_name, cfg in STRUCTURED_SOURCES.items():
        try:
            metrics, best = train_source(source_name, cfg)
            summary[source_name] = best
        except Exception as e:
            optional = cfg.get("optional", False)
            level = "WARNING (optional source, skipped)" if optional else "ERROR"
            print(f"\n[{level}] Could not train '{source_name}': {e}")
            if not optional:
                raise

    print("\n=== Summary: selected model per source ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

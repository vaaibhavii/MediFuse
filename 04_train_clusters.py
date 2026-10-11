from config import STRUCTURED_SOURCES
from src.clustering import train_clusters_for_source

if __name__ == "__main__":
    for source_name, cfg in STRUCTURED_SOURCES.items():
        try:
            train_clusters_for_source(source_name, cfg)
        except Exception as e:
            print(f"[skip] {source_name}: {e}")

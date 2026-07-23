"""Helpers for using a project's latest EcoTaxa validation export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"object_id", "object_annotation_status"}


def validated_object_ids(project: Path, logger) -> set[str] | None:
    """Read validations from the newest usable Object or Classification Export."""
    data_dir = project / "data"
    candidates = sorted(data_dir.glob("*.tsv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        logger.info("No EcoTaxa export found; predicting all objects")
        return None

    for export_path in candidates:
        try:
            export = pd.read_csv(export_path, sep="\t", quotechar='"', dtype=str, low_memory=False)
        except Exception as exc:
            logger.warning(f"Could not read TSV '{export_path}': {exc}")
            continue
        if not REQUIRED_COLUMNS.issubset(export.columns):
            continue

        object_ids = export["object_id"].fillna("").astype(str).str.strip().str.strip('"')
        statuses = export["object_annotation_status"].fillna("").astype(str).str.strip().str.strip('"').str.lower()
        validated = set(object_ids[(statuses == "validated") & (object_ids != "") & (object_ids != "[t]")])
        logger.info(f"Using EcoTaxa validation export '{export_path.name}': skipping {len(validated)} validated object(s)")
        return validated

    logger.warning("No usable EcoTaxa Object or Classification Export found; predicting all objects")
    return None

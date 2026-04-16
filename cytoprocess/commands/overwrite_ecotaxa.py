from pathlib import Path

import pandas as pd
import requests
import yaml

from cytoprocess import ecotaxa
from cytoprocess.logging import log_command_start, log_command_success, setup_logging
from cytoprocess.project import list_sample_assets, path_to_sample_asset
from cytoprocess.utils import raiseCytoError


def _get_prediction_files(project: Path, sample_filter: str | None) -> list[Path]:
    prediction_files = sorted(project.glob("work/*/predictions.parquet"))
    if sample_filter:
        prediction_files = [path for path in prediction_files if path.parent.name == sample_filter]
    return prediction_files


def _normalise_sequence(value) -> list:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if converted is None:
            return []
        if isinstance(converted, list):
            return converted
        return [converted]
    return [value]


def _prepare_updates(
    predictions_df: pd.DataFrame,
    object_map: dict[str, int],
) -> tuple[list[int], list[list[int]], list[list[float]], int]:
    if "object_id" not in predictions_df.columns:
        raise ValueError("Prediction parquet must contain 'object_id'")

    df = predictions_df.copy()
    df["object_id"] = df["object_id"].astype(str)
    df = df.drop_duplicates(subset=["object_id"], keep="last")

    target_ids: list[int] = []
    classifications: list[list[int]] = []
    scores: list[list[float]] = []
    missing = 0

    for row in df.itertuples(index=False):
        ecotaxa_object_id = object_map.get(row.object_id)
        if ecotaxa_object_id is None:
            missing += 1
            continue

        if hasattr(row, "object_annotation_category_ids"):
            raw_category_ids = _normalise_sequence(getattr(row, "object_annotation_category_ids"))
            raw_probability_values = _normalise_sequence(getattr(row, "object_annotation_probabilities"))

            category_ids_series = pd.to_numeric(
                pd.Series(raw_category_ids),
                errors="coerce",
            )
            probability_series = pd.to_numeric(
                pd.Series(raw_probability_values),
                errors="coerce",
            ).fillna(1.0).clip(0.0, 1.0)

            max_len = max(len(category_ids_series), len(probability_series))
            if max_len == 0:
                continue

            category_ids_series = category_ids_series.reindex(range(max_len))
            probability_series = probability_series.reindex(range(max_len), fill_value=1.0)

            valid_pairs = [
                (int(category_id), float(probability))
                for category_id, probability in zip(category_ids_series.tolist(), probability_series.tolist())
                if pd.notna(category_id)
            ]

            category_ids = [category_id for category_id, _ in valid_pairs]
            probability_values = [probability for _, probability in valid_pairs]
        else:
            if not hasattr(row, "object_annotation_category_id"):
                raise ValueError(
                    "Prediction parquet must contain 'object_annotation_category_id' or 'object_annotation_category_ids'"
                )

            category_id = pd.to_numeric(
                pd.Series([getattr(row, "object_annotation_category_id", None)]),
                errors="coerce",
            ).dropna()
            if category_id.empty:
                continue
            category_ids = [int(category_id.iloc[0])]

            score_col = next(
                (
                    column
                    for column in (
                        "object_annotation_score",
                        "object_annotation_probability",
                        "object_annotation_confidence",
                    )
                    if hasattr(row, column)
                ),
                None,
            )
            probability = getattr(row, score_col) if score_col else 1.0
            probability_values = [
                float(
                    pd.to_numeric(pd.Series([probability]), errors="coerce")
                    .fillna(1.0)
                    .clip(0.0, 1.0)
                    .iloc[0]
                )
            ]

        if not category_ids:
            continue
        if len(probability_values) < len(category_ids):
            probability_values.extend([1.0] * (len(category_ids) - len(probability_values)))

        target_ids.append(ecotaxa_object_id)
        classifications.append(category_ids)
        scores.append(probability_values[: len(category_ids)])

    return target_ids, classifications, scores, missing


def run(ctx, project, username: str | None = None, password: str | None = None):
    logger = setup_logging(command="overwrite_ecotaxa", project=project, debug=ctx.obj["debug"])

    log_command_start(logger, "Updating EcoTaxa predictions for existing samples", project)
    logger.debug("Context: %s", getattr(ctx, "obj", {}))

    project = Path(project)
    sample_filter = getattr(ctx, "obj", {}).get("sample")

    config_path = project / "config" / "config.yaml"
    if not config_path.exists():
        raiseCytoError(f"Config file not found: '{config_path}', run 'cytoprocess create {project}' again.", logger)

    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle) or {}

    ecotaxa_config = config.get("ecotaxa", {}) or {}
    project_id = ecotaxa_config.get("project_id")
    eco_url = ecotaxa_config.get("url", "https://ecotaxa.obs-vlfr.fr")
    api_url = f"{eco_url}/api"
    if not project_id:
        raiseCytoError(
            f"EcoTaxa project_id missing from '{config_path}'\n"
            "Edit the file to set 'ecotaxa: project_id'\n"
            "You can find your EcoTaxa numeric project ID in the table at\n"
            f"  {eco_url}/prj",
            logger,
        )

    prediction_files = _get_prediction_files(project, sample_filter)
    if not prediction_files:
        raiseCytoError(
            f"No prediction parquet files found in '{project / 'work'}', "
            f"run 'cytoprocess predict_images {project}' first.",
            logger,
        )

    token = ecotaxa.authenticate(api_url, username=username, password=password, logger=logger)
    if token is None:
        raiseCytoError("Authentication failed, cannot proceed with EcoTaxa update", logger)

    project_info = ecotaxa.get_project_info(api_url, int(project_id), token, logger)
    project_name = project_info.get("title", "Unknown") if project_info else "Unknown"
    logger.info(f"Updating EcoTaxa project '{project_name}' [{project_id}]")

    project_samples = ecotaxa.get_project_samples(api_url, int(project_id), token, logger)
    if not project_samples:
        raiseCytoError("No samples could be retrieved from EcoTaxa for this project.", logger)

    logger.info(f"Found {len(prediction_files)} prediction file(s) to sync")

    total_updated = 0
    total_missing_samples = 0
    total_missing_objects = 0

    for prediction_file in prediction_files:
        sample_id = prediction_file.parent.name
        logger.info(f"'{sample_id}'")

        sample_ecotaxa_id = project_samples.get(sample_id)
        if sample_ecotaxa_id is None:
            total_missing_samples += 1
            logger.warning("  Sample not found in EcoTaxa project, skipping")
            continue

        predictions_df = pd.read_parquet(prediction_file)
        if predictions_df.empty:
            logger.warning("  Prediction file is empty, skipping")
            continue

        logger.info(f"  Reading {len(predictions_df)} predicted object(s)")
        object_map: dict[str, int] = {}
        window_start = 0
        while True:
            payload = requests.post(
                f"{api_url}/object_set/{int(project_id)}/query",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "fields": "obj.orig_id",
                    "window_start": window_start,
                    "window_size": 1000,
                },
                json={"samples": str(sample_ecotaxa_id)},
                timeout=120,
            )
            if payload.status_code != 200:
                raiseCytoError(f"Failed to query EcoTaxa objects: {payload.text}", logger)

            response_payload = payload.json()
            object_ids = response_payload.get("object_ids", [])
            details = response_payload.get("details", [])
            total_ids = int(response_payload.get("total_ids", 0) or 0)
            if not object_ids:
                break
            for object_id, detail in zip(object_ids, details):
                if detail and detail[0] is not None:
                    object_map[str(detail[0])] = int(object_id)
            window_start += len(object_ids)
            if window_start >= total_ids:
                break

        if not object_map:
            logger.warning("  No objects found in EcoTaxa for this sample, skipping")
            continue

        try:
            target_ids, batch_classifications, batch_scores, missing_objects = _prepare_updates(predictions_df, object_map)
        except ValueError as exc:
            raiseCytoError(f"Invalid prediction file '{prediction_file.name}': {exc}", logger)

        total_missing_objects += missing_objects

        if missing_objects:
            logger.warning(f"  {missing_objects} predicted object(s) were not found in EcoTaxa")
        if not target_ids:
            logger.warning("  No matching objects to update, skipping")
            continue

        logger.info(f"  Updating {len(target_ids)} EcoTaxa object(s)")
        updated = 0
        total = len(target_ids)
        for start in range(0, total, 1000):
            end = start + 1000
            batch_target_ids = target_ids[start:end]
            batch_classifications = batch_classifications[start:end]
            batch_scores = batch_scores[start:end]
            batch_number = (start // 1000) + 1
            batch_end = min(end, total)
            logger.info(
                f"    Batch {batch_number}: sending {len(batch_target_ids)} object(s) "
                f"({start + 1}-{batch_end}/{total})"
            )
            response = requests.post(
                f"{api_url}/object_set/classify_auto_multiple",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "target_ids": batch_target_ids,
                    "classifications": batch_classifications,
                    "scores": batch_scores,
                    "keep_log": True,
                },
                timeout=120,
            )
            if response.status_code != 200:
                raiseCytoError(f"Failed to update EcoTaxa object metadata: {response.text}", logger)
            batch_updated = int(response.json() or 0)
            updated += batch_updated
            logger.info(
                f"    Batch {batch_number}: updated {batch_updated} object(s) "
                f"(total {updated}/{total})"
            )
        total_updated += updated
        logger.info(f"  > Updated {updated} object(s)")

    logger.info(
        "Summary: "
        f"{total_updated} object(s) updated, "
        f"{total_missing_samples} sample(s) missing in EcoTaxa, "
        f"{total_missing_objects} object(s) not matched"
    )
    logger.info(f"Your data is at {eco_url}/prj/{project_id}")
    log_command_success(logger, "Overwrite EcoTaxa metadata")

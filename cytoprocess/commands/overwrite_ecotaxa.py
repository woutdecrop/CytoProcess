import getpass
import logging
from pathlib import Path

import keyring
import pandas as pd
import requests
import yaml

from cytoprocess.utils import (
    log_command_start,
    log_command_success,
    raiseCytoError,
    setup_logging,
)

ECOTAXA_API_URL = "https://ecotaxa.obs-vlfr.fr/api"
KEYRING_SERVICE = "cytoprocess-ecotaxa"
OBJECT_QUERY_WINDOW_SIZE = 1000
CLASSIFY_BATCH_SIZE = 1000


def _get_stored_token(logger: logging.Logger) -> str | None:
    try:
        return keyring.get_password(KEYRING_SERVICE, "token")
    except Exception as exc:
        logger.debug(f"Could not retrieve token from keyring: {exc}")
        return None


def _store_token(logger: logging.Logger, token: str) -> bool:
    try:
        keyring.set_password(KEYRING_SERVICE, "token", token)
        return True
    except Exception as exc:
        logger.warning(f"Could not store token in keyring: {exc}")
        return False


def _clear_token(logger: logging.Logger) -> None:
    try:
        keyring.delete_password(KEYRING_SERVICE, "token")
    except Exception:
        pass


def _validate_token(logger: logging.Logger, token: str) -> bool:
    try:
        response = requests.get(
            f"{ECOTAXA_API_URL}/users/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def _login(logger: logging.Logger, username: str, password: str) -> str | None:
    try:
        response = requests.post(
            f"{ECOTAXA_API_URL}/login",
            json={"username": username, "password": password},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        logger.error(f"Login failed: {response.text}")
        return None
    except requests.RequestException as exc:
        logger.error(f"Login request failed: {exc}")
        return None


def _get_user_info(logger: logging.Logger, token: str) -> dict | None:
    try:
        response = requests.get(
            f"{ECOTAXA_API_URL}/users/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None


def _get_project_info(logger: logging.Logger, token: str, project_id: int) -> dict | None:
    try:
        response = requests.get(
            f"{ECOTAXA_API_URL}/projects/{project_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        if response.status_code == 403:
            logger.error(f"Access denied to project {project_id}")
        elif response.status_code == 404:
            logger.error(f"Project {project_id} not found")
        return None
    except requests.RequestException as exc:
        logger.error(f"Failed to get project info: {exc}")
        return None


def _get_project_samples(logger: logging.Logger, token: str, project_id: int) -> dict[str, int]:
    try:
        response = requests.get(
            f"{ECOTAXA_API_URL}/samples/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"project_ids": str(project_id), "id_pattern": "*"},
            timeout=60,
        )
        if response.status_code != 200:
            logger.warning(f"Failed to get samples: {response.text}")
            return {}

        sample_map = {}
        for sample in response.json():
            orig_id = sample.get("orig_id")
            sample_id = sample.get("sampleid")
            if orig_id and sample_id is not None:
                sample_map[str(orig_id)] = int(sample_id)
        return sample_map
    except requests.RequestException as exc:
        logger.warning(f"Failed to get project samples: {exc}")
        return {}


def _get_sample_object_map(
    logger: logging.Logger,
    token: str,
    project_id: int,
    sample_ecotaxa_id: int,
) -> dict[str, int]:
    object_map: dict[str, int] = {}
    window_start = 0

    while True:
        response = requests.post(
            f"{ECOTAXA_API_URL}/object_set/{project_id}/query",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "fields": "obj.orig_id",
                "window_start": window_start,
                "window_size": OBJECT_QUERY_WINDOW_SIZE,
            },
            json={"samples": str(sample_ecotaxa_id)},
            timeout=120,
        )

        if response.status_code != 200:
            raiseCytoError(f"Failed to query EcoTaxa objects: {response.text}", logger)

        payload = response.json()
        object_ids = payload.get("object_ids", [])
        details = payload.get("details", [])
        total_ids = int(payload.get("total_ids", 0) or 0)

        if not object_ids:
            break

        for object_id, detail in zip(object_ids, details):
            if not detail:
                continue
            orig_id = detail[0]
            if orig_id is None:
                continue
            object_map[str(orig_id)] = int(object_id)

        window_start += len(object_ids)
        if window_start >= total_ids:
            break

    return object_map


def _classify_objects(
    logger: logging.Logger,
    token: str,
    target_ids: list[int],
    classifications: list[list[int]],
    scores: list[list[float]],
) -> int:
    updated = 0
    total = len(target_ids)

    for start in range(0, total, CLASSIFY_BATCH_SIZE):
        end = start + CLASSIFY_BATCH_SIZE
        batch_target_ids = target_ids[start:end]
        batch_classifications = classifications[start:end]
        batch_scores = scores[start:end]
        batch_number = (start // CLASSIFY_BATCH_SIZE) + 1
        batch_end = min(end, total)

        logger.info(
            f"    Batch {batch_number}: sending {len(batch_target_ids)} object(s) "
            f"({start + 1}-{batch_end}/{total})"
        )

        payload = {
            "target_ids": batch_target_ids,
            "classifications": batch_classifications,
            "scores": batch_scores,
            "keep_log": True,
        }

        response = requests.post(
            f"{ECOTAXA_API_URL}/object_set/classify_auto_multiple",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
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

    return updated


def authenticate(logger: logging.Logger, username: str | None = None, password: str | None = None) -> str | None:
    token = _get_stored_token(logger)
    if token and _validate_token(logger, token):
        user_info = _get_user_info(logger, token)
        if user_info:
            logger.info(f"Authenticated as: {user_info.get('name', 'Unknown')} ({user_info.get('email', 'Unknown')})")
        return token
    if token:
        logger.warning("Stored token is invalid, need to re-authenticate")
        _clear_token(logger)

    if not username:
        print("\nEcoTaxa Authentication Required")
        username = input("username (email): ").strip()
    if not username:
        raiseCytoError("EcoTaxa username is required", logger)

    if not password:
        password = getpass.getpass("password: ")
    if not password:
        raiseCytoError("EcoTaxa password is required", logger)

    token = _login(logger, username, password)
    if token is None:
        raiseCytoError("Authentication failed. Please check your EcoTaxa username and password.", logger)

    if _store_token(logger, token):
        logger.info("Authentication token stored securely in system keyring")

    user_info = _get_user_info(logger, token)
    if user_info:
        logger.info(f"Authenticated as: {user_info.get('name', 'Unknown')} ({user_info.get('email', 'Unknown')})")

    return token


def _get_prediction_files(project: Path, sample_filter: str | None) -> list[Path]:
    work_dir = project / "work"
    if not work_dir.exists():
        return []

    prediction_files = sorted(work_dir.glob("*_image_predictions_top3.parquet"))
    if not prediction_files:
        prediction_files = sorted(work_dir.glob("*_image_predictions.parquet"))

    if sample_filter:
        expected_top3 = work_dir / f"{sample_filter}_image_predictions_top3.parquet"
        expected_top1 = work_dir / f"{sample_filter}_image_predictions.parquet"
        expected = expected_top3 if expected_top3.exists() else expected_top1
        prediction_files = [path for path in prediction_files if path == expected]

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
    if not project_id:
        raiseCytoError(
            f"EcoTaxa project_id missing from '{config_path}'\n"
            "Edit the file to set 'ecotaxa: project_id'\n"
            "You can find your EcoTaxa numeric project ID in the table at\n"
            "  https://ecotaxa.obs-vlfr.fr/prj",
            logger,
        )

    prediction_files = _get_prediction_files(project, sample_filter)
    if not prediction_files:
        raiseCytoError(
            f"No '*_image_predictions[_top3].parquet' files found in '{project / 'work'}', "
            f"run 'cytoprocess predict_images {project}' first.",
            logger,
        )

    token = authenticate(logger, username=username, password=password)
    if token is None:
        raiseCytoError("Authentication failed, cannot proceed with EcoTaxa update", logger)

    project_info = _get_project_info(logger, token, int(project_id))
    project_name = project_info.get("title", "Unknown") if project_info else "Unknown"
    logger.info(f"Updating EcoTaxa project '{project_name}' [{project_id}]")

    project_samples = _get_project_samples(logger, token, int(project_id))
    if not project_samples:
        raiseCytoError("No samples could be retrieved from EcoTaxa for this project.", logger)

    logger.info(f"Found {len(prediction_files)} prediction file(s) to sync")

    total_updated = 0
    total_missing_samples = 0
    total_missing_objects = 0

    for prediction_file in prediction_files:
        sample_id = (
            prediction_file.stem
            .replace("_image_predictions_top3", "")
            .replace("_image_predictions", "")
        )
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
        object_map = _get_sample_object_map(logger, token, int(project_id), sample_ecotaxa_id)
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
        updated = _classify_objects(logger, token, target_ids, batch_classifications, batch_scores)
        total_updated += updated
        logger.info(f"  > Updated {updated} object(s)")

    logger.info(
        "Summary: "
        f"{total_updated} object(s) updated, "
        f"{total_missing_samples} sample(s) missing in EcoTaxa, "
        f"{total_missing_objects} object(s) not matched"
    )
    log_command_success(logger, "Overwrite EcoTaxa metadata")

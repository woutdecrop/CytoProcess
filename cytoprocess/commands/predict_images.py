from __future__ import annotations

import csv
import mimetypes
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import local

import click
import pandas as pd
import requests

from cytoprocess.logging import log_command_start, log_command_success, setup_logging
from cytoprocess.project import list_sample_assets, path_to_sample_asset
from cytoprocess.utils import raiseCytoError

PREDICT_URL_REMOTE = "http://127.0.0.1:5000/v2/models/planktonclas/predict/?ckpt_name=final_model.h5"
DOCKER_CONTAINER_NAME = "phyto_classifier_container_cyto"
DOCKER_IMAGE = "wdecrop/cyto-plankton-classifier"
HEALTH_URL = "http://127.0.0.1:5000/api"
CLASSIFIER_NAME = "cyto_classifier"
CLASSIFIER_EMAIL = "wout.decrop@vliz.be"

_THREAD_LOCAL = local()
_CATEGORY_MAPPING: dict[str, int] | None = None


def _run_command(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def _check_api_available() -> bool:
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _start_container(logger) -> None:
    status = _run_command(f'docker inspect -f "{{{{.State.Status}}}}" {DOCKER_CONTAINER_NAME}')

    if not status:
        logger.info("  Creating prediction container")
        subprocess.check_call(
            f"docker run -d -p 5000:5000 --name {DOCKER_CONTAINER_NAME} {DOCKER_IMAGE}",
            shell=True,
        )
        return

    if status == "running":
        logger.debug("Prediction container already running")
        return

    if status in ["exited", "created"]:
        logger.info("  Starting prediction container")
        subprocess.check_call(f"docker start {DOCKER_CONTAINER_NAME}", shell=True)
        return

    raise RuntimeError(f"Docker container '{DOCKER_CONTAINER_NAME}' is in unexpected state '{status}'")


def _wait_for_api(timeout_sec: int = 120) -> None:
    start_time = time.time()
    while time.time() - start_time <= timeout_sec:
        if _check_api_available():
            return
        time.sleep(2)
    raise RuntimeError(f"Prediction API did not become ready within {timeout_sec} seconds")


def _ensure_predictor_ready(logger) -> None:
    if _check_api_available():
        return
    _start_container(logger)
    _wait_for_api()


def _get_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.session = session
    return session


def _first_value(value):
    while isinstance(value, list) and value:
        value = value[0]
    return value


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            return _as_list(value[0])
        return value
    return [value]


def _mapping_file() -> Path:
    return Path(__file__).resolve().parents[2] / "ecotaxa-classes.tsv"


def _load_category_mapping(tsv_path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}

    with tsv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            label = row["object_annotation_category"].strip().strip('"')
            category_id = row["object_annotation_category_id"].strip().strip('"')
            if not label or not category_id or category_id == "[t]":
                continue
            try:
                mapping[label] = int(category_id)
            except ValueError:
                continue

    return mapping


def _get_category_mapping() -> dict[str, int]:
    global _CATEGORY_MAPPING
    if _CATEGORY_MAPPING is None:
        _CATEGORY_MAPPING = _load_category_mapping(_mapping_file())
    return _CATEGORY_MAPPING


def _map_prediction_label(label, category_mapping: dict[str, int]) -> tuple[str | None, int | None]:
    if isinstance(label, bytes):
        label = label.decode()

    if label is None:
        return None, None

    label = str(label)
    category_id = category_mapping.get(label)

    if category_id is None:
        alt_label = label.replace("_", "<")
        category_id = category_mapping.get(alt_label)
        if category_id is not None:
            label = alt_label

    if category_id is None:
        alt_label = label.replace("_", "<").replace(">", " ")
        category_id = category_mapping.get(alt_label)
        if category_id is not None:
            label = alt_label

    return label, category_id


def _extract_top_predictions(payload: dict, category_mapping: dict[str, int], top_n: int = 3) -> list[dict]:
    predictions = payload.get("predictions", payload)
    labels = []
    probabilities = []

    if isinstance(predictions, dict):
        for key in ("pred_lab", "label", "labels", "class_name", "class_names", "category", "categories"):
            if key in predictions:
                labels = _as_list(predictions[key])
                if labels:
                    break
        for key in ("pred_prob", "prob", "confidence", "score", "scores"):
            if key in predictions:
                probabilities = _as_list(predictions[key])
                break

    top_predictions = []
    for index, raw_label in enumerate(labels[:top_n]):
        label, category_id = _map_prediction_label(raw_label, category_mapping)
        raw_probability = probabilities[index] if index < len(probabilities) else None
        probability = None if raw_probability is None else float(_first_value(raw_probability))

        if label:
            top_predictions.append(
                {
                    "label": label,
                    "category_id": category_id,
                    "probability": probability,
                }
            )

    return top_predictions


def _predict_image(image_file: Path, sample_id: str, annotation_date: str, annotation_time: str) -> dict:
    session = _get_session()
    mime_type = mimetypes.guess_type(image_file.name)[0] or "application/octet-stream"

    with image_file.open("rb") as handle:
        response = session.post(
            PREDICT_URL_REMOTE,
            files={"image": (image_file.name, handle, mime_type)},
            timeout=60,
        )
    response.raise_for_status()

    payload = response.json()
    if payload.get("status") == "error":
        message = payload.get("message", "Unknown prediction API error")
        raise ValueError(f"Prediction API rejected '{image_file.name}': {message}")

    top_predictions = _extract_top_predictions(payload, _get_category_mapping(), top_n=3)
    if not top_predictions:
        raise ValueError(f"No prediction label returned for '{image_file.name}'")

    top_prediction = top_predictions[0]
    object_name = image_file.stem.replace("_img", "")
    object_id = f"{sample_id}_{object_name}"

    return {
        "sample_id": sample_id,
        "object_id": object_id,
        "object_annotation_date": annotation_date,
        "object_annotation_time": annotation_time,
        "object_annotation_category": top_prediction["label"],
        "object_annotation_category_id": top_prediction["category_id"],
        "object_annotation_categories": [prediction["label"] for prediction in top_predictions],
        "object_annotation_category_ids": [prediction["category_id"] for prediction in top_predictions],
        "object_annotation_probabilities": [prediction["probability"] for prediction in top_predictions],
        "object_annotation_person_name": CLASSIFIER_NAME,
        "object_annotation_person_email": CLASSIFIER_EMAIL,
        "object_annotation_status": "predicted",
        "object_annotation_probability": top_prediction["probability"],
    }


def run(ctx: click.Context, project: Path, force: bool = False):
    logger = setup_logging(command="predict_images", project=project, debug=ctx.obj["debug"])
    log_command_start(logger, "Predicting image classes", project)
    logger.debug("Context: %s", getattr(ctx, "obj", {}))
    if force:
        logger.debug("Force flag enabled, existing prediction files will be overwritten")

    sample_dirs = list_sample_assets(project, "dir", logger, samples_mask=ctx.obj["sample"])
    if not sample_dirs:
        return

    try:
        _ensure_predictor_ready(logger)
    except Exception as exc:
        raiseCytoError(f"Unable to start prediction API: {exc}", logger)

    max_workers = min(32, max(1, (os.cpu_count() or 1) * 4))
    logger.info(f"Processing {len(sample_dirs)} sample(s)")

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        images_dir = project / path_to_sample_asset(sample_id, "images", logger)
        output_file = project / path_to_sample_asset(sample_id, "predictions", logger)

        logger.info(f"'{sample_id}'")

        if output_file.exists() and not force:
            logger.info("  Skipping, predictions file already exists (use --force to overwrite)")
            continue

        if not images_dir.exists():
            logger.warning(f"  Images not found, run `cytoprocess --sample '{sample_id}' extract_images {project}`")
            continue

        image_files = sorted(images_dir.glob("*_img.jpg"))
        if not image_files:
            logger.warning(f"  No extracted images found in '{images_dir}', skipping")
            continue

        logger.info(f"  {len(image_files)} images to predict")
        now = datetime.now(timezone.utc)
        annotation_date = now.strftime("%Y-%m-%d")
        annotation_time = now.strftime("%H:%M:%S")

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    executor.map(
                        _predict_image,
                        image_files,
                        [sample_id] * len(image_files),
                        [annotation_date] * len(image_files),
                        [annotation_time] * len(image_files),
                    )
                )
        except Exception as exc:
            raiseCytoError(f"Error predicting sample '{sample_id}': {exc}", logger)

        df = pd.DataFrame(results).sort_values("object_id").reset_index(drop=True)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, index=False)

        logger.info(f"  Saved {df.shape[0]} predictions to\n  '{output_file}'")

    log_command_success(logger, "Predict images")

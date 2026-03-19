from __future__ import annotations

import os
import mimetypes
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import local

import pandas as pd
import requests

from cytoprocess.utils import ensure_project_dir, log_command_start, log_command_success, raiseCytoError, setup_logging
import csv

predict_url_remote = "http://127.0.0.1:5000/v2/models/planktonclas/predict/?ckpt_name=final_model.h5"
docker_container_name = "phyto_classifier_container_cyto"
docker_image = "wdecrop/cyto-plankton-classifier "
health_url = "http://127.0.0.1:5000/api"

_thread_local = local()


def _run_command(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def _check_api_available() -> bool:
    try:
        response = requests.get(health_url, timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _start_container(logger) -> None:
    status = _run_command(f'docker inspect -f "{{{{.State.Status}}}}" {docker_container_name}')

    if not status:
        logger.info("  Creating prediction container")
        subprocess.check_call(
            f"docker run -d -p 5000:5000 --name {docker_container_name} {docker_image}",
            shell=True,
        )
        return

    if status == "running":
        logger.debug("Prediction container already running")
        return

    # if status == "exited":
    #     logger.info("  Starting prediction container")
    #     subprocess.check_call(f"docker start {docker_container_name}", shell=True)
    #     return
    if status in ["exited", "created"]:
        logger.info("  Starting prediction container")
        subprocess.check_call(f"docker start {docker_container_name}", shell=True)
        return
    raise RuntimeError(f"Docker container '{docker_container_name}' is in unexpected state '{status}'")


def _wait_for_api(logger, timeout_sec: int = 120) -> None:
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
    _wait_for_api(logger)


def _get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
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


def _map_prediction_label(label, category_mapping: dict) -> tuple[str | None, int | None]:
    if isinstance(label, bytes):
        label = label.decode()

    if label is None:
        return None, None

    label = str(label)
    category_id = category_mapping.get(label)
    print(f"Trying direct mapping for predicted label '{label}'")

    if category_id is None:
        alt_label = label.replace("_", "<")
        category_id = category_mapping.get(alt_label)
        print(f"Trying alternative label '{alt_label}' for original label '{label}'")
        if category_id is not None:
            label = alt_label

    if category_id is None:
        alt_label = label.replace("_", "<").replace(">", " ")
        category_id = category_mapping.get(alt_label)
        print(f"Trying alternative label '{alt_label}' for original label '{label}'")
        if category_id is not None:
            label = alt_label

    print(f"Mapped predicted label '{label}' to category_id {category_id}")
    return label, category_id


def _extract_top_predictions(payload: dict, category_mapping: dict, top_n: int = 3) -> list[dict]:
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

def load_category_mapping(tsv_path: Path) -> dict[str, int]:
    mapping = {}

    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            label = row["object_annotation_category"].strip().strip('"')
            category_id = row["object_annotation_category_id"].strip().strip('"')

            # ✅ skip invalid rows like [t]
            if not label or not category_id or category_id == "[t]":
                continue

            try:
                mapping[label] = int(category_id)
            except ValueError:
                # optional: log or print bad rows
                print(f"Skipping invalid row: {row}")
                continue

    return mapping
category_mapping = load_category_mapping(Path("cytoprocess\\ecotaxa-classes.tsv"))

def _predict_image(image_file: Path, sample_id: str, annotation_date: str, annotation_time: str) -> dict:
    session = _get_session()
    mime_type = mimetypes.guess_type(image_file.name)[0] or "application/octet-stream"

    with image_file.open("rb") as fh:
        response = session.post(
            predict_url_remote,
            files={"image": (image_file.name, fh, mime_type)},
            timeout=60,
        )
    response.raise_for_status()

    payload = response.json()
    if payload.get("status") == "error":
        message = payload.get("message", "Unknown prediction API error")
        raise ValueError(f"Prediction API rejected '{image_file.name}': {message}")

    top_predictions = _extract_top_predictions(payload, category_mapping, top_n=3)
    if not top_predictions:
        raise ValueError(f"No prediction label returned for '{image_file.name}'")

    top_prediction = top_predictions[0]
    primary_row = {
        "sample_id": sample_id,
        "object_id": f"{sample_id}_{image_file.stem}",
        "object_annotation_date": annotation_date,
        "object_annotation_time": annotation_time,
        "object_annotation_category": top_prediction["label"],
        "object_annotation_category_id": top_prediction["category_id"],
        "object_annotation_person_name": "cyto_classifier",
        "object_annotation_person_email": "wout.decrop@vliz.be",
        "object_annotation_status": "predicted",
        "object_annotation_probability": top_prediction["probability"],
    }

    top3_row = {
        "sample_id": sample_id,
        "object_id": f"{sample_id}_{image_file.stem}",
        "object_annotation_date": annotation_date,
        "object_annotation_time": annotation_time,
        "object_annotation_categories": [prediction["label"] for prediction in top_predictions],
        "object_annotation_category_ids": [prediction["category_id"] for prediction in top_predictions],
        "object_annotation_probabilities": [prediction["probability"] for prediction in top_predictions],
        "object_annotation_person_name": "cyto_classifier",
        "object_annotation_person_email": "wout.decrop@vliz.be",
        "object_annotation_status": "predicted",
    }

    return {"primary": primary_row, "top3": top3_row}


def run(ctx, project, force: bool = False):
    logger = setup_logging(command="predict_images", project=project, debug=ctx.obj["debug"])

    log_command_start(logger, "Predicting image classes", project)
    logger.debug("Context: %s", getattr(ctx, "obj", {}))

    project = Path(project)
    images_dir = project / "images"
    if not images_dir.exists():
        raiseCytoError(f"Images directory not found: '{images_dir}'. Run extract_images first.", logger)

    sample_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
    if not sample_dirs:
        raiseCytoError(f"No sample directories found in '{images_dir}', run 'cytoprocess extract_images {project}' first.", logger)

    sample = getattr(ctx, "obj", {}).get("sample")
    if sample:
        sample_dirs = [d for d in sample_dirs if d.name == sample]
        if not sample_dirs:
            raiseCytoError(f"No image directory found for sample '{sample}', run 'cytoprocess --sample \"{sample}\" extract_images {project}' first.", logger)

    work_dir = ensure_project_dir(project, "work")

    try:
        _ensure_predictor_ready(logger)
    except Exception as exc:
        raiseCytoError(f"Unable to start prediction API: {exc}", logger)

    max_workers = min(32, max(1, (os.cpu_count() or 1) * 4))
    logger.info(f"Processing {len(sample_dirs)} sample(s)")

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        output_file = work_dir / f"{sample_id}_image_predictions.parquet"
        top3_output_file = work_dir / f"{sample_id}_image_predictions_top3.parquet"

        logger.info(f"'{sample_id}'")

        if output_file.exists() and top3_output_file.exists() and not force:
            logger.info("  Skipping, output file already exists (use --force to overwrite)")
            continue

        image_files = sorted(sample_dir.glob("*.jpg"))
        if not image_files:
            logger.warning(f"No JPG images found in '{sample_dir}', run 'cytoprocess --sample \"{sample_id}\" extract_images {project}' first.")
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

        df = pd.DataFrame(result["primary"] for result in results).sort_values("object_id").reset_index(drop=True)
        df_top3 = pd.DataFrame(result["top3"] for result in results).sort_values("object_id").reset_index(drop=True)

        df.to_parquet(output_file, index=False)
        df_top3.to_parquet(top3_output_file, index=False)

        logger.info(f"  Saved {df.shape[0]} top-1 predictions to\n  '{output_file}'")
        logger.info(f"  Saved {df_top3.shape[0]} top-3 predictions to\n  '{top3_output_file}'")

    log_command_success(logger, "Predict images")

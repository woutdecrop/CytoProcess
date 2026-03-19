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
docker_container_name = "phyto_classifier_container_v1"
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



def _extract_prediction(payload: dict, category_mapping: dict) -> tuple[str | None, int | None]:
    predictions = payload.get("predictions", payload)
    label = None

    if isinstance(predictions, dict):
        for key in ("pred_lab", "label", "labels", "class_name", "class_names", "category", "categories"):
            if key in predictions:
                label = _first_value(predictions[key])
                if label is not None:
                    break

    if isinstance(label, bytes):
        label = label.decode()

    if label is not None:
        label = str(label)

        # 1️⃣ direct mapping
        category_id = category_mapping.get(label)
        print(f"Trying direct mapping for predicted label '{label}'")
        # 2️⃣ fallback: replace '_' with '<'
        if category_id is None:
            alt_label = label.replace("_", "<")
            category_id = category_mapping.get(alt_label)
            print(f"Trying alternative label '{alt_label}' for original label '{label}'")
            if category_id is not None:
                label = alt_label  # update label to match the mapping key if this one works
        # 3️⃣ fallback: replace '>' with spaces
        if category_id is None:
            alt_label = label.replace("_", "<").replace(">", " ")
            category_id = category_mapping.get(alt_label)
            print(f"Trying alternative label '{alt_label}' for original label '{label}'")
            if category_id is not None:
                label = alt_label  # update label to match the mapping key if this one works

        print(f"Mapped predicted label '{label}' to category_id {category_id}")
    else:
        category_id = None

    return label, category_id

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

    label, category_id = _extract_prediction(payload, category_mapping)
    if not label:
        raise ValueError(f"No prediction label returned for '{image_file.name}'")

    return {
        "sample_id": sample_id,
        "object_id": f"{sample_id}_{image_file.stem}",
        "object_annotation_date": annotation_date,
        "object_annotation_time": annotation_time,
        "object_annotation_category": label,
        "object_annotation_category_id": category_id,
        "object_annotation_person_name": "cyto_classifier",
        "object_annotation_person_email": "wout.decrop@vliz.be",
        "object_annotation_status": "predicted",
    }


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

        logger.info(f"'{sample_id}'")

        if output_file.exists() and not force:
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
                rows = list(
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

        df = pd.DataFrame(rows).sort_values("object_id").reset_index(drop=True)
        df.to_parquet(output_file, index=False)
        logger.info(f"  Saved {df.shape[0]} predictions to\n  '{output_file}'")

    log_command_success(logger, "Predict images")

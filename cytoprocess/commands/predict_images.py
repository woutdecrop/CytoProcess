from __future__ import annotations

import csv
import json
import mimetypes
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import local

import click
import pandas as pd
import requests
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from cytoprocess.logging import log_command_start, log_command_success, setup_logging
from cytoprocess.project import list_sample_assets, path_to_sample_asset
from cytoprocess.utils import raiseCytoError

PREDICT_URL_REMOTE = "http://127.0.0.1:5000/v2/models/planktonclas/predict/"
MODEL_INFO_URL = "http://127.0.0.1:5000/v2/models/planktonclas/"
SWAGGER_URL = "http://127.0.0.1:5000/swagger.json"
DOCKER_CONTAINER_NAME = "phyto_classifier_container_flowcyto_obsea"
DOCKER_IMAGE = "wdecrop/cyto-plankton-classifier:flowcyto-obsea"
HEALTH_URL = "http://127.0.0.1:5000/api"
CLASSIFIER_NAME = "flowcyto_obsea_classifier"
CLASSIFIER_EMAIL = "wout.decrop@vliz.be"
DEFAULT_CKPT_NAME = "final_model.keras"
DEFAULT_LOCAL_MODEL_ROOT = "FlowCytoClassifier"
DEFAULT_PROJECT_MODEL_ROOT = "models"
LOCAL_MODEL_ROOT_ENV = "CYTOPROCESS_LOCAL_MODEL_ROOT"
PREDICT_BACKEND_ENV = "CYTOPROCESS_PREDICT_BACKEND"
LOCAL_PREDICT_CHUNK_SIZE = 256
PREDICT_CONNECT_TIMEOUT_SEC = 10
PREDICT_READ_TIMEOUT_SEC = 240
DEFAULT_PREDICT_MAX_WORKERS = 4
PREDICT_MAX_RETRIES = 3
PREDICT_RETRY_BACKOFF_SEC = 5


_THREAD_LOCAL = local()
_CATEGORY_MAPPING: dict[str, int] | None = None
_LOCAL_PREDICTOR: dict | None = None


def _run_command(cmd: str) -> tuple[int, str, str]:
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _ensure_docker_available() -> None:
    code, _, stderr = _run_command("docker version --format '{{.Server.Version}}'")
    if code == 0:
        return

    detail = stderr or "Docker engine is not reachable."
    raise RuntimeError(
        "Docker is required for image prediction but is not available. "
        f"Please start Docker Desktop or the Docker engine and retry. Details: {detail}"
    )


def _check_api_available() -> bool:
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _start_container(logger) -> None:
    _ensure_docker_available()
    code, status, stderr = _run_command(f'docker inspect -f "{{{{.State.Status}}}}" {DOCKER_CONTAINER_NAME}')

    if code != 0 and "No such object" not in stderr:
        raise RuntimeError(f"Unable to inspect Docker container '{DOCKER_CONTAINER_NAME}': {stderr or 'unknown docker error'}")

    if code != 0 or not status:
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


def _reset_session() -> None:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is not None:
        session.close()
        _THREAD_LOCAL.session = None


def _prediction_params(ckpt_name: str | None = None, quoted: bool = False) -> dict[str, str]:
    ckpt_name = ckpt_name or DEFAULT_CKPT_NAME
    ckpt_name = f'"{ckpt_name}"' if quoted else ckpt_name
    return {"ckpt_name": ckpt_name}


def _clean_api_string(value):
    if isinstance(value, str):
        return value.strip().strip('"')
    return value


def _log_predictor_details(logger) -> None:
    logger.info(f"Prediction container: {DOCKER_CONTAINER_NAME}")
    logger.info(f"Prediction image: {DOCKER_IMAGE}")
    logger.info(f"Prediction endpoint: {PREDICT_URL_REMOTE}")

    try:
        model_response = requests.get(MODEL_INFO_URL, timeout=5)
        model_response.raise_for_status()
        metadata = model_response.json()

        model_name = _clean_api_string(metadata.get("name") or metadata.get("id"))
        model_version = _clean_api_string(metadata.get("version"))
        authors = metadata.get("author") or []
        if not isinstance(authors, list):
            authors = [authors]
        authors = [_clean_api_string(author) for author in authors if author]

        # if model_name:
        #     suffix = f" v{model_version}" if model_version else ""
        #     logger.info(f"Model: {model_name}{suffix}")
        if authors:
            logger.info(f"Model author(s): {', '.join(authors)}")
    except requests.RequestException as exc:
        logger.debug("Unable to fetch model metadata: %s", exc)

    try:
        swagger_response = requests.get(SWAGGER_URL, timeout=5)
        swagger_response.raise_for_status()
        swagger = swagger_response.json()
        predict_post = swagger["paths"]["/v2/models/planktonclas/predict/"]["post"]
        parameters = predict_post.get("parameters", [])

        timestamp_values = []
        ckpt_values = []
        for parameter in parameters:
            if parameter.get("name") == "timestamp":
                timestamp_values = [_clean_api_string(value) for value in parameter.get("enum", [])]
            if parameter.get("name") == "ckpt_name":
                ckpt_values = [_clean_api_string(value) for value in parameter.get("enum", [])]

        if timestamp_values:
            logger.info(f"Available model timestamp(s): {', '.join(timestamp_values)}")
        if ckpt_values:
            logger.info(f"Available checkpoint(s): {', '.join(ckpt_values)}")
    except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
        logger.debug("Unable to fetch prediction schema metadata: %s", exc)


def _should_retry_with_quoted_params(response: requests.Response, ckpt_name: str | None = None) -> bool:
    if response.status_code != 422:
        return False

    try:
        payload = response.json()
    except ValueError:
        return False

    ckpt_errors = payload.get("ckpt_name")
    if not isinstance(ckpt_errors, list):
        return False

    requested_ckpt = ckpt_name or DEFAULT_CKPT_NAME
    return any(f'"{requested_ckpt}"' in str(error) for error in ckpt_errors)


def _first_value(value):
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    while isinstance(value, list) and value:
        value = value[0]
    return value


def _as_list(value):
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if value is None:
        return []
    if isinstance(value, tuple):
        value = list(value)
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


def _build_prediction_row(
    image_file: Path,
    sample_id: str,
    annotation_date: str,
    annotation_time: str,
    top_predictions: list[dict],
) -> dict:
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


def _predict_image(
    image_file: Path,
    sample_id: str,
    annotation_date: str,
    annotation_time: str,
    ckpt_name: str | None = None,
    logger=None,
) -> dict:
    mime_type = mimetypes.guess_type(image_file.name)[0] or "application/octet-stream"

    last_error = None
    for attempt in range(1, PREDICT_MAX_RETRIES + 1):
        session = _get_session()
        try:
            with image_file.open("rb") as handle:
                response = session.post(
                    PREDICT_URL_REMOTE,
                    params=_prediction_params(ckpt_name=ckpt_name),
                    files={"image": (image_file.name, handle, mime_type)},
                    timeout=(PREDICT_CONNECT_TIMEOUT_SEC, PREDICT_READ_TIMEOUT_SEC),
                )

                if _should_retry_with_quoted_params(response, ckpt_name=ckpt_name):
                    handle.seek(0)
                    response = session.post(
                        PREDICT_URL_REMOTE,
                        params=_prediction_params(ckpt_name=ckpt_name, quoted=True),
                        files={"image": (image_file.name, handle, mime_type)},
                        timeout=(PREDICT_CONNECT_TIMEOUT_SEC, PREDICT_READ_TIMEOUT_SEC),
                    )
            break
        except requests.exceptions.ReadTimeout as exc:
            last_error = exc
            _reset_session()
            if attempt == PREDICT_MAX_RETRIES:
                raise
            if logger is not None:
                logger.warning(
                    f"  Timeout predicting '{image_file.name}' "
                    f"(attempt {attempt}/{PREDICT_MAX_RETRIES}), "
                    f"retrying in {PREDICT_RETRY_BACKOFF_SEC * attempt}s"
                )
            time.sleep(PREDICT_RETRY_BACKOFF_SEC * attempt)
        except requests.RequestException:
            _reset_session()
            raise
    else:
        raise last_error if last_error is not None else RuntimeError(f"Prediction failed for '{image_file.name}'")
    response.raise_for_status()

    payload = response.json()
    if payload.get("status") == "error":
        message = payload.get("message", "Unknown prediction API error")
        raise ValueError(f"Prediction API rejected '{image_file.name}': {message}")

    top_predictions = _extract_top_predictions(payload, _get_category_mapping(), top_n=3)
    if not top_predictions:
        raise ValueError(f"No prediction label returned for '{image_file.name}'")

    return _build_prediction_row(
        image_file,
        sample_id,
        annotation_date,
        annotation_time,
        top_predictions,
    )


def _write_prediction_failures(failure_file: Path, failures: list[dict]) -> None:
    failure_file.parent.mkdir(parents=True, exist_ok=True)
    with failure_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_file", "error"])
        writer.writeheader()
        writer.writerows(failures)


def _make_progress(total: int, sample_id: str):
    if tqdm is None:
        return None
    return tqdm(
        total=total,
        desc=f"{sample_id}",
        unit="img",
        leave=False,
    )


def _resolve_local_model_root(local_model_root: str | os.PathLike[str] | None) -> Path:
    candidate = Path(local_model_root or DEFAULT_LOCAL_MODEL_ROOT).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _candidate_local_model_roots(project: Path) -> list[Path]:
    candidates = []
    env_root = os.environ.get(LOCAL_MODEL_ROOT_ENV)
    if env_root:
        candidates.append(Path(env_root).expanduser())

    try:
        from cytoprocess.commands.install import get_predict_model_install_dir

        candidates.append(get_predict_model_install_dir())
    except Exception:
        pass

    candidates.extend(
        [
            project / DEFAULT_PROJECT_MODEL_ROOT,
            Path.cwd() / DEFAULT_PROJECT_MODEL_ROOT,
            project.parent / DEFAULT_PROJECT_MODEL_ROOT,
            Path.cwd() / DEFAULT_LOCAL_MODEL_ROOT,
            project.parent / DEFAULT_LOCAL_MODEL_ROOT,
            project / DEFAULT_LOCAL_MODEL_ROOT,
        ]
    )

    seen = set()
    resolved_candidates = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        resolved_candidates.append(resolved)
    return resolved_candidates


def _discover_local_model_root(project: Path, local_model_root: str | os.PathLike[str] | None) -> Path | None:
    if local_model_root is not None:
        resolved = _resolve_local_model_root(local_model_root)
        return resolved if (resolved / "models").exists() else None

    for candidate in _candidate_local_model_roots(project):
        if (candidate / "models").exists():
            return candidate
    return None


def _available_model_timestamps(models_dir: Path) -> list[str]:
    if not models_dir.exists():
        return []
    return sorted(path.name for path in models_dir.iterdir() if path.is_dir())


def _resolve_local_timestamp(models_dir: Path, local_timestamp: str | None) -> str:
    timestamps = _available_model_timestamps(models_dir)
    if not timestamps:
        raise FileNotFoundError(f"No trained model directories found in '{models_dir}'")

    if local_timestamp:
        if local_timestamp not in timestamps:
            raise FileNotFoundError(
                f"Model timestamp '{local_timestamp}' not found in '{models_dir}'. "
                f"Available timestamps: {', '.join(timestamps)}"
            )
        return local_timestamp

    return timestamps[-1]


def _load_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_local_checkpoint(timestamp_dir: Path, conf: dict, ckpt_name: str | None) -> tuple[str, Path]:
    checkpoints_dir = timestamp_dir / "ckpts"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: '{checkpoints_dir}'")

    candidates = []
    if ckpt_name:
        candidates.append(ckpt_name)

    configured_ckpt = conf.get("testing", {}).get("ckpt_name")
    if configured_ckpt:
        candidates.append(configured_ckpt)

    candidates.extend(["best_model.keras", "final_model.keras", "final_model.h5"])

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        checkpoint_path = checkpoints_dir / candidate
        if checkpoint_path.exists():
            return candidate, checkpoint_path

    available = sorted(path.name for path in checkpoints_dir.iterdir() if path.is_file())
    raise FileNotFoundError(
        f"No supported checkpoint found in '{checkpoints_dir}'. "
        f"Available files: {', '.join(available)}"
    )


def _get_local_predictor(
    local_model_root: str | os.PathLike[str] | None,
    local_timestamp: str | None,
    ckpt_name: str | None,
    logger,
) -> dict:
    global _LOCAL_PREDICTOR

    model_root = _resolve_local_model_root(local_model_root)
    models_dir = model_root / "models"
    timestamp = _resolve_local_timestamp(models_dir, local_timestamp)
    timestamp_dir = models_dir / timestamp

    conf_path = timestamp_dir / "conf" / "conf.json"
    if not conf_path.exists():
        raise FileNotFoundError(f"Model configuration not found: '{conf_path}'")

    conf = _load_json_file(conf_path)
    resolved_ckpt_name, checkpoint_path = _resolve_local_checkpoint(timestamp_dir, conf, ckpt_name)

    cache_key = (str(model_root), timestamp, resolved_ckpt_name)
    if _LOCAL_PREDICTOR is not None and _LOCAL_PREDICTOR.get("cache_key") == cache_key:
        return _LOCAL_PREDICTOR

    try:
        from planktonclas.data_utils import load_class_names
        from planktonclas.test_utils import predict as plankton_predict
        from planktonclas.utils import get_custom_objects
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise RuntimeError(
            "Local prediction requires the 'planktonclas' and TensorFlow packages to be installed."
        ) from exc

    dataset_files_dir = timestamp_dir / "dataset_files"
    class_names = load_class_names(str(dataset_files_dir))
    model = load_model(str(checkpoint_path), custom_objects=get_custom_objects())

    logger.info(f"Local model root: {model_root}")
    logger.info(f"Local model timestamp: {timestamp}")
    logger.info(f"Local checkpoint: {resolved_ckpt_name}")

    _LOCAL_PREDICTOR = {
        "cache_key": cache_key,
        "conf": conf,
        "class_names": list(class_names),
        "model": model,
        "predict_fn": plankton_predict,
    }
    return _LOCAL_PREDICTOR


def _top_predictions_from_local_output(
    labels_row,
    probabilities_row,
    class_names: list[str],
    category_mapping: dict[str, int],
    top_n: int = 3,
) -> list[dict]:
    top_predictions = []

    for raw_label_index, raw_probability in zip(_as_list(labels_row)[:top_n], _as_list(probabilities_row)[:top_n]):
        label_index = int(raw_label_index)
        raw_label = class_names[label_index]
        label, category_id = _map_prediction_label(raw_label, category_mapping)
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


def _predict_images_local_chunk(
    image_files: list[Path],
    sample_id: str,
    annotation_date: str,
    annotation_time: str,
    predictor: dict,
) -> list[dict]:
    image_paths = [str(image_file.resolve()) for image_file in image_files]
    labels, probabilities = predictor["predict_fn"](
        predictor["model"],
        image_paths,
        predictor["conf"],
        top_K=3,
        filemode="local",
        merge=False,
    )

    category_mapping = _get_category_mapping()
    results = []
    for image_file, labels_row, probabilities_row in zip(image_files, labels, probabilities):
        top_predictions = _top_predictions_from_local_output(
            labels_row,
            probabilities_row,
            predictor["class_names"],
            category_mapping,
            top_n=3,
        )
        if not top_predictions:
            raise ValueError(f"No prediction label returned for '{image_file.name}'")
        results.append(
            _build_prediction_row(
                image_file,
                sample_id,
                annotation_date,
                annotation_time,
                top_predictions,
            )
        )

    return results


def _predict_images_local(
    image_files: list[Path],
    sample_id: str,
    annotation_date: str,
    annotation_time: str,
    predictor: dict,
    logger,
    progress=None,
) -> tuple[list[dict], list[dict]]:
    results = []
    failures = []

    for start in range(0, len(image_files), LOCAL_PREDICT_CHUNK_SIZE):
        batch = image_files[start : start + LOCAL_PREDICT_CHUNK_SIZE]
        try:
            results.extend(
                _predict_images_local_chunk(
                    batch,
                    sample_id,
                    annotation_date,
                    annotation_time,
                    predictor,
                )
            )
            if progress is not None:
                progress.update(len(batch))
        except Exception as exc:
            logger.warning(
                f"  Batch prediction failed for {len(batch)} image(s); "
                f"falling back to per-image local prediction. Details: {exc}"
            )
            for image_file in batch:
                try:
                    results.extend(
                        _predict_images_local_chunk(
                            [image_file],
                            sample_id,
                            annotation_date,
                            annotation_time,
                            predictor,
                        )
                    )
                except Exception as image_exc:
                    failures.append({"image_file": image_file.name, "error": str(image_exc)})
                    logger.error(f"  Failed to predict '{image_file.name}': {image_exc}")
                finally:
                    if progress is not None:
                        progress.update(1)

    return results, failures


def run(
    ctx: click.Context,
    project: Path,
    force: bool = False,
    max_workers: int = DEFAULT_PREDICT_MAX_WORKERS,
    backend: str = "local",
    local_model_root: str | os.PathLike[str] | None = None,
    local_timestamp: str | None = None,
    ckpt_name: str | None = None,
    zenodo_version: str | None = None,
):
    logger = setup_logging(command="predict_images", project=project, debug=ctx.obj["debug"])
    log_command_start(logger, "Predicting image classes", project)
    logger.debug("Context: %s", getattr(ctx, "obj", {}))
    requested_backend = (os.environ.get(PREDICT_BACKEND_ENV) or backend or "local").lower()
    if force:
        logger.debug("Force flag enabled, existing prediction files will be overwritten")

    sample_dirs = list_sample_assets(project, "dir", logger, samples_mask=ctx.obj["sample"])
    if not sample_dirs:
        return

    discovered_local_model_root = _discover_local_model_root(project, local_model_root)
    backend = requested_backend

    logger.info(f"Prediction backend: {backend}")

    predictor = None
    if backend == "docker":
        try:
            _ensure_predictor_ready(logger)
        except Exception as exc:
            raiseCytoError(f"Unable to start prediction API: {exc}", logger)

        _log_predictor_details(logger)
        if ckpt_name:
            logger.info(f"Requested Docker checkpoint: {ckpt_name}")
    elif backend == "local":
        try:
            if local_model_root is None and discovered_local_model_root is not None:
                predictor = _get_local_predictor(
                    discovered_local_model_root,
                    local_timestamp,
                    ckpt_name,
                    logger,
                )
            else:
                if local_model_root is None:
                    from cytoprocess.commands.install import ensure_predict_model_available

                    managed_local_model_root = ensure_predict_model_available(
                        logger=logger,
                        force=False,
                        zenodo_version=zenodo_version,
                        project=project,
                    )
                    predictor = _get_local_predictor(
                        managed_local_model_root,
                        local_timestamp,
                        ckpt_name,
                        logger,
                    )
                else:
                    if discovered_local_model_root is None:
                        raise FileNotFoundError(
                            "No local model root was found. "
                            "Set --local-model-root or define CYTOPROCESS_LOCAL_MODEL_ROOT."
                        )
                    predictor = _get_local_predictor(discovered_local_model_root, local_timestamp, ckpt_name, logger)
        except Exception as exc:
            raiseCytoError(f"Unable to load local prediction model: {exc}", logger)
    else:
        raiseCytoError(f"Unsupported prediction backend '{backend}'", logger)

    max_workers = max(1, max_workers)
    logger.info(f"Processing {len(sample_dirs)} sample(s)")

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        images_dir = project / path_to_sample_asset(sample_id, "images", logger)
        output_file = project / path_to_sample_asset(sample_id, "predictions", logger)
        failure_file = output_file.with_name("prediction_failures.csv")

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

        progress = _make_progress(len(image_files), sample_id)

        try:
            if backend == "local":
                logger.info(f"  Predicting locally in batches of up to {LOCAL_PREDICT_CHUNK_SIZE} image(s)")
                results, failures = _predict_images_local(
                    image_files,
                    sample_id,
                    annotation_date,
                    annotation_time,
                    predictor,
                    logger,
                    progress=progress,
                )
            else:
                sample_workers = min(max_workers, len(image_files))
                logger.info(f"  Using up to {sample_workers} parallel request(s)")

                results = []
                failures = []
                with ThreadPoolExecutor(max_workers=sample_workers) as executor:
                    future_to_image = {
                        executor.submit(
                            _predict_image,
                            image_file,
                            sample_id,
                            annotation_date,
                            annotation_time,
                            ckpt_name,
                            logger,
                        ): image_file
                        for image_file in image_files
                    }

                    for future in as_completed(future_to_image):
                        image_file = future_to_image[future]
                        try:
                            results.append(future.result())
                        except Exception as exc:
                            failures.append({"image_file": image_file.name, "error": str(exc)})
                            logger.error(f"  Failed to predict '{image_file.name}': {exc}")
                        finally:
                            if progress is not None:
                                progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        if results:
            df = pd.DataFrame(results).sort_values("object_id").reset_index(drop=True)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_file, index=False)
            logger.info(f"  Saved {df.shape[0]} predictions to\n  '{output_file}'")
        else:
            logger.warning("  No successful predictions for this sample")

        if failures:
            _write_prediction_failures(failure_file, failures)
            logger.warning(f"  {len(failures)} image(s) failed; details saved to\n  '{failure_file}'")
        elif failure_file.exists():
            failure_file.unlink()

    log_command_success(logger, "Predict images")

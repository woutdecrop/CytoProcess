from __future__ import annotations

import shutil
import subprocess
import zipfile
from datetime import datetime
import os
import re
from pathlib import Path

import click
import pandas as pd
import requests
import yaml

from cytoprocess import ecotaxa
from cytoprocess.logging import log_command_start, log_command_success, setup_logging

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*]')
EXPORT_GLOB = "ecotaxa_export*.tsv"
EXPORT_PREFIX = "ecotaxa_export"
DATA_DIRNAME = "data"
TRAINING_IMAGES_DIRNAME = "images_validated"
TRAINING_MANIFEST_FILENAME = "validated_images.tsv"
TRAINING_ROOT_DIRNAME = "train"
PLANKTONCLASS_CONFIG_FILENAME = "config.yaml"


def _raise_train_error(message: str, logger=None):
    if logger is not None:
        logger.debug(message)
    raise click.ClickException(click.style(message, fg="red"))


def _clean_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().strip('"')


def _safe_path_part(value: object, fallback: str) -> str:
    cleaned = INVALID_PATH_CHARS.sub("_", _clean_value(value)).strip()
    return cleaned or fallback


def _check_training_project_inputs(project: Path, logger) -> None:
    required_dirs = ["work", "config", "meta"]
    for dirname in required_dirs:
        if not (project / dirname).exists():
            _raise_train_error(f"Expected directory '{dirname}' is missing in '{project}'", logger)


def _discover_export_tsv(project: Path) -> Path | None:
    data_dir = project / DATA_DIRNAME
    candidates = sorted(data_dir.glob(EXPORT_GLOB), key=lambda path: path.stat().st_mtime)
    if candidates:
        return candidates[-1]

    legacy_roots = [project, project.parent, Path.cwd()]
    seen: set[str] = set()
    for root in legacy_roots:
        resolved_root = root.resolve()
        key = str(resolved_root)
        if key in seen or not resolved_root.exists():
            continue
        seen.add(key)
        candidates.extend(resolved_root.glob(EXPORT_GLOB))

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_project_metadata_config(project: Path, logger) -> dict:
    config_path = project / "config" / "config.yaml"
    if not config_path.exists():
        _raise_train_error(
            f"Config file not found: '{config_path}', run 'cytoprocess create {project}' again.",
            logger,
        )
    try:
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        _raise_train_error(f"Failed to read project config '{config_path}': {exc}", logger)


def _extract_project_id(project_config: dict, config_path: Path, logger) -> int:
    ecotaxa_config = project_config.get("ecotaxa", {}) or {}
    project_id = ecotaxa_config.get("project_id")
    if not project_id:
        _raise_train_error(
            f"EcoTaxa project_id missing from '{config_path}'.",
            logger,
        )
    try:
        return int(project_id)
    except Exception as exc:
        _raise_train_error(f"Invalid EcoTaxa project_id '{project_id}': {exc}", logger)


def _get_api_url(project_config: dict) -> str:
    ecotaxa_config = project_config.get("ecotaxa", {}) or {}
    eco_url = ecotaxa_config.get("url", "https://ecotaxa.obs-vlfr.fr")
    return f"{str(eco_url).rstrip('/')}/api"


def _download_export_archive(
    api_url: str,
    token: str,
    project_id: int,
    export_archive_path: Path,
    logger,
) -> Path:
    request_payload = {
        "filters": {},
        "request": {
            "project_id": project_id,
            "exp_type": "TSV",
            "tsv_entities": "O",
            "split_by": "",
            "with_types_row": False,
            "with_internal_ids": False,
            "format_dates_times": True,
            "coma_as_separator": False,
        },
    }

    logger.info(f"Requesting a fresh EcoTaxa TSV export for project {project_id}")
    try:
        response = requests.post(
            f"{api_url}/object_set/export",
            headers={"Authorization": f"Bearer {token}"},
            json=request_payload,
            timeout=120,
        )
    except requests.RequestException as exc:
        _raise_train_error(f"Failed to start EcoTaxa export: {exc}", logger)

    if response.status_code != 200:
        _raise_train_error(f"Failed to start EcoTaxa export: {response.text}", logger)

    response_payload = response.json() or {}
    job_id = response_payload.get("job_id")
    if not job_id:
        _raise_train_error(f"EcoTaxa export did not return a job id: {response_payload}", logger)

    if not ecotaxa.monitor_job(api_url, int(job_id), token, logger=logger):
        _raise_train_error(f"EcoTaxa export job {job_id} did not complete successfully", logger)

    logger.info(f"Downloading EcoTaxa export job file {job_id}")
    try:
        file_response = requests.get(
            f"{api_url}/jobs/{int(job_id)}/file",
            headers={"Authorization": f"Bearer {token}"},
            timeout=300,
        )
    except requests.RequestException as exc:
        _raise_train_error(f"Failed to download EcoTaxa export file: {exc}", logger)

    if file_response.status_code != 200:
        _raise_train_error(f"Failed to download EcoTaxa export file: {file_response.text}", logger)

    export_archive_path.parent.mkdir(parents=True, exist_ok=True)
    export_archive_path.write_bytes(file_response.content)
    logger.info(f"Downloaded EcoTaxa export archive: '{export_archive_path}'")
    return export_archive_path


def _extract_export_tsv(export_archive_path: Path, tsv_path: Path, logger) -> Path:
    if zipfile.is_zipfile(export_archive_path):
        with zipfile.ZipFile(export_archive_path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".tsv")]
            if not members:
                _raise_train_error(
                    f"EcoTaxa export archive '{export_archive_path}' does not contain a TSV file.",
                    logger,
                )
            with archive.open(members[0]) as source, open(tsv_path, "wb") as destination:
                shutil.copyfileobj(source, destination)
    else:
        shutil.copyfile(export_archive_path, tsv_path)

    logger.info(f"Saved EcoTaxa export TSV: '{tsv_path}'")
    return tsv_path


def _download_fresh_export(project: Path, logger) -> Path:
    project_config = _load_project_metadata_config(project, logger)
    config_path = project / "config" / "config.yaml"
    project_id = _extract_project_id(project_config, config_path, logger)
    api_url = _get_api_url(project_config)
    token = ecotaxa.authenticate(api_url, logger=logger)
    if token is None:
        _raise_train_error("Authentication failed, cannot download EcoTaxa export", logger)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    data_dir = project / DATA_DIRNAME
    export_archive_path = data_dir / f"{EXPORT_PREFIX}_{project_id}_{timestamp}.zip"
    export_tsv_path = data_dir / f"{EXPORT_PREFIX}_{project_id}_{timestamp}.tsv"

    _download_export_archive(
        api_url=api_url,
        token=token,
        project_id=project_id,
        export_archive_path=export_archive_path,
        logger=logger,
    )
    return _extract_export_tsv(export_archive_path, export_tsv_path, logger)


def _resolve_export_tsv(project: Path, explicit_export_tsv: Path | None, logger) -> Path:
    if explicit_export_tsv is not None:
        explicit_export_tsv = explicit_export_tsv.expanduser().resolve()
        if not explicit_export_tsv.exists():
            _raise_train_error(f"EcoTaxa export not found: '{explicit_export_tsv}'", logger)
        logger.info(f"Using explicit EcoTaxa export: '{explicit_export_tsv}'")
        return explicit_export_tsv

    existing_export = _discover_export_tsv(project)
    if existing_export is None:
        return _download_fresh_export(project, logger)

    use_existing = click.confirm(
        f"Reuse the existing EcoTaxa export '{existing_export.name}'?",
        default=True,
        show_default=True,
    )
    if use_existing:
        logger.info(f"Using existing EcoTaxa export: '{existing_export}'")
        return existing_export

    return _download_fresh_export(project, logger)


def _load_validated_export(export_tsv: Path, logger) -> pd.DataFrame:
    try:
        df = pd.read_csv(export_tsv, sep="\t", quotechar='"', dtype=str, low_memory=False)
    except Exception as exc:
        _raise_train_error(f"Failed to read EcoTaxa export '{export_tsv}': {exc}", logger)

    required_columns = {"object_id", "object_annotation_status", "object_annotation_category", "object_annotation_person_name"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        _raise_train_error(
            f"EcoTaxa export '{export_tsv}' is missing required column(s): {', '.join(missing_columns)}",
            logger,
        )

    df = df[df["object_id"] != "[t]"].copy()
    df["object_id_clean"] = df["object_id"].map(_clean_value)
    df["annotation_status_clean"] = df["object_annotation_status"].map(_clean_value).str.lower()
    df["category_clean"] = df["object_annotation_category"].map(
        lambda value: _safe_path_part(value, "unclassified")
    )
    df = df[df["annotation_status_clean"] == "validated"].copy()
    df = df[df["object_id_clean"] != ""].copy()
    df = df[df["object_annotation_person_name"].map(_clean_value) == "Luz Amadei Matinez"].copy()
    df = df[df["object_id_clean"] != ""].copy()

    if df.empty:
        _raise_train_error(f"No validated objects were found in '{export_tsv}'", logger)

    return df


def _build_project_image_index(project: Path) -> dict[str, dict[str, Path]]:
    source_index: dict[str, dict[str, Path]] = {}

    for sample_dir in sorted((project / "work").glob("*")):
        if not sample_dir.is_dir():
            continue

        images_dir = sample_dir / "images"
        if not images_dir.exists():
            continue

        files_by_object_id: dict[str, Path] = {}
        for file_path in sorted(images_dir.iterdir()):
            if not file_path.is_file() or file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            stem = file_path.stem
            if stem.endswith("_img"):
                stem = stem[:-4]
            files_by_object_id[stem] = file_path

        if files_by_object_id:
            source_index[sample_dir.name] = files_by_object_id

    return source_index


def _prepare_training_dataset(
    project: Path,
    export_tsv: Path,
    output_root: Path,
    manifest_path: Path,
    force: bool,
    logger,
) -> tuple[Path, Path, int]:
    validated_df = _load_validated_export(export_tsv, logger)
    source_index = _build_project_image_index(project)

    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    skipped = 0
    records = []

    for row in validated_df.itertuples(index=False):
        object_id = row.object_id_clean

        try:
            prefix, file_id = object_id.rsplit("_", 1)
        except ValueError:
            skipped += 1
            continue

        src_file = source_index.get(prefix, {}).get(file_id)
        if src_file is None:
            missing += 1
            continue

        category_dir = output_root / row.category_clean
        category_dir.mkdir(parents=True, exist_ok=True)

        dst_file = category_dir / f"{_safe_path_part(object_id, 'unknown_object')}{src_file.suffix.lower()}"
        shutil.copy2(src_file, dst_file)
        copied += 1
        records.append(
            {
                "object_id": object_id,
                "category": row.category_clean,
                "source_image": str(src_file.resolve()),
                "training_image": str(dst_file.resolve()),
            }
        )

    if copied == 0:
        _raise_train_error(
            "No validated images could be matched between the EcoTaxa export and the project's extracted images.",
            logger,
        )

    pd.DataFrame(records).sort_values(["category", "object_id"]).to_csv(
        manifest_path,
        sep="\t",
        index=False,
    )

    logger.info(f"Validated images copied: {copied}")
    if missing:
        logger.warning(f"Validated objects missing local source image: {missing}")
    if skipped:
        logger.warning(f"Validated objects skipped because of malformed object_id: {skipped}")
    logger.info(f"Training images directory: '{output_root}'")
    logger.info(f"Training dataset manifest: '{manifest_path}'")

    return output_root, manifest_path, copied


def _ensure_PLANKTONCLASS_project(project: Path, training_root: Path, logger) -> None:
    required_paths = [
        training_root / PLANKTONCLASS_CONFIG_FILENAME,
        training_root / DATA_DIRNAME,
        training_root / "models",
    ]
    if all(path.exists() for path in required_paths):
        return

    logger.info(f"Preparing project training layout with 'PLANKTONCLASS init {training_root.name}'")
    try:
        subprocess.run(
            ["PLANKTONCLASS", "init", training_root.name],
            check=True,
            cwd=str(project),
        )
    except subprocess.CalledProcessError as exc:
        _raise_train_error(f"planktonclass init failed with exit code {exc.returncode}", logger)
    except FileNotFoundError as exc:
        _raise_train_error(f"Unable to start planktonclass init: {exc}", logger)


def _set_nested_value(config: dict, section: str, option: str, value) -> None:
    config.setdefault(section, {})
    config[section].setdefault(option, {})
    config[section][option]["value"] = value


def _relative_config_path(target: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()
    except ValueError:
        return target.resolve().as_posix()


def _prepare_training_config(training_root: Path, images_dir: Path, logger) -> Path:
    config_path = training_root / PLANKTONCLASS_CONFIG_FILENAME
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        _raise_train_error(f"Failed to read training config '{config_path}': {exc}", logger)

    _set_nested_value(config, "general", "base_directory", ".")
    _set_nested_value(config, "general", "images_directory", _relative_config_path(images_dir, training_root))
    _set_nested_value(config, "testing", "timestamp", "")
    _set_nested_value(config, "testing", "ckpt_name", "")

    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8")
    logger.info(f"Training config ready at '{config_path}'")
    logger.info(f"Training images will be read from '{images_dir}'")
    return config_path


def _latest_local_model_dir(training_root: Path) -> Path | None:
    models_dir = training_root / "models"
    if not models_dir.exists():
        return None

    candidates = sorted(path for path in models_dir.iterdir() if path.is_dir())
    if not candidates:
        return None
    return candidates[-1]


def run(
    ctx: click.Context,
    project: Path,
    export_tsv: Path | None = None,
    force: bool = False,
    config_only: bool = False,
):
    logger = setup_logging(command="train", project=project, debug=ctx.obj["debug"])
    log_command_start(logger, "Preparing training dataset and model", project)

    _check_training_project_inputs(project, logger)
    training_root = project / TRAINING_ROOT_DIRNAME
    _ensure_PLANKTONCLASS_project(project, training_root, logger)

    resolved_export_tsv = _resolve_export_tsv(project, export_tsv, logger)

    data_dir = project / DATA_DIRNAME
    training_images_dir = data_dir / TRAINING_IMAGES_DIRNAME
    manifest_path = data_dir / TRAINING_MANIFEST_FILENAME

    _prepare_training_dataset(
        project=project,
        export_tsv=resolved_export_tsv,
        output_root=training_images_dir,
        manifest_path=manifest_path,
        force=force,
        logger=logger,
    )
    config_path = _prepare_training_config(training_root=training_root, images_dir=training_images_dir, logger=logger)

    if config_only:
        logger.info(
            f"Training dataset and config are ready. Edit '{config_path}' if needed, then run:\n"
            f"  cytoprocess train '{project}'"
        )
        log_command_success(logger, "Prepare training config")
        return

    try:
        subprocess.run(
            ["planktonclass", "train", "--config", str(config_path.resolve())],
            check=True,
            cwd=str(training_root),
        )
    except subprocess.CalledProcessError as exc:
        _raise_train_error(f"planktonclass training failed with exit code {exc.returncode}", logger)
    except FileNotFoundError as exc:
        _raise_train_error(f"Unable to start planktonclass training: {exc}", logger)

    latest_model_dir = _latest_local_model_dir(training_root)
    if latest_model_dir is not None:
        logger.info(f"Latest trained model: '{latest_model_dir}'")

    log_command_success(logger, "Train local classifier")

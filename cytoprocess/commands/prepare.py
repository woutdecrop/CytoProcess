import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import click
import pandas as pd

from cytoprocess.logging import setup_logging, log_command_start, log_command_success
from cytoprocess.project import list_sample_assets, path_to_sample_asset
from cytoprocess.utils import raiseCytoError


class MissingEcoTaxaMediaError(RuntimeError):
    """Raised when the EcoTaxa TSV would reference media files missing from disk."""

    def __init__(self, sample_id: str, missing_files: list[str]):
        self.sample_id = sample_id
        self.missing_files = missing_files
        super().__init__(sample_id)


def _infer_ecotaxa_type(series: pd.Series) -> str:
    """
    Infer EcoTaxa column type from pandas Series.
        
    Args:
        series: pandas Series
        
    Returns:
        str: '[t]' for text columns and '[f]' for numeric columns.
    """
    # Check if column is numeric
    if pd.api.types.is_numeric_dtype(series):
        return '[f]'
    else:
        return '[t]'


def _warn_about_extra_samples(project: Path, samples: list[str], logger: logging.Logger) -> None:
    """
    List samples in raw that are not included in the provided sample list, and log a warning if any are found.
    
    Args:
        project: Path to project directory
        samples: List of existing sample ids
        logger: Logger instance
        
    Returns:
        List of sample_ids to process
    """
    avail_raw_files = list_sample_assets(project, kind="cyz", logger=logger)
    used_raw_files = [project / path_to_sample_asset(s, kind="cyz", logger=logger) for s in samples]
    extra_raw_files = set(avail_raw_files) - set(used_raw_files)
    if len(extra_raw_files) == 1:
        logger.warning(f"Found one unprocessed .cyz file':\n{(str(list(extra_raw_files)[0]))}\nTo include it, re-run `cytoprocess list {project}`, fill in the metadata for this sample, run all processing steps, and then re-run `cytoprocess prepare {project}`.")
    elif len(extra_raw_files) > 1:
        logger.warning(f"Found {len(extra_raw_files)} unprocessed .cyz files':\n{sorted(str(f) for f in extra_raw_files)}\nTo include them, re-run `cytoprocess list {project}`, fill in the metadata for these samples, run all processing steps, and then re-run `cytoprocess prepare {project}`.")


def _ensure_complete_samples(project: Path, samples: list[str], logger: logging.Logger) -> None:
    """
    Validate all required input data/files exist for requested samples.
    
    Args:
        project: Path to project directory
        samples: List of sample_ids to validate
        logger: Logger instance
        
    Raises:
        CytoError if any required files are missing
    """

    # Read the global meta/samples.csv to get a list of the sample ids it contains
    meta_file = project / "meta" / "samples.csv"
    logger.debug(f"Checking that '{meta_file}' exists for sample validation")
    if not meta_file.exists():
        raiseCytoError(f"Missing samples metadata file, run `cytoprocess list {project}`.", logger)
    logger.debug(f"Reading samples list from '{meta_file}'")
    meta_df = pd.read_csv(meta_file, usecols=['sample_id'])
    samples_in_meta = meta_df['sample_id'].unique().tolist()

    logger.debug("Verifying required input files for all requested samples")    
    at_least_one_missing = False
    for sample_id in samples:

        if sample_id not in samples_in_meta:
            logger.warning(f"Sample '{sample_id}' not found in '{meta_file}', run `cytoprocess list {project}` to update the sample list and fill in the metadata for this sample")
            at_least_one_missing = True

        sample_meta_file = project / path_to_sample_asset(sample_id, 'metadata', logger)
        if not sample_meta_file.exists():
            logger.warning(f"Missing metadata from the instrument, run `cytoprocess --sample '{sample_id}' extract_meta {project}`")
            at_least_one_missing = True

        cytometric_features_file = project / path_to_sample_asset(sample_id, 'cytometric_features', logger)
        if not cytometric_features_file.exists():
            logger.warning(f"Missing cytometric features, run `cytoprocess --sample '{sample_id}' extract_cyto {project}`")
            at_least_one_missing = True
        # TODO check consistency in number of objects, features, images, etc.
        # TODO use status here (most checks are done already)

        pulses_summaries_file = project / path_to_sample_asset(sample_id, 'pulses_summaries', logger)
        if not pulses_summaries_file.exists():
            logger.warning(f"Missing pulses summaries, run `cytoprocess --sample '{sample_id}' summarise_pulses {project}`")
            at_least_one_missing = True

        pulses_plots_dir = project / path_to_sample_asset(sample_id, 'pulses_plots', logger)
        if not pulses_plots_dir.exists():
            logger.warning(f"Missing pulses plots directory, run `cytoprocess --sample '{sample_id}' summarise_pulses {project}`")
            at_least_one_missing = True

        images_dir = project / path_to_sample_asset(sample_id, 'images', logger)
        if not images_dir.exists():
            logger.warning(f"Images not found, run `cytoprocess --sample '{sample_id}' extract_images {project}`")
            at_least_one_missing = True

        image_features_file = project / path_to_sample_asset(sample_id, 'image_features', logger)
        if not image_features_file.exists():
            logger.warning(f"Missing image features, run `cytoprocess --sample '{sample_id}' extract_images {project}`")
            at_least_one_missing = True

    if at_least_one_missing:
        raiseCytoError("Missing input for some samples. Please run the required extraction steps before preparing EcoTaxa files.", logger)
    

def _recover_prepare_prerequisites(ctx: click.Context, project: Path, samples: list[str], logger: logging.Logger) -> None:
    from cytoprocess.commands import extract_images, summarise_pulses

    samples_needing_images: list[str] = []
    samples_needing_pulses: list[str] = []

    for sample_id in samples:
        pulses_summaries_file = project / path_to_sample_asset(sample_id, 'pulses_summaries', logger)
        pulses_plots_dir = project / path_to_sample_asset(sample_id, 'pulses_plots', logger)
        if not pulses_summaries_file.exists() or not pulses_plots_dir.exists():
            samples_needing_pulses.append(sample_id)

        images_dir = project / path_to_sample_asset(sample_id, 'images', logger)
        image_features_file = project / path_to_sample_asset(sample_id, 'image_features', logger)
        if not images_dir.exists() or not image_features_file.exists():
            samples_needing_images.append(sample_id)

    previous_sample = ctx.obj.get("sample")
    try:
        for sample_id in samples_needing_pulses:
            logger.warning(f"Auto-repair: re-running `summarise_pulses` for '{sample_id}'")
            ctx.obj["sample"] = sample_id
            summarise_pulses.run(ctx, project=project, n_poly=10, force=True, max_cores=None)

        for sample_id in samples_needing_images:
            logger.warning(f"Auto-repair: re-running `extract_images` for '{sample_id}'")
            ctx.obj["sample"] = sample_id
            extract_images.run(ctx, project=project, force=True, max_cores=None)
    finally:
        ctx.obj["sample"] = previous_sample


def _merge_sample_data(
    project: Path,
    sample_id: str,
    samples_meta: pd.DataFrame,
    logger,
    include_predictions: bool = True,
) -> pd.DataFrame:
    """
    Merge all data sources for a sample into a single DataFrame.
    
    Args:
        project: Path to project directory
        sample_id: The sample identifier
        samples_meta: DataFrame with user added sample-level metadata
        logger: Logger instance
        
    Returns:
        Merged DataFrame
    """

    # Get sample-level metadata for this sample
    sample_meta_df = samples_meta[samples_meta['sample_id'] == sample_id]
    
    # Read instrument metadata for this sample (one row)
    instrument_meta_df = pd.read_parquet(project / path_to_sample_asset(sample_id, 'metadata', logger))

    # Read object level data for this sample
    cytometric_df = pd.read_parquet(project / path_to_sample_asset(sample_id, 'cytometric_features', logger))
    image_features_df = pd.read_parquet(project / path_to_sample_asset(sample_id, 'image_features', logger))
    pulses_summaries_df = pd.read_parquet(project / path_to_sample_asset(sample_id, 'pulses_summaries', logger))

    # If the cytometric dataframe is empty, it means there were no particles detected for this sample,
    # so we can skip the rest of the processing and return empty results
    if cytometric_df.empty:
        return pd.DataFrame()

    # Merge all data
    df = cytometric_df.merge(image_features_df, on=['sample_id', 'object_id'], how='left')
    df = df.merge(pulses_summaries_df, on=['sample_id', 'object_id'], how='left')
    df = df.merge(sample_meta_df, on=['sample_id'], how='left')
    df = df.merge(instrument_meta_df, on=['sample_id'], how='left')

    # Optionally merge predictions if they exist
    predictions_file = project / path_to_sample_asset(sample_id, 'predictions', logger)
    if include_predictions and predictions_file.exists():
        predictions_df = pd.read_parquet(predictions_file)
        df = df.merge(predictions_df, on=['sample_id', 'object_id'], how='left')
        logger.debug(f"Merged predictions for sample '{sample_id}'")

    # Prepend sample id to acq_id to avoid conflicts
    # (and name process id the same)
    df['acq_id'] = df['sample_id'] + "_" + df['acq_id']
    df['process_id'] = df['acq_id']

    logger.debug(f"Found {len(df)} objects for sample '{sample_id}'")
    
    return df


def _format_missing_media_message(sample_id: str, missing_files: list[str]) -> str:
    preview = ", ".join(f"'{name}'" for name in missing_files[:10])
    suffix = "" if len(missing_files) <= 10 else f", ... ({len(missing_files)} total)"
    return (
        f"Sample '{sample_id}' is missing {len(missing_files)} EcoTaxa media file(s): "
        f"{preview}{suffix}. Re-run the extraction steps or remove the corresponding rows before preparing."
    )


def _skipped_objects_report_path(project: Path, sample_id: str, logger) -> Path:
    sample_dir = project / path_to_sample_asset(sample_id, "dir", logger)
    return sample_dir / "prepare_skipped_objects.csv"


def _skipped_objects_master_log_path(project: Path) -> Path:
    return project / "logs" / "prepare_skipped_objects_master.csv"


def _pipeline_failures_master_log_path(project: Path) -> Path:
    return project / "logs" / "pipeline_failures_master.csv"


def _append_pipeline_failure_rows(project: Path, rows: list[dict]) -> None:
    if not rows:
        return
    failure_log_path = _pipeline_failures_master_log_path(project)
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not failure_log_path.exists()
    pd.DataFrame(rows).to_csv(failure_log_path, mode="a", header=write_header, index=False)


def _object_token_from_missing_file(file_name: str) -> str:
    for suffix in ("_img.jpg", "_mask.png", "_pulses.png"):
        if file_name.endswith(suffix):
            return file_name[: -len(suffix)]
    return Path(file_name).stem


def _missing_files_by_object(sample_id: str, missing_files: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    prefix = f"{sample_id}_"
    for file_name in missing_files:
        object_token = _object_token_from_missing_file(file_name)
        object_id = f"{prefix}{object_token}"
        grouped.setdefault(object_id, []).append(file_name)
    return grouped


def _write_skipped_objects_report(project: Path, sample_id: str, missing_files: list[str], logger) -> Path:
    report_path = _skipped_objects_report_path(project, sample_id, logger)
    grouped = _missing_files_by_object(sample_id, missing_files)
    event_timestamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for object_id, object_missing_files in sorted(grouped.items()):
        object_token = object_id.replace(f"{sample_id}_", "", 1)
        rows.append(
            {
                "event_timestamp_utc": event_timestamp,
                "sample_id": sample_id,
                "object_id": object_id,
                "object_local_id": object_token,
                "reason": "missing_media_after_regeneration",
                "action": "dropped_from_prepare_export",
                "missing_files": ";".join(sorted(object_missing_files)),
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df = pd.DataFrame(rows)
    report_df.to_csv(report_path, index=False)
    _append_skipped_objects_master_log(project, report_df)
    _append_pipeline_failure_rows(
        project,
        [
            {
                "event_timestamp_utc": row["event_timestamp_utc"],
                "stage": "prepare",
                "sample_id": row["sample_id"],
                "object_id": row["object_id"],
                "status": "dropped",
                "reason": row["reason"],
                "details": row["missing_files"],
            }
            for row in rows
        ],
    )
    return report_path


def _clear_skipped_objects_report(project: Path, sample_id: str, logger) -> None:
    report_path = _skipped_objects_report_path(project, sample_id, logger)
    report_path.unlink(missing_ok=True)


def _drop_objects_with_missing_media(df: pd.DataFrame, sample_id: str, missing_files: list[str]) -> pd.DataFrame:
    object_ids_to_drop = set(_missing_files_by_object(sample_id, missing_files))
    if not object_ids_to_drop:
        return df
    return df[~df["object_id"].isin(object_ids_to_drop)].copy()


def _append_skipped_objects_master_log(project: Path, report_df: pd.DataFrame) -> None:
    master_log_path = _skipped_objects_master_log_path(project)
    master_log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not master_log_path.exists()
    report_df.to_csv(master_log_path, mode="a", header=write_header, index=False)


def _ensure_referenced_media_exists(project: Path, sample_id: str, df: pd.DataFrame, logger) -> None:
    """Fail early when the EcoTaxa TSV would reference files missing from disk."""
    images_dir = project / path_to_sample_asset(sample_id, 'images', logger)
    pulses_dir = project / path_to_sample_asset(sample_id, 'pulses_plots', logger)

    missing_files = []
    for file_name in df["img_file_name"].dropna().astype(str).unique():
        if file_name.endswith("_pulses.png"):
            candidate = pulses_dir / file_name
        else:
            candidate = images_dir / file_name

        if not candidate.exists():
            missing_files.append(file_name)

    if missing_files:
        raise MissingEcoTaxaMediaError(sample_id, missing_files)


def _repair_missing_media(ctx: click.Context, project: Path, sample_id: str, missing_files: list[str], logger) -> None:
    from cytoprocess.commands import extract_images, summarise_pulses

    needs_images = any(name.endswith(("_img.jpg", "_mask.png")) for name in missing_files)
    needs_pulses = any(name.endswith("_pulses.png") for name in missing_files)

    previous_sample = ctx.obj.get("sample")
    ctx.obj["sample"] = sample_id
    try:
        if needs_pulses:
            logger.warning(f"  Missing pulse plots detected, re-running `summarise_pulses` for '{sample_id}'")
            summarise_pulses.run(ctx, project=project, n_poly=10, force=True, max_cores=None)
        if needs_images:
            logger.warning(f"  Missing images or masks detected, re-running `extract_images` for '{sample_id}'")
            extract_images.run(ctx, project=project, force=True, max_cores=None)
    finally:
        ctx.obj["sample"] = previous_sample


def _prepare_ecotaxa_tsv(df: pd.DataFrame, tsv_file: Path, logger) -> pd.DataFrame:
    """
    Prepare and write EcoTaxa TSV file with column type inference.
    
    Enforces EcoTaxa column limits and writes TSV with type indicator row.
    
    Args:
        df: Merged DataFrame with all sample data
        tsv_file: Path to output TSV file
        logger: Logger instance
        
    Returns:
        The sorted DataFrame used for the TSV (for further processing)
    """
    # Get the sample_id value from the assembled data (same for all rows)
    sample_id = df["sample_id"].iloc[0]
    
    # Add image filename based on object_id (this is the actual image)
    df['img_file_name'] = df['object_id'].str.replace(f"{sample_id}_", "", n=1) + "_img.jpg"
    
    # Add img_rank (0-based index for multiple images per object)
    df['img_rank'] = 0

    # Reorder columns to put all *_id columns first, for cleanness
    id_cols = [c for c in df.columns if c.endswith('_id')]
    other_cols = [c for c in df.columns if not c.endswith('_id')]
    df = df[id_cols + other_cols]

    # Count columns per prefix and enforce EcoTaxa limits
    cols = df.columns.tolist()
    img_cols = [c for c in cols if c.startswith('img_')]
    object_cols = [c for c in cols if c.startswith('object_')]
    # For objects, some columns do not count towards the maximum of 500 free metadata columns
    object_cols_exclude = {"object_lon", "object_lat", "object_date", "object_time", "object_depth_min", "object_depth_max"}
    object_cols_for_nb = [c for c in object_cols if not c.startswith('object_annotation_') and c not in object_cols_exclude]
    process_cols = [c for c in cols if c.startswith('process_')]
    acq_cols = [c for c in cols if c.startswith('acq_')]
    sample_cols = [c for c in cols if c.startswith('sample_')]

    # Limit object metadata columns to 500
    # NB: since object_id does not count as metadata, this means a maximum of 501 columns
    if len(object_cols_for_nb) > 501:
        logger.warning(f"Sample '{sample_id}' has {len(object_cols)-1} object metadata columns, truncating to 500 (EcoTaxa limit)")
        object_cols = object_cols[:501]
    # Limit sample, process, and acq columns
    if len(process_cols) > 31:
        logger.warning(f"Sample '{sample_id}' has {len(process_cols)-1} process metadata columns, truncating to 30 (EcoTaxa limit)")
        process_cols = process_cols[:31]
    if len(acq_cols) > 31:
        logger.warning(f"Sample '{sample_id}' has {len(acq_cols)-1} acq metadata columns, truncating to 30 (EcoTaxa limit)")
        acq_cols = acq_cols[:31]
    if len(sample_cols) > 61:
        logger.warning(f"Sample '{sample_id}' has {len(sample_cols)-1} sample metadata columns, truncating to 60 (EcoTaxa limit)")
        sample_cols = sample_cols[:61]

    # Order columns for cleanness
    ordered_cols = img_cols + object_cols + process_cols + acq_cols + sample_cols
    df = df[ordered_cols]
    
    # Create type indicators row
    type_row = {col: _infer_ecotaxa_type(df[col]) for col in df.columns}

    # Duplicate the DataFrame to reference masks and pulse shape images
    # (they have the same id but are stored as PNG files)
    df_masks = df.copy()
    # TODO review once https://github.com/ecotaxa/ecotaxa/issues/69 is solved
    # # Empty all columns that don't start with "img_" or end with "_id"
    # for col in df_masks.columns:
    #     if not col.startswith('img_') and not col.endswith('_id'):
    #         df_masks[col] = None
    df_masks['img_file_name'] = df_masks['img_file_name'].str.replace('_img.jpg', '_mask.png', regex=False)
    df_masks['img_rank'] = 1

    df_pulses = df_masks.copy()
    df_pulses['img_file_name'] = df_pulses['img_file_name'].str.replace('_mask.png', '_pulses.png', regex=False)
    df_pulses['img_rank'] = 2

    # Combine the two DataFrames
    df = pd.concat([df, df_masks, df_pulses], ignore_index=True)
    # Sort by object_id for consistent ordering
    df = df.sort_values(by="object_id").reset_index(drop=True)
    _ensure_referenced_media_exists(tsv_file.parents[1], sample_id, df, logger)
    
    # Create the EcoTaxa .tsv file
    with open(tsv_file, 'w') as f:
        f.write('\t'.join(df.columns) + '\n')
        f.write('\t'.join([type_row[col] for col in df.columns]) + '\n')
        df.to_csv(f, sep='\t', index=False, header=False)
    
    logger.debug(f"Saved {df.shape[1]} fields for {int(df.shape[0]/2)} objects to '{tsv_file}'")
    return df


def _create_ecotaxa_zip(project: Path, sample_id: str, tsv_file: Path, zip_file: Path, logger) -> None:
    """
    Create EcoTaxa ZIP file containing TSV and processed images with scale bars.
    
    Cleans up temporary files (TSV and processed images) after creating the ZIP.
    
    Args:
        project: Path to project directory
        sample_id: ID of the sample for which to create the ZIP file
        tsv_file: Path to the TSV file to include
        zip_file: Path to output ZIP file
        logger: Logger instance
    """

    # List images to include in the zip file
    images_dir = project / path_to_sample_asset(sample_id, 'images', logger)
    pulses_dir = project / path_to_sample_asset(sample_id, 'pulses_plots', logger)

    image_files = list(images_dir.glob("*_img.jpg"))
    mask_files = list(images_dir.glob("*_mask.png"))
    pulses_files = list(pulses_dir.glob("*_pulses.png"))

    all_images = image_files + mask_files + pulses_files
    
    logger.debug(f"Creating zip file '{zip_file}'")
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(tsv_file, tsv_file.name)

        for image_file in all_images:
            zf.write(image_file, image_file.name)
    
    logger.debug(f"Created zip file '{zip_file}' with {len(pulses_files)} objects")

    # Remove the TSV file after adding it to the zip
    logger.debug(f"Removing temporary TSV file '{tsv_file}'")
    tsv_file.unlink()


def run(ctx: click.Context, project: Path, force=False, include_predictions: bool = True):
    # Housekeeping for the command
    logger = setup_logging(command="prepare", project=project, debug=ctx.obj["debug"])
    log_command_start(logger, "Preparing EcoTaxa files", project)
    logger.debug("Context: %s", getattr(ctx, "obj", {}))
    if force:
        logger.debug("Force flag enabled, existing ecotaxa files will be overwritten")


    # List samples in work, filtered by --sample if provided
    samples_mask = ctx.obj["sample"]
    sample_dirs = list_sample_assets(project, "dir", logger, samples_mask=samples_mask)
    if not sample_dirs:
        return
    sample_ids = [d.name for d in sample_dirs]

    logger.info(f"Preparing EcoTaxa file for {len(sample_ids)} sample(s)")

    # If we are not filtering samples, detect extra ones in raw and warn the user
    if samples_mask == None:
        _warn_about_extra_samples(project, sample_ids, logger)

    # Check that all required input data/files exist for the target sample(s)
    _recover_prepare_prerequisites(ctx, project, sample_ids, logger)
    _ensure_complete_samples(project, sample_ids, logger)

    # Prepare storage
    ecotaxa_dir = project / "ecotaxa"
    ecotaxa_dir.mkdir(parents=True, exist_ok=True)

    # Read the global sample-level metadata (one row per sample)
    # to merge it below with the sample-level information
    samples_meta_df = pd.read_csv(project / "meta" / "samples.csv")

    for sample_id in sample_ids:
        logger.info(f"'{sample_id}'")

        zip_file = project / path_to_sample_asset(sample_id, 'zip', logger)
        tsv_file = project / path_to_sample_asset(sample_id, 'tsv', logger)

        # Skip if output file exists and force is not set
        if (zip_file.exists() and not force):
            logger.info(f"  Skipping, {zip_file} file already exists (use --force to overwrite)")
            continue
        
        logger.info(f"  Collating '{tsv_file}'")

        # Merge all data for this sample
        df = _merge_sample_data(
            project,
            sample_id,
            samples_meta_df,
            logger,
            include_predictions=include_predictions,
        )
        # If the merged dataframe is empty, skip to the next sample
        if df.empty:
            logger.warning(f"No imaged particles for sample '{sample_id}', skipping.")
            continue

        # Prepare TSV file
        try:
            _prepare_ecotaxa_tsv(df, tsv_file, logger)
            _clear_skipped_objects_report(project, sample_id, logger)
        except MissingEcoTaxaMediaError as exc:
            logger.warning(f"  {_format_missing_media_message(exc.sample_id, exc.missing_files)}")
            logger.warning("  Attempting to regenerate the missing media before retrying prepare")
            _repair_missing_media(ctx, project, sample_id, exc.missing_files, logger)
            df = _merge_sample_data(
                project,
                sample_id,
                samples_meta_df,
                logger,
                include_predictions=include_predictions,
            )
            if df.empty:
                logger.warning(f"No imaged particles for sample '{sample_id}' after media regeneration, skipping.")
                continue
            try:
                _prepare_ecotaxa_tsv(df, tsv_file, logger)
                _clear_skipped_objects_report(project, sample_id, logger)
            except MissingEcoTaxaMediaError as retry_exc:
                report_path = _write_skipped_objects_report(project, sample_id, retry_exc.missing_files, logger)
                df = _drop_objects_with_missing_media(df, sample_id, retry_exc.missing_files)
                if df.empty:
                    raiseCytoError(
                        f"After excluding unrecoverable objects, sample '{sample_id}' has no objects left for EcoTaxa. "
                        f"Skipped objects are listed in '{report_path}'.",
                        logger,
                    )
                skipped_count = len(_missing_files_by_object(sample_id, retry_exc.missing_files))
                logger.warning(
                    f"  Excluding {skipped_count} object(s) with unrecoverable media from sample '{sample_id}'. "
                    f"Details saved to '{report_path}'."
                )
                _prepare_ecotaxa_tsv(df, tsv_file, logger)
        
        # Create zip file
        logger.info(f"  Assembling '{zip_file}'")
        _create_ecotaxa_zip(project, sample_id, tsv_file, zip_file, logger)

    log_command_success(logger, "Prepare EcoTaxa files")

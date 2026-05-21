from pathlib import Path
import re

import click
import pandas as pd

from cytoprocess.logging import setup_logging, log_command_start, log_command_success
from cytoprocess.project import list_sample_assets


DEFAULT_EXTRA_FIELDS = "object_lon,object_lat,object_date,object_time,object_depth_min,object_depth_max,object_lon_end,object_lat_end"
SAMPLE_DATETIME_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})(?:%20|[_\s-]+)(?P<hour>\d{2})h(?P<minute>\d{2})")


def _date_time_from_sample_id(sample_id: str) -> tuple[str | None, str | None]:
    match = SAMPLE_DATETIME_RE.search(str(sample_id))
    if not match:
        return None, None
    return match.group("date"), f"{match.group('hour')}:{match.group('minute')}:00"


def _is_empty(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _fill_sample_date_time(samples: pd.DataFrame, logger) -> pd.DataFrame:
    if "object_date" not in samples.columns and "object_time" not in samples.columns:
        return samples

    samples = samples.copy()
    filled_date = 0
    filled_time = 0
    for index, sample_id in samples["sample_id"].items():
        sample_date, sample_time = _date_time_from_sample_id(sample_id)
        if sample_date and "object_date" in samples.columns and _is_empty(samples.at[index, "object_date"]):
            samples.at[index, "object_date"] = sample_date
            filled_date += 1
        if sample_time and "object_time" in samples.columns and _is_empty(samples.at[index, "object_time"]):
            samples.at[index, "object_time"] = sample_time
            filled_time += 1

    if filled_date or filled_time:
        logger.info(f"Filled object_date for {filled_date} sample(s) and object_time for {filled_time} sample(s) from sample names")
    return samples


def run(ctx: click.Context, project: Path, extra_fields=DEFAULT_EXTRA_FIELDS):
    # Housekeeping for the command
    logger = setup_logging(command="list", project=project, debug=ctx.obj["debug"])
    log_command_start(logger, "Listing samples", project)
    logger.debug("Context: %s", getattr(ctx, "obj", {}))


    # List raw files
    raw_files = list_sample_assets(project, kind="cyz",
                                   logger=logger, samples_mask=ctx.obj["sample"])


    # If there are none, warn and exit
    if not raw_files:
        logger.warning(f"Then copy/move cyz files to '{project}/raw'")
        return


    # If there are some, print them to the console
    logger.info(f"{len(raw_files)} sample(s) found")
    for file in raw_files:
        print(f"   {file.stem}")


    # And write them to the metadata file
    # Parse extra fields
    if extra_fields:
        extra_field_list = [f.strip() for f in extra_fields.split(',') if f.strip()]
    else:
        extra_field_list = []
    logger.debug(f"Extra fields: {extra_field_list}")
    # TODO if the file exists and extra_fields is not explicitly provided, just keep the columns already existing (to avoid having to specify extra_fields everytime); it might be the case already

    # Create metadata CSV with sample information   
    meta_dir = project / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_file = meta_dir / "samples.csv"
    
    # Create 'samples' DataFrame
    samples = pd.DataFrame({
        'sample_id': [f.stem for f in raw_files]
    })
    for field in extra_field_list:
        samples[field] = None
    samples = _fill_sample_date_time(samples, logger)
    
    # Read existing metadata if it exists, otherwise create new
    update_meta_file = True
    if meta_file.exists():
        existing_samples = pd.read_csv(meta_file)
        
        # Detect which samples are new
        new_samples = samples[~samples['sample_id'].isin(existing_samples['sample_id'])]
        
        # If there are no new samples, just ensure extra fields are present
        if new_samples.empty:
            missing_fields = [f for f in extra_field_list if f not in existing_samples.columns]
            final_df = existing_samples
            if not missing_fields:
                final_df = _fill_sample_date_time(final_df, logger)
                if final_df.equals(existing_samples):
                    logger.info(f"No new samples or fields to add to '{meta_file}'")
                    # In that case do not even rewrite the file
                    update_meta_file = False
            else:
                logger.info(f"Adding {len(missing_fields)} new field(s) to '{meta_file}'")
                for field in missing_fields:
                    logger.debug(f"Adding new column '{field}' to '{meta_file}'")
                    final_df[field] = None
                final_df = _fill_sample_date_time(final_df, logger)
        # TODO remove samples that are not longer present in the raw directory
                    
        # If there are new samples, append them
        else:
            # Detect potentially missing fields in existing samples to inform the user about it
            missing_fields = [f for f in extra_field_list if f not in existing_samples.columns]
            logger.info(f"Adding {len(new_samples)} new sample(s)" + (f" and {len(missing_fields)} new field(s)" if missing_fields else "") + f" to '{meta_file}'")
            logger.debug(f"Missing samples: {new_samples['sample_id'].tolist()}")
            logger.debug(f"Missing fields: {missing_fields}")
            final_df = pd.concat([existing_samples, new_samples], ignore_index=True)
            for field in missing_fields:
                if field not in final_df.columns:
                    final_df[field] = None
            final_df = _fill_sample_date_time(final_df, logger)
   
    else:
        final_df = samples
        logger.info(f"Created file '{meta_file}' with {len(samples)} sample(s) and {samples.shape[1]-1} field(s), you can now add custom metadata.")
    
    # Still save if we added new columns
    if update_meta_file:
        final_df.to_csv(meta_file, index=False)
 
    log_command_success(logger, "List samples")

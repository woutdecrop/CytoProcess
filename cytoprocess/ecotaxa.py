import base64
import getpass
import logging
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import keyring
import requests

from cytoprocess.utils import format_file_size, raiseCytoError

KEYRING_SERVICE = "cytoprocess-ecotaxa"
TUS_CREATE_TIMEOUT_SEC = 30
TUS_PATCH_TIMEOUT_SEC = 120
TUS_OFFSET_TIMEOUT_SEC = 30
TUS_MAX_RETRIES = 5
TUS_RETRY_BACKOFF_SEC = 5
JOB_STATUS_MAX_RETRIES = 5
JOB_STATUS_RETRY_BACKOFF_SEC = 5


def _get_stored_token(logger: logging.Logger) -> str | None:
    """Retrieve stored token from keyring."""
    try:
        return keyring.get_password(KEYRING_SERVICE, "token")
    except Exception as e:
        logger.debug(f"Could not retrieve token from keyring: {e}")
        return None


def _store_token(logger: logging.Logger, token: str) -> bool:
    """Store token in keyring."""
    try:
        keyring.set_password(KEYRING_SERVICE, "token", token)
        return True
    except Exception as e:
        logger.warning(f"Could not store token in keyring: {e}")
        return False


def _clear_token(logger: logging.Logger) -> None:
    """Clear stored token from keyring."""
    try:
        keyring.delete_password(KEYRING_SERVICE, "token")
    except Exception:
        pass


def _validate_token(api_url: str, token: str, logger: logging.Logger) -> bool:
    """Check if the token is still valid by calling /users/me."""
    try:
        response = requests.get(
            f"{api_url}/users/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def _login(api_url: str, username: str, password: str, logger: logging.Logger) -> str | None:
    """
    Authenticate with EcoTaxa API and return JWT token.

    Returns None if authentication fails.
    """
    try:
        response = requests.post(
            f"{api_url}/login",
            json={"username": username, "password": password},
            timeout=30,
        )
        if response.status_code == 200:
            # The API returns the token as a plain string (JSON string)
            return response.json()
        logger.error(f"Login failed: {response.text}")
        return None
    except requests.RequestException as e:
        logger.error(f"Login request failed: {e}")
        return None


def _get_user_info(api_url: str, token: str, logger: logging.Logger) -> dict | None:
    """Get current user information."""
    try:
        response = requests.get(
            f"{api_url}/users/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None


def get_project_info(api_url: str, project_id: int, token: str, logger: logging.Logger) -> dict | None:
    """
    Get project information from EcoTaxa.

    Args:
        api_url: EcoTaxa API URL
        project_id: EcoTaxa project ID
        token: JWT authentication token
        logger: Logger instance

    Returns:
        Project information dict or None if request fails.
        Contains fields like 'title', 'projid', 'status', etc.
    """
    try:
        response = requests.get(
            f"{api_url}/projects/{project_id}",
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
    except requests.RequestException as e:
        logger.error(f"Failed to get project info: {e}")
        return None


def get_project_samples(api_url: str, project_id: int, token: str, logger: logging.Logger) -> dict[str, int]:
    """
    Get the mapping of sample orig_id to EcoTaxa sample id for a project.

    Args:
        api_url: EcoTaxa API URL
        project_id: EcoTaxa project ID
        token: JWT authentication token
        logger: Logger instance

    Returns:
        Dict mapping sample orig_id to EcoTaxa sampleid.
    """
    try:
        response = requests.get(
            f"{api_url}/samples/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"project_ids": str(project_id), "id_pattern": "*"},
            timeout=60,
        )
        if response.status_code == 200:
            sample_map: dict[str, int] = {}
            for sample in response.json():
                orig_id = sample.get("orig_id")
                sample_id = sample.get("sampleid")
                if orig_id and sample_id is not None:
                    sample_map[str(orig_id)] = int(sample_id)
            return sample_map
        logger.warning(f"Failed to get samples: {response.text}")
        return {}
    except requests.RequestException as e:
        logger.warning(f"Failed to get project samples: {e}")
        return {}


def authenticate(
    api_url: str,
    username: str | None = None,
    password: str | None = None,
    logger: logging.Logger = None,
) -> str | None:
    """
    Authenticate with EcoTaxa API.

    First tries to use a stored token. If not available or invalid,
    uses provided credentials or prompts the user.

    Args:
        api_url: EcoTaxa API URL
        username: Optional email address. If not provided, will prompt.
        password: Optional password. If not provided, will prompt.
        logger: Logger instance

    Returns:
        JWT token if authentication successful, None otherwise.
    """
    # Try stored token first
    token = _get_stored_token(logger)
    if token and _validate_token(api_url, token, logger):
        user_info = _get_user_info(api_url, token, logger)
        if user_info:
            logger.debug(
                f"Authenticated as: {user_info.get('name', 'Unknown')} ({user_info.get('email', 'Unknown')})"
            )
        return token
    if token:
        logger.warning("Stored token is invalid, need to re-authenticate")
        _clear_token(logger)

    # Use provided credentials or prompt
    if not username:
        print("\nEcoTaxa Authentication Required")
        username = input("username (email): ").strip()
    if not username:
        raiseCytoError("EcoTaxa username is required", logger)

    if not password:
        password = getpass.getpass("password: ")
    if not password:
        raiseCytoError("EcoTaxa password is required", logger)

    # Attempt login
    token = _login(api_url, username, password, logger)
    if token is None:
        raiseCytoError(
            "Authentication failed. Please check your EcoTaxa username and password.",
            logger,
        )

    # Store the token
    if _store_token(logger, token):
        logger.info("Authentication token stored securely in system keyring")

    # Show user info
    user_info = _get_user_info(api_url, token, logger)
    if user_info:
        logger.info(
            f"Authenticated as: {user_info.get('name', 'Unknown')} ({user_info.get('email', 'Unknown')})"
        )

    return token


def _list_user_files(api_url: str, token: str, sub_path: str = "", logger: logging.Logger = None) -> dict | None:
    """List files in the user's EcoTaxa files directory."""
    try:
        response = requests.get(
            f"{api_url}/user_files/{sub_path}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        logger.debug(f"list_user_files HTTP {response.status_code}: {response.text}")
        return None
    except requests.RequestException as e:
        logger.debug(f"list_user_files failed: {e}")
        return None


def _tus_create_upload(api_url: str, token: str, file_size: int, filename_b64: str, logger: logging.Logger | None) -> str:
    create_headers = {
        "Authorization": f"Bearer {token}",
        "Tus-Resumable": "1.0.0",
        "Upload-Length": str(file_size),
        "Upload-Metadata": f"filename {filename_b64}",
    }
    response = None
    last_error = None
    for attempt in range(1, TUS_MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{api_url}/user_files/upload/",
                headers=create_headers,
                timeout=TUS_CREATE_TIMEOUT_SEC,
            )
        except requests.RequestException as e:
            last_error = e
            if attempt == TUS_MAX_RETRIES:
                raiseCytoError(f"TUS upload creation request failed: {e}", logger)
            if logger:
                logger.warning(
                    "  TUS upload creation request failed (attempt %s/%s): %s",
                    attempt,
                    TUS_MAX_RETRIES,
                    e,
                )
            time.sleep(TUS_RETRY_BACKOFF_SEC * attempt)
            continue

        if response.status_code == 201:
            break

        if response.status_code >= 500 and attempt < TUS_MAX_RETRIES:
            if logger:
                logger.warning(
                    "  TUS upload creation failed with HTTP %s (attempt %s/%s), retrying",
                    response.status_code,
                    attempt,
                    TUS_MAX_RETRIES,
                )
            time.sleep(TUS_RETRY_BACKOFF_SEC * attempt)
            continue

        raiseCytoError(f"TUS upload creation failed (HTTP {response.status_code}): {response.text}", logger)
    else:
        raiseCytoError(
            f"TUS upload creation request failed: {last_error}" if last_error else "TUS upload creation failed",
            logger,
        )

    location = response.headers.get("Location")
    if not location:
        raiseCytoError("TUS upload creation response is missing the Location header", logger)

    if not location.startswith("http"):
        parsed = urlparse(api_url)
        return f"{parsed.scheme}://{parsed.netloc}{location}"
    return location


def _tus_get_offset(upload_url: str, token: str, logger: logging.Logger | None) -> int:
    headers = {
        "Authorization": f"Bearer {token}",
        "Tus-Resumable": "1.0.0",
    }
    try:
        response = requests.head(upload_url, headers=headers, timeout=TUS_OFFSET_TIMEOUT_SEC)
    except requests.RequestException as e:
        raiseCytoError(f"Unable to query TUS upload offset: {e}", logger)

    if response.status_code != 200:
        raiseCytoError(
            f"Unable to query TUS upload offset (HTTP {response.status_code}): {response.text}",
            logger,
        )

    offset_header = response.headers.get("Upload-Offset")
    if offset_header is None:
        raiseCytoError("TUS offset response is missing the Upload-Offset header", logger)

    try:
        return int(offset_header)
    except ValueError:
        raiseCytoError(f"Invalid TUS Upload-Offset value '{offset_header}'", logger)


def upload_file_tus(
    api_url: str,
    token: str,
    zip_path: Path,
    chunk_size: int = 8 * 1024 * 1024,
    logger: logging.Logger = None,
) -> dict:
    """
    Upload a zip file to EcoTaxa using the TUS resumable upload protocol.

    Uploads data in chunks and displays live progress to the terminal.

    Args:
        api_url: EcoTaxa API URL
        token: JWT authentication token
        zip_path: Path to the zip file to upload
        chunk_size: Size of each chunk in bytes (default 8 MB)
        logger: Logger instance

    Returns:
        Dictionary with 'server_path' if successful.
    """
    if not zip_path.exists():
        raiseCytoError(f"File not found: {zip_path}", logger)

    file_size = zip_path.stat().st_size
    filename_b64 = base64.b64encode(zip_path.name.encode()).decode()

    logger.debug(f"TUS upload: creating upload resource for '{zip_path.name}' ({file_size} bytes)")

    upload_url = _tus_create_upload(api_url, token, file_size, filename_b64, logger)

    logger.debug(f"TUS upload resource created: {upload_url}")

    # Step 2: Upload in chunks, reporting progress
    offset = 0
    with open(zip_path, "rb") as f:
        while offset < file_size:
            f.seek(offset)
            chunk = f.read(chunk_size)
            if not chunk:
                break

            patch_headers = {
                "Authorization": f"Bearer {token}",
                "Tus-Resumable": "1.0.0",
                "Content-Type": "application/offset+octet-stream",
                "Content-Length": str(len(chunk)),
                "Upload-Offset": str(offset),
            }

            for attempt in range(1, TUS_MAX_RETRIES + 1):
                try:
                    resp = requests.patch(
                        upload_url,
                        headers=patch_headers,
                        data=chunk,
                        timeout=TUS_PATCH_TIMEOUT_SEC,
                    )
                except requests.RequestException as e:
                    if attempt == TUS_MAX_RETRIES:
                        sys.stdout.write("\n")
                        raiseCytoError(f"TUS upload failed: {e}", logger)
                    if logger:
                        logger.warning(
                            "  TUS chunk upload failed at offset %s (attempt %s/%s): %s",
                            offset,
                            attempt,
                            TUS_MAX_RETRIES,
                            e,
                        )
                    time.sleep(TUS_RETRY_BACKOFF_SEC * attempt)
                    offset = _tus_get_offset(upload_url, token, logger)
                    break

                if resp.status_code == 409:
                    offset = _tus_get_offset(upload_url, token, logger)
                    if logger:
                        logger.warning(f"  TUS server offset changed, resuming at {format_file_size(offset)}")
                    break

                if resp.status_code >= 500:
                    if attempt == TUS_MAX_RETRIES:
                        sys.stdout.write("\n")
                        raiseCytoError(
                            f"TUS chunk upload failed at offset {offset} "
                            f"(HTTP {resp.status_code}): {resp.text}",
                            logger,
                        )
                    if logger:
                        logger.warning(
                            "  TUS chunk upload failed at offset %s with HTTP %s (attempt %s/%s), retrying",
                            offset,
                            resp.status_code,
                            attempt,
                            TUS_MAX_RETRIES,
                        )
                    time.sleep(TUS_RETRY_BACKOFF_SEC * attempt)
                    offset = _tus_get_offset(upload_url, token, logger)
                    break

                if resp.status_code != 204:
                    sys.stdout.write("\n")
                    raiseCytoError(
                        f"TUS chunk upload failed at offset {offset} "
                        f"(HTTP {resp.status_code}): {resp.text}",
                        logger,
                    )

                offset = int(resp.headers.get("Upload-Offset", offset + len(chunk)))
                pct = int(100 * offset / file_size) if file_size else 100
                sys.stdout.write(
                    f"\r  Upload: {pct}% "
                    f"({format_file_size(offset)} / {format_file_size(file_size)})"
                )
                sys.stdout.flush()
                break
            else:
                sys.stdout.write("\n")
                raiseCytoError("TUS upload failed after exhausting all retry attempts", logger)

    sys.stdout.write("\n")

    # Step 3: Determine the server path for the import step.
    # List the user's files directory and look for the file we just uploaded.
    listing = _list_user_files(api_url, token, "", logger)
    if listing:
        base_dir = listing.get("path", "")
        for entry in listing.get("entries", []):
            if entry.get("name") == zip_path.stem and entry.get("type") == "D":
                server_path = f"{base_dir}/{zip_path.stem}"
                logger.debug(f"TUS upload complete, server path: {server_path}")
                return {"server_path": server_path}

    raiseCytoError(
        f"TUS upload completed but '{zip_path.stem}' was not found in user files listing. "
        "The import step cannot proceed.",
        logger,
    )


def import_file(api_url: str, project_id: int, token: str, server_path: str, update_mode: str = "", logger: logging.Logger = None) -> dict:
    """
    Start an import job for a file already uploaded to EcoTaxa.

    Args:
        api_url: EcoTaxa API URL
        project_id: EcoTaxa project ID
        token: JWT authentication token
        server_path: Path to the file on EcoTaxa server (from upload_file)
        logger: Logger instance
        update_mode: Update mode for the import ('Yes' to update data, 'Cla' to also update classification)

    Returns:
        Dictionary with 'job_id' if successful, or 'errors' list if failed.
    """
    logger.info(f"  Importing {server_path}")

    import_req = {
        "source_path": server_path,
        "skip_loaded_files": False,
        "skip_existing_objects": False if update_mode == "" else True,
        "update_mode": update_mode,
    }

    try:
        response = requests.post(
            f"{api_url}/file_import/{project_id}",
            headers={"Authorization": f"Bearer {token}"},
            json=import_req,
            timeout=60,
        )

        # let the import job start
        time.sleep(2)

        if response.status_code == 200:
            result = response.json()
            if result.get("job_id", 0) > 0:
                logger.debug(f"Import job created: {result['job_id']}")
            return result
        raiseCytoError(f"Import failed: {response.text}", logger)

    except requests.RequestException as e:
        raiseCytoError(f"Import request failed: {e}", logger)


def get_job(api_url: str, job_id: int, token: str, logger: logging.Logger) -> dict | None:
    """
    Get job status from EcoTaxa API.

    Args:
        api_url: EcoTaxa API URL
        job_id: Job ID to check
        token: JWT authentication token
        logger: Logger instance

    Returns:
        Job information dict or None if request fails
    """
    try:
        response = requests.get(
            f"{api_url}/jobs/{job_id}/",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None


def _get_job_with_retries(api_url: str, job_id: int, token: str, logger: logging.Logger | None) -> dict | None:
    for attempt in range(1, JOB_STATUS_MAX_RETRIES + 1):
        job_info = get_job(api_url, job_id, token, logger)
        if job_info is not None:
            return job_info
        if attempt == JOB_STATUS_MAX_RETRIES:
            return None
        if logger:
            logger.warning(
                "  Failed to get job status for job %s (attempt %s/%s), retrying",
                job_id,
                attempt,
                JOB_STATUS_MAX_RETRIES,
            )
        time.sleep(JOB_STATUS_RETRY_BACKOFF_SEC * attempt)
    return None


def monitor_job(api_url: str, job_id: int, token: str, poll_interval: float = 2.0, logger: logging.Logger = None) -> bool:
    """
    Monitor a job until it completes.

    Args:
        api_url: EcoTaxa API URL
        job_id: Job ID to monitor
        token: JWT authentication token
        poll_interval: Seconds between status checks
        logger: Logger instance

    Returns:
        True if job completed successfully (state 'F'), False otherwise
    """
    last_progress = -1
    while True:
        job_info = _get_job_with_retries(api_url, job_id, token, logger)
        if job_info is None:
            logger.error("Failed to get job status")
            return False

        state = job_info.get("state", "")
        progress = job_info.get("progress_pct", 0) or 0
        progress_msg = job_info.get("progress_msg", "")

        # Only print if progress changed
        if progress != last_progress:
            print(f"  Progress: {progress}% - {progress_msg}")
            last_progress = progress

        # Check terminal states
        # P: Pending, R: Running, A: Asking, E: Error, F: Finished
        if state == "F":
            logger.debug("Job completed successfully")
            return True
        if state == "E":
            errors = job_info.get("errors", [])
            logger.error(f"Job failed with errors: {errors}")
            return False
        if state == "A":
            # Job is asking for user input - we cannot handle this in CLI
            logger.error("Job requires user input on EcoTaxa web interface")
            return False

        time.sleep(poll_interval)

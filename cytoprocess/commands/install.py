import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import click

from cytoprocess.logging import setup_logging, log_command_start, log_command_success
from cytoprocess.utils import raiseCytoError

PREDICT_MODEL_URL = "https://zenodo.org/records/19709957/files/FlowCytoClassifier.zip?download=1"
PREDICT_MODEL_DIRNAME = "FlowCytoClassifier"
PREDICT_MODEL_FILENAME = "FlowCytoClassifier.zip"
PREDICT_MODEL_DEFAULT_RECORD_ID = "19709957"
PREDICT_MODEL_SOURCE_METADATA = ".cytoprocess_model_source.json"
PREDICT_MODEL_PROJECT_DIRNAME = "models"


def _get_or_create_bin_dir() -> Path:
    """Get (and create if necessary) the directory for storing executables."""
    if platform.system() == "Windows":
        # Prefer a per-user application data location on Windows
        appdata_root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if appdata_root:
            bin_dir = Path(appdata_root) / "Cyz2Json" / "bin"
        else:
            # Fallback if env vars are unavailable
            bin_dir = Path.home() / "AppData" / "Local" / "Cyz2Json" / "bin"
    else:
        # For Unix-like systems, use ~/.bin
        bin_dir = Path.home() / ".bin"

    # create the directory if it doesn't exist
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    return bin_dir


def _get_or_create_model_cache_dir() -> Path:
    """Get (and create if necessary) the directory for storing downloaded model assets."""
    if platform.system() == "Windows":
        appdata_root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if appdata_root:
            cache_dir = Path(appdata_root) / "CytoProcess" / "models"
        else:
            cache_dir = Path.home() / "AppData" / "Local" / "CytoProcess" / "models"
    else:
        cache_dir = Path.home() / ".cache" / "cytoprocess" / "models"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_checkout_neighbor_model_dir() -> Path | None:
    """
    Prefer installing the model next to the cytoprocess checkout when running from a repo clone.

    Example:
      <workspace>/cytoprocess/... -> install model into <workspace>/models
    """
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    workspace_root = repo_root.parent

    if (repo_root / "pyproject.toml").exists():
        workspace_root.mkdir(parents=True, exist_ok=True)
        return workspace_root / PREDICT_MODEL_PROJECT_DIRNAME

    return None


def get_predict_model_install_dir(project: Path | None = None) -> Path:
    """Return the preferred local install path for the downloaded prediction model."""
    if project is not None:
        return project.expanduser().resolve() / PREDICT_MODEL_PROJECT_DIRNAME

    checkout_neighbor = _get_checkout_neighbor_model_dir()
    if checkout_neighbor is not None:
        return checkout_neighbor

    return _get_or_create_model_cache_dir() / PREDICT_MODEL_DIRNAME


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode())


def _extract_zenodo_record_id(spec: str) -> str | None:
    spec = spec.strip()
    if spec.isdigit():
        return spec

    for marker in ("/records/", "/api/records/", "zenodo."):
        if marker in spec:
            tail = spec.split(marker, 1)[1]
            digits = []
            for char in tail:
                if char.isdigit():
                    digits.append(char)
                else:
                    break
            if digits:
                return "".join(digits)
    return None


def _resolve_predict_model_source(logger: logging.Logger, zenodo_version: str | None = None) -> dict:
    if zenodo_version and zenodo_version.lower().startswith("http") and zenodo_version.lower().endswith(".zip"):
        return {
            "record_id": None,
            "conceptrecid": None,
            "version": None,
            "download_url": zenodo_version,
            "source_label": zenodo_version,
        }

    if zenodo_version is None:
        metadata_url = f"https://zenodo.org/api/records/{PREDICT_MODEL_DEFAULT_RECORD_ID}/versions/latest"
        logger.info("Resolving latest FlowCytoClassifier model on Zenodo")
    else:
        record_id = _extract_zenodo_record_id(zenodo_version)
        if record_id is None:
            raiseCytoError(
                f"Unable to parse Zenodo version '{zenodo_version}'. "
                "Use a record id such as '19709957', a Zenodo record URL, a DOI, or a direct zip URL.",
                logger,
            )
        metadata_url = f"https://zenodo.org/api/records/{record_id}"
        logger.info(f"Resolving FlowCytoClassifier model from Zenodo record {record_id}")

    try:
        payload = _fetch_json(metadata_url)
    except Exception as e:
        raiseCytoError(f"Failed to fetch Zenodo model metadata: {e}", logger)

    files = payload.get("files", [])
    matching_file = next((item for item in files if item.get("key") == PREDICT_MODEL_FILENAME), None)
    if matching_file is None and files:
        matching_file = next((item for item in files if str(item.get("key", "")).lower().endswith(".zip")), files[0])

    if matching_file is None:
        raiseCytoError("Zenodo record does not contain a downloadable model archive.", logger)

    download_url = matching_file.get("links", {}).get("self")
    if not download_url:
        raiseCytoError("Zenodo record did not expose a file download URL.", logger)

    metadata = payload.get("metadata", {})
    return {
        "record_id": str(payload.get("id")) if payload.get("id") is not None else None,
        "conceptrecid": payload.get("conceptrecid"),
        "version": metadata.get("version"),
        "download_url": download_url,
        "source_label": payload.get("links", {}).get("self_html") or metadata_url,
    }


def _read_installed_predict_model_metadata(install_dir: Path) -> dict | None:
    metadata_path = install_dir / PREDICT_MODEL_SOURCE_METADATA
    if not metadata_path.exists():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _write_installed_predict_model_metadata(install_dir: Path, metadata: dict) -> None:
    metadata_path = install_dir / PREDICT_MODEL_SOURCE_METADATA
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def ensure_predict_model_available(
    logger: logging.Logger,
    force: bool = False,
    zenodo_version: str | None = None,
    project: Path | None = None,
) -> Path:
    return _download_predict_model(
        logger=logger,
        force=force,
        zenodo_version=zenodo_version,
        project=project,
    )


def _get_executable_name() -> str:
    """Define the name of the cyz2json executable based on OS."""
    executable_name = "Cyz2Json.exe" if platform.system() == "Windows" else "Cyz2Json"
    return executable_name


def _get_preferred_executable_path() -> Path:
    """Return the executable path we should actually invoke on this platform."""
    bin_dir = _get_or_create_bin_dir()
    executable_name = _get_executable_name()

    if platform.system() == "Windows":
        # On Windows the executable depends on sibling DLLs in the extracted folder.
        # Running the copied launcher from bin/ can fail if those files are not next to it.
        return bin_dir / "cyz2json_dlls" / executable_name

    return bin_dir / executable_name


def _get_release_file_name(logger: logging.Logger) -> str:
    """Get the appropriate release file name based on OS."""
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        release_file = "cyz2json-macos-latest.zip"
    elif system == "linux":
        release_file = "cyz2json-ubuntu-latest.zip"
    elif system == "windows":
        release_file = "cyz2json-windows-latest.zip"
    else:
        raiseCytoError(f"Unsupported OS: {system}", logger)
    
    logger.debug(f"Determined release file name: {release_file}")
    return release_file


def _download_latest_release(logger: logging.Logger) -> str:
    """Download the latest release of cyz2json and return the path to the executable."""
    # 1. Fetch latest release info from GitHub API
    logger.info("Fetching latest cyz2json release info from GitHub")
    
    # get list of files in latest release
    api_url = "https://api.github.com/repos/OBAMANEXT/cyz2json/releases/latest"
    try:
        with urllib.request.urlopen(api_url) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        raiseCytoError(f"Failed to fetch latest release: {e}", logger)
    
    # search for the appropriate release file
    release_file = _get_release_file_name(logger)
    assets = data.get("assets", [])
    
    matching_asset = None
    for asset in assets:
        if release_file in asset["name"]:
            matching_asset = asset
            break
    
    if not matching_asset:
        available_assets = [a["name"] for a in assets]
        raiseCytoError(f"No file {release_file} within {available_assets}", logger)
    
    # 2. Download and extract the appropriate release file
    logger.info(f"Downloading and installing {matching_asset['name']}")
    
    download_url = matching_asset["browser_download_url"]
    logger.debug(f"Downloading from {download_url}")
    
    # we are actually downloading a bunch of files
    # determine where to store them
    bin_dir = _get_or_create_bin_dir()
    cyz2json_dir = bin_dir / "cyz2json_dlls"
    
    try:
        # download to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp_path = tmp.name
        urllib.request.urlretrieve(download_url, tmp_path)
        logger.debug(f"Downloaded to {tmp_path}")
        
        # clean up the existing cyz2json_dir if it exists
        if cyz2json_dir.exists():
            logger.debug(f"Removing existing cyz2json directory at {cyz2json_dir}")
            shutil.rmtree(cyz2json_dir)
                    
        # extract the zip file
        cyz2json_dir.mkdir(parents=True)
        logger.debug(f"Extracting to {cyz2json_dir}")
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            zip_ref.extractall(cyz2json_dir)
        
        # clean up temporary file
        logger.debug(f"Removing temporary file {tmp_path}")
        os.remove(tmp_path)
       
        # Define the extracted executable and the optional launcher path exposed to the rest of the app
        executable_path = cyz2json_dir / _get_executable_name()
        launcher_path = bin_dir / _get_executable_name()
        
        # make the executable actually executable
        logger.debug(f"Setting execute permissions for {executable_path}")
        os.chmod(executable_path, 0o755)

        # remove existing launcher if it exists
        if launcher_path.exists() or launcher_path.is_symlink():
            logger.debug(f"Removing existing launcher at {launcher_path}")
            launcher_path.unlink()

        if platform.system() == "Windows":
            # On Windows we invoke the executable directly from the extracted folder
            # so it can find its sibling DLLs.
            logger.debug(f"Using extracted executable directly at {executable_path}")
        else:
            logger.debug(f"Creating symlink at {launcher_path} -> {executable_path}")
            os.symlink(executable_path, launcher_path)
            logger.debug(f"Successfully installed cyz2json to {launcher_path}")
    
    except Exception as e:
        raiseCytoError(f"Failed to download and install cyz2json: {e}", logger)
    
    if platform.system() == "Windows":
        logger.debug(f"Successfully installed cyz2json to {executable_path}")
        return str(executable_path)

    return str(launcher_path)


def _download_file(url: str, logger: logging.Logger, suffix: str = ".zip") -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
    logger.debug(f"Downloading from {url} to temporary file {tmp_path}")
    urllib.request.urlretrieve(url, tmp_path)
    return tmp_path


def _find_extracted_model_root(extract_dir: Path) -> Path:
    direct_root = extract_dir / PREDICT_MODEL_DIRNAME
    if (direct_root / "models").exists():
        return direct_root

    if (extract_dir / "models").exists():
        return extract_dir

    children = [child for child in extract_dir.iterdir() if child.is_dir()]
    if len(children) == 1 and (children[0] / "models").exists():
        return children[0]

    raise FileNotFoundError(
        f"Could not find an extracted '{PREDICT_MODEL_DIRNAME}' model directory inside '{extract_dir}'."
    )


def _download_predict_model(
    logger: logging.Logger,
    force: bool = False,
    zenodo_version: str | None = None,
    project: Path | None = None,
) -> Path:
    install_dir = get_predict_model_install_dir(project=project)
    try:
        source = _resolve_predict_model_source(logger, zenodo_version=zenodo_version)
    except Exception:
        if install_dir.exists() and not force:
            logger.warning(
                f"Could not resolve the Zenodo model source right now; using the already installed model at '{install_dir}'"
            )
            return install_dir
        raise

    installed_metadata = _read_installed_predict_model_metadata(install_dir) if install_dir.exists() else None

    if install_dir.exists() and not force:
        if installed_metadata and installed_metadata.get("record_id") == source.get("record_id"):
            logger.info(f"Prediction model already installed at '{install_dir}'")
            return install_dir
        if source.get("record_id") is None and installed_metadata and installed_metadata.get("download_url") == source.get("download_url"):
            logger.info(f"Prediction model already installed at '{install_dir}'")
            return install_dir

    logger.info("Downloading FlowCytoClassifier prediction model")
    tmp_zip = _download_file(source["download_url"], logger, suffix=".zip")
    tmp_extract_dir = Path(tempfile.mkdtemp(prefix="cytoprocess-model-"))

    try:
        logger.debug(f"Extracting model archive to {tmp_extract_dir}")
        with zipfile.ZipFile(tmp_zip, "r") as zip_ref:
            zip_ref.extractall(tmp_extract_dir)

        extracted_root = _find_extracted_model_root(tmp_extract_dir)

        if install_dir.exists():
            logger.debug(f"Removing existing prediction model directory at {install_dir}")
            shutil.rmtree(install_dir)

        install_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted_root), str(install_dir))
        _write_installed_predict_model_metadata(install_dir, source)
        logger.info(f"Installed prediction model to '{install_dir}'")
        return install_dir
    except Exception as e:
        raiseCytoError(f"Failed to download and install prediction model: {e}", logger)
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)


def _check_or_get_cyz2json(logger: logging.Logger, force: bool = False) -> str:
    """Get the path to the cyz2json executable, downloading if necessary."""
    executable_path = _get_preferred_executable_path()

    if force:
        logger.info("Downloading latest cyz2json release")
        return _download_latest_release(logger)
    
    if not executable_path.exists():
        logger.info(f"cyz2json not found at {executable_path}, downloading")
        return _download_latest_release(logger)
    
    logger.debug(f"Using existing cyz2json at {executable_path}")
    return str(executable_path)


def run(
    ctx: click.Context,
    force: bool = False,
    predict_model: bool = False,
    zenodo_version: str | None = None,
    project: Path | None = None,
):
    logger = setup_logging(command="install", project=None, debug=ctx.obj["debug"])
    target_description = "Installing cyz2json and prediction model" if predict_model else "Installing cyz2json"
    log_command_start(logger, target_description, project=None)

    try:
        path = _check_or_get_cyz2json(force=force, logger=logger)
        result = subprocess.run([path, '--version'], check=True, capture_output=True, text=True)
        cyz2json_version = result.stdout.strip().removeprefix('Cyz2Json-')
        logger.info(f"cyz2json installed at {path}, at version {cyz2json_version}")

        if predict_model:
            model_dir = _download_predict_model(
                logger=logger,
                force=force,
                zenodo_version=zenodo_version,
                project=project,
            )
            logger.info(f"Prediction model ready at '{model_dir}'")
    except Exception as e:
        raiseCytoError(f"Failed to install dependencies: {e}", logger)
        raise

    log_command_success(logger, target_description)

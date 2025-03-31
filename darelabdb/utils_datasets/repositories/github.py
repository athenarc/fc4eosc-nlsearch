import glob
import logging
import os
import urllib.request
from typing import Any, Callable, List, Optional, Tuple


def get_files_from_github_repo(
    repo_id: str,
    branch: str,
    file_paths: List[str],
    cache_dir: str,
    post_processing: Optional[List[Tuple[Callable, List[Any]]]] = None,
) -> None:
    """
    Downloads files from a Github repo.

    Args:
        repo_id (str): The repository ID of the file to download (e.g., salesforce/WikiSQL).
        branch (str): The branch of the repository to download from.
        file_paths (List[str]): The paths of the files to download from the repo.
        cache_dir (str): The directory to download the file to.
        post_processing (Optional[List[Tuple[Callable, List[Any]]]]): A list of tuples containing a function and its
            arguments to be applied to the downloaded files (i.e. [(unzip_files, [cache/folder/, [file.zip]]).
    """
    folder_name = repo_id.split("/")[-1]
    cache_dir = cache_dir + "/" if cache_dir[-1] != "/" else cache_dir

    cached_datasets = list(map(os.path.basename, glob.glob(f"{cache_dir}*")))

    if folder_name in cached_datasets:  # Check if file is already downloaded in cache
        logging.debug(f"Using cached version of {folder_name}")
        return None

    os.makedirs(cache_dir + folder_name, exist_ok=True)

    for file_path in file_paths:
        urllib.request.urlretrieve(
            url=f"http://raw.githubusercontent.com/{repo_id}/{branch}/{file_path}",
            filename=f"{cache_dir}{folder_name}/{file_path}",
        )

    if post_processing is not None:
        for func, args in post_processing:
            func(*args)

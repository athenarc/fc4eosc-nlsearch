import glob
import logging
import os
from typing import Any, Callable, List, Optional, Tuple

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


def get_file_from_huggingface_hub(
    repo_id: str,
    cache_dir: str,
    branch: Optional[str] = None,
    post_processing: Optional[List[Tuple[Callable, List[Any]]]] = None,
) -> None:
    """
    Downloads a file from the HuggingFace Hub.

    Args:
        repo_id (str): The repository ID of the file to download (i.e. MikeXydas/iris).
        cache_dir (str): The directory to download the file to.
        branch (Optional[str]): The branch to download the file from.
        post_processing (Optional[List[Tuple[Callable, List[Any]]]]): A list of tuples containing a function and its
            arguments to be applied to the downloaded files (i.e. [(unzip_files, [cache/folder/, [file.zip]]).
    """

    load_dotenv()

    folder_name = repo_id.split("/")[-1]
    cache_dir = cache_dir + "/" if cache_dir[-1] != "/" else cache_dir

    cached_datasets = list(map(os.path.basename, glob.glob(f"{cache_dir}*")))

    if folder_name in cached_datasets:  # Check if file is already downloaded in cache
        logging.debug(f"Using cached version of {folder_name}")
        return None

    print(branch)

    os.makedirs(cache_dir + folder_name, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=branch,
        local_dir=cache_dir + folder_name,
        repo_type="dataset",
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", ".gitattributes"],
        token=os.getenv("huggingface_access_token"),
    )

    if post_processing is not None:
        for func, args in post_processing:
            func(*args)

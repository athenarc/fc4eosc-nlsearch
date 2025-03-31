import gzip
import os
import shutil
import tarfile
import zipfile
from typing import List


def unzip_files(root_folder: str, zipped_file_paths: List[str]):
    """
    Unzips the files in the given list of zipped_file_paths in the root directory given by `root_folder`.
    Depending on the file extension, the appropriate unzip function is called.
    Args:
        root_folder (str): The root folder where the zipped files are located.
        zipped_file_paths (List[str]): A list of paths to the zipped files.
    """
    for zipped_file_path in zipped_file_paths:
        if zipped_file_path.endswith(".zip"):
            unzip_zip_files(root_folder, [zipped_file_path])
        elif zipped_file_path.endswith(".gz"):
            unzip_gz_files(root_folder, [zipped_file_path])
        else:
            raise ValueError(
                f"Zipped file extension of file {zipped_file_path} not supported."
            )


def unzip_zip_files(root_folder: str, zipped_file_paths: List[str]):
    for zipped_file_path in zipped_file_paths:
        with zipfile.ZipFile(root_folder + zipped_file_path) as zip_ref:
            zip_ref.extractall(root_folder)


def unzip_gz_files(root_folder: str, zipped_file_paths: List[str]):
    for zipped_file_path in zipped_file_paths:
        file_name = (os.path.basename(zipped_file_path)).rsplit(".", 1)[
            0
        ]  # get file name for file within
        with gzip.open(root_folder + zipped_file_path, "rb") as f_in, open(
            root_folder + file_name, "wb"
        ) as f_out:
            shutil.copyfileobj(f_in, f_out)


def unzip_tarfile(root_folder: str, zipped_file_paths: List[str]):
    for zipped_file_path in zipped_file_paths:
        with tarfile.open(root_folder + zipped_file_path) as tar:
            tar.extractall(path=root_folder, filter="data")

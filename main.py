from tqdm import tqdm
from unitsyncer.sync import Synchronizer
from pylspclient.lsp_structs import LANGUAGE_IDENTIFIER, Location, Position, Range
from returns.maybe import Maybe, Nothing, Some
from unitsyncer.util import parallel_starmap as starmap, path2uri
from unitsyncer.source_code import get_function_code
import json
import jsonlines
import os
from pathos.multiprocessing import ProcessPool
import logging
import fire


def id2path(id):
    return id.split("::")[0]


def java_workdir_dict(test_ids: list[str]) -> dict[str, list[str]]:
    """split a list of test ids into a dict of workdir to file path
    this solves the LSP TimeoutError for JAVA with too much subdirectories

    Args:
        test_ids (list[str]): [path/to/test/file::test_func_name]

    Returns:
        dict[str, list[str]]: {workdir: [test_id, ...], ...}
    """
    workdir_dict = {}
    for test_id in test_ids:
        file_path = id2path(test_id)
        workdir = file_path.split("/test")[0]
        if workdir not in workdir_dict.keys():
            workdir_dict[workdir] = []
        workdir_dict[workdir].append(test_id)
    return workdir_dict


def focal2result(syncer: Synchronizer, repos_root, obj):
    p = id2path(obj["test_id"])
    if p[0] == "/":
        p = p[1:]
    file_path = os.path.join(repos_root, p)
    src_lineno, src_col_offset = obj["focal_loc"]
    test_lineno, test_col_offset = obj["test_loc"]

    langID = syncer.langID

    # only python ast is 1-indexed, tree-sitter and LSP are 0-indexed
    match langID:
        case LANGUAGE_IDENTIFIER.PYTHON:
            src_lineno -= 1
            test_lineno -= 1

    code, docstring = syncer.get_source_of_call(
        file_path, src_lineno, src_col_offset
    ).value_or((None, None))

    # since the test's delc node is already capture by frontend, it can store the test code
    if "test" in obj.keys():
        test = obj["test"]
    else:
        fake_loc = Location(
            path2uri(file_path),
            Range(
                Position(test_lineno, test_col_offset),
                Position(test_lineno, test_col_offset + 1),
            ),
        )
        test, _ = get_function_code(fake_loc, syncer.langID).value_or((None, None))

    if "focal_id" in obj.keys():
        code_id = obj["focal_id"]
    else:
        code_id = None

    return {
        "test_id": obj["test_id"],
        "test": test,
        "code_id": code_id,
        "code": code,
        "docstring": docstring,
    }


def process_one_focal_file(
    repos_root="data/repos",
    focal_file="./data/focal/ageitgey-face_recognition.jsonl",
    language="python",
):
    with open(focal_file) as f:
        objs = [json.loads(line) for line in f.readlines()]
        test_ids = [obj["test_id"] for obj in objs]

    match language:
        case LANGUAGE_IDENTIFIER.PYTHON:
            workdir = "/".join(id2path(test_ids[0]).split("/")[:2])
            wd = {
                workdir: None,
            }
        case LANGUAGE_IDENTIFIER.JAVA:
            wd = java_workdir_dict(test_ids)
        case _:
            return 1
    results = []
    logging.info(f"number of workdir_dict: {len(wd.keys())}")
    repos_root = os.path.abspath(repos_root)
    for workdir, _ in tqdm(wd.items()):
        if workdir[0] == "/":
            workdir = workdir[1:]
        full_workdir = os.path.join(repos_root, workdir)
        logging.info(f"workdir: {full_workdir}")

        syncer = Synchronizer(full_workdir, language)
        syncer.start_lsp_server()
        syncer.initialize()

        results += [focal2result(syncer, repos_root, obj) for obj in objs]

        syncer.stop()

    with jsonlines.open(focal_file.replace("focal", "source"), "w") as f:
        f.write_all(results)


def main(repos_root="data/repos", focal_dir="data/focal", language="python", jobs=4):
    all_focal_files = []
    for root, dirs, files in os.walk(os.path.abspath(focal_dir)):
        for file in files:
            if file.endswith(".jsonl"):
                all_focal_files.append(os.path.join(root, file))

    logging.info(f"Processing {len(all_focal_files)} focal files")
    os.makedirs("./data/source", exist_ok=True)
    with ProcessPool(jobs) as pool:
        pool.map(process_one_focal_file, all_focal_files, language)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)

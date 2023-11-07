import logging
from typing import Optional
import fire
import os
from frontend.util import wrap_repo, parallel_subprocess
import subprocess
from os.path import join as pjoin, basename, splitext as psplitext, abspath


def transform_repos(repos: list[str], jobs: int):
    def transform_one_repo(repo_path: str):
        return subprocess.Popen(
            ["rust-fuzzer-gen", repo_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    logging.info(f"Running rust-fuzz-gen on {len(repos)} repos")
    parallel_subprocess(repos, jobs, transform_one_repo, on_exit=None)


def get_target_list(p: subprocess.Popen):
    match p.stdout:
        case None:
            return []
        case _:
            return p.stdout.read().decode("utf-8").split("\n")


def fuzz_one_target(target: tuple[str, str], timeout):
    repo_path, target_name = target
    with open(pjoin(repo_path, "fuzz_inputs", target_name), "w") as f:
        return subprocess.Popen(
            # todo: find out why -max_total_time doesn't work
            # ["cargo", "fuzz", "run", target_name, "--", f"-max_total_time={timeout}"],
            [
                "bash",
                "-c",
                f"timeout {timeout} cargo fuzz run {target_name}",
            ],
            cwd=repo_path,
            stdout=f,
            stderr=subprocess.DEVNULL,
        )


def fuzz_repos(repos: list[str], jobs: int, timeout: int = 60):
    logging.info(f"Initializing fuzzing targets in {len(repos)} repos")
    parallel_subprocess(
        repos,
        jobs,
        lambda path: subprocess.Popen(
            ["cargo", "fuzz", "init"],
            cwd=path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ),
        on_exit=None,
    )
    logging.info(f"Building fuzzing targets in {len(repos)} repos")
    parallel_subprocess(
        repos,
        jobs,
        lambda path: subprocess.Popen(
            ["cargo", "fuzz", "build"],
            cwd=path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ),
        on_exit=None,
    )

    logging.info("Collecting all fuzz targets")

    target_map = parallel_subprocess(
        repos,
        jobs,
        lambda path: subprocess.Popen(
            ["cargo", "fuzz", "list"], cwd=path, stdout=subprocess.PIPE
        ),
        on_exit=get_target_list,
    )
    targets: list[tuple[str, str]] = [
        (k, v) for k, vs in target_map.items() for v in vs if len(v) > 0
    ]
    for repo in repos:
        os.makedirs(pjoin(repo, "fuzz_inputs"), exist_ok=True)

    logging.info(f"Running cargo fuzz on {len(targets)} targets for {timeout} seconds")
    parallel_subprocess(
        targets, jobs, lambda p: fuzz_one_target(p, timeout), on_exit=None
    )


def test_gen_repos(repos: list[str], jobs: int):
    pass


def main(
    repo_id: str = "marshallpierce/rust-base64",
    repo_root: str = "data/rust_repos/",
    timeout: int = 60,
    nprocs: int = 0,
    limits: Optional[int] = None,
    pipeline: str = "transform",
):
    try:
        repo_id_list = [
            ll for l in open(repo_id, "r").readlines() if len(ll := l.strip()) > 0
        ]
    except:
        repo_id_list = [repo_id]
    if limits is not None:
        repo_id_list = repo_id_list[:limits]
    logging.info(f"Loaded {len(repo_id_list)} repos to be processed")

    logging.info(f"Collecting all rust repos")
    repos = []
    for repo_id in repo_id_list:
        repo_path = os.path.join(repo_root, wrap_repo(repo_id))
        if os.path.exists(repo_path) and os.path.isdir(repo_path):
            subdirectories = [
                os.path.join(repo_path, d)
                for d in os.listdir(repo_path)
                if os.path.isdir(os.path.join(repo_path, d))
            ]
            repos.append(abspath(subdirectories[0]))

    match pipeline:
        case "transform":
            transform_repos(repos, nprocs)
        case "fuzz":
            fuzz_repos(repos, nprocs, timeout=timeout)
        case "testgen":
            test_gen_repos(repos, nprocs)
        case _:
            logging.error(f"Unknown pipeline {pipeline}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)

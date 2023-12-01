import frontend.python.collect_test as collect_test
import frontend.python.collect_focal as collect_focal
import frontend.python.collect_focal_org as collect_focal_org
import fire


def main(
    repo_id: str = "ageitgey/face_recognition",
    test_root: str = "data/tests",
    repo_root: str = "data/repos",
    focal_root: str = "data/focal",
    timeout: int = 300,
    nprocs: int = 0,
    limits: int = -1,
):
    collect_test.main(
        repo_id=repo_id,
        test_root=test_root,
        repo_root=repo_root,
        timeout=timeout,
        nprocs=nprocs,
        limits=limits,
    )
    collect_focal_org.main(
        repo_id_list=repo_id,
        test_root=test_root,
        repo_root=repo_root,
        focal_root=focal_root,
        timeout=timeout,
        nprocs=nprocs,
        limits=limits,
    )


if __name__ == "__main__":
    fire.Fire(main)

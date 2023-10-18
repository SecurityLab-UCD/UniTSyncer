import ast
from typing import Iterable
import fire
import os
from pathlib import Path
from frontend.python.utils import wrap_repo
from tree_sitter.binding import Node
from frontend.parser.langauges import JAVA_LANGUAGE
from frontend.parser.ast_util import ASTUtil
from returns.maybe import Maybe, Nothing, Some


def has_test(file_path):
    # follow TeCo to check for JUnit4 and JUnit5
    # todo: support different usage as in google/closure-compiler
    def has_junit4(code):
        return "@Test" in code and "import org.junit.Test" in code

    def has_junit5(code):
        return "@Test" in code and "import org.junit.jupiter.api.Test" in code

    with open(file_path, "r") as f:
        code = f.read()
    return has_junit4(code) or has_junit5(code)


def collect_test_files(root: str):
    """Get all files end with .java in the given root directory

    Args:
        root (str): path to repo root
    """
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".java"):
                if has_test(p := os.path.join(dirpath, filename)):
                    yield p


def collect_test_funcs(file_path: str) -> Iterable[Maybe[str]]:
    """collect testing functions from the target file"""

    with open(file_path, "r") as f:
        ast_util = ASTUtil(f.read())

    tree = ast_util.tree(JAVA_LANGUAGE)
    root_node = tree.root_node

    decls = ast_util.get_all_nodes_of_type(root_node, "method_declaration")

    def has_test_modifier(node: Node):
        modifiers = ast_util.get_method_modifiers(node)
        return modifiers.map(lambda x: "@Test" in x).value_or(False)

    test_funcs = map(ast_util.get_method_name, filter(has_test_modifier, decls))
    return test_funcs


def collect_from_repo(repo_id: str, repo_root: str, test_root: str):
    """collect all test functions in the given project
    return (status, nfile, ntest)
    status can be 0: success, 1: repo not found, 2: test not found, 3: skip when output file existed
    """
    repo_path = os.path.join(repo_root, wrap_repo(repo_id))
    if not os.path.exists(repo_path) or not os.path.isdir(repo_path):
        return 1, 0, 0
    test_path = os.path.join(test_root, wrap_repo(repo_id) + ".txt")
    # skip if exist
    if os.path.exists(test_path):
        return 3, 0, 0
    # collect potential testing modules
    all_files = collect_test_files(repo_path)
    tests = {}
    for f in all_files:
        funcs = collect_test_funcs(f)
        tests[f] = funcs
    if len(tests.keys()) == 0:
        return 2, len(tests.keys()), sum(len(list(v)) for v in tests.values())
    # save to disk
    n_test_func = 0
    with open(test_path, "w") as outfile:
        for k in tests.keys():
            for v in tests[k]:
                match v:
                    case Some(func_name):
                        outfile.write(f"{k}::{func_name}\n")
                        n_test_func += 1
                    case Nothing:
                        continue

    return 0, len(tests.keys()), n_test_func


def main(
    repo_id_list: str = "spring-cloud/spring-cloud-netflix",
    repo_root: str = "data/repos/",
    test_root: str = "data/tests/",
    timeout: int = 120,
    nprocs: int = 0,
    limits: int = -1,
):
    status, nfile, ntest = collect_from_repo(
        repo_id_list,
        repo_root,
        test_root,
    )
    print(status, nfile, ntest)


if __name__ == "__main__":
    fire.Fire(main)

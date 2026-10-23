import subprocess
import sys

from auditcoder.common.callable_execution import (
    build_callable_driver,
    callable_arguments,
)


def test_callable_arguments_preserves_quoted_version_strings():
    assert callable_arguments(['"1.01"', '"1.001"']) == ["1.01", "1.001"]


def test_callable_driver_invokes_solution_method_without_running_main_adapter():
    source = """
class Solution:
    def merge(self, intervals):
        return intervals[:1]

if __name__ == "__main__":
    print("wrong adapter")
"""
    driver = build_callable_driver(source, "merge", [[[1, 3], [2, 6]]])
    completed = subprocess.run(
        [sys.executable, "-c", driver], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "[[1,3]]"

from typing import *
import contextlib
import signal

from .executor_utils import function_with_timeout


def evaluate_io(
    sample_io: list[str],
    completion: str,
    timeout: int = 5,
    stop_early: bool = False,
):
    if len(sample_io) == 0:
        return True, ""

    test_log = ""
    passed = True
    for io in sample_io:
        try:
            code = ("from typing import *\n" if "from typing import *" not in completion else "") + \
                completion + "\n" + io + "\n"
            function_with_timeout(
                exec,
                (code, globals()),
                timeout
            )
            test_log += f"Passed in test case: {io}\n"
        except Exception as e:
            if stop_early:
                return False, f"Failed in test case: {io}\n"
            passed = False
            test_log += f"Failed in test case: {io}\n"

    return passed, test_log


def evaluate_io_et(
    sample_io: list[str],
    completion: str,
    timeout: int = 5,
    prompt: str = "",
):
    io = "\n".join(sample_io)
    try:
        code = ("from typing import *\n" if "from typing import *" not in completion else "") + \
            prompt + completion + "\n" + io + "\n"
        function_with_timeout(
            exec,
            (code, globals()),
            timeout
        )
        return True
    except Exception as e:
        return False


def evaluate_functional_correctness(
    test: str,
    entry_point: str,
    completion: str,
    timeout: int = 5,
):
    try:
        code = ("from typing import *\n" if "from typing import *" not in completion else "") + \
            completion + "\n" + test + \
            "\n" + f"check({entry_point})"

        function_with_timeout(
            exec,
            (code, globals()),
            timeout
        )
        return "passed"
    except Exception as e:
        return f"failed: {e}"


class TimeoutException(Exception):
    pass

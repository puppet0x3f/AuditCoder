"""Deterministic output comparison for ordinary and special-judge tasks.

Competitive-programming judges normally ignore whitespace between output
tokens.  Some tasks additionally accept many different witnesses, so comparing
against one reference output is not sound.  This module keeps those two cases
explicit and returns machine-readable judging evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from itertools import product
import math
from typing import Any


Judge = Callable[[str, str, str], tuple[bool, str]]


def _token_equal(_input_data: str, expected: str, actual: str) -> tuple[bool, str]:
    passed = actual.split() == expected.split()
    return passed, "whitespace-insensitive token comparison"


def _judge_cf_1118e(
    input_data: str,
    _expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate any legal construction for Codeforces 1118E."""
    try:
        n, k = map(int, input_data.split())
    except (TypeError, ValueError):
        return False, "invalid test input for CF 1118E"

    lines = [line.strip() for line in actual.splitlines() if line.strip()]
    if not lines:
        return False, "empty output"

    feasible = n <= k * (k - 1)
    verdict = lines[0].upper()
    if not feasible:
        return verdict == "NO", "infeasible instance requires NO"
    if verdict != "YES":
        return False, "feasible instance requires YES"
    if len(lines) != n + 1:
        return False, f"expected {n} witness pairs, got {len(lines) - 1}"

    try:
        pairs = [tuple(map(int, line.split())) for line in lines[1:]]
    except ValueError:
        return False, "witness contains a non-integer pair"
    if any(len(pair) != 2 for pair in pairs):
        return False, "each witness row must contain exactly two colors"
    if any(not (1 <= a <= k and 1 <= b <= k and a != b) for a, b in pairs):
        return False, "color range or within-pair inequality violated"
    if len(set(pairs)) != n:
        return False, "witness pairs are not unique"
    if any(
        pairs[i][0] == pairs[i + 1][0] or pairs[i][1] == pairs[i + 1][1]
        for i in range(n - 1)
    ):
        return False, "adjacent pairs repeat a man or woman color"
    return True, "valid CF 1118E construction"


def _judge_kattis_bookingaroom(
    input_data: str,
    _expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Accept any available room for Kattis bookingaroom."""
    try:
        values = list(map(int, input_data.split()))
        room_count, booked_count = values[:2]
        booked = set(values[2 : 2 + booked_count])
    except (TypeError, ValueError):
        return False, "invalid test input for bookingaroom"

    answer = actual.strip()
    available = set(range(1, room_count + 1)) - booked
    if not available:
        return answer == "too late", "all rooms are booked"
    try:
        room = int(answer)
    except ValueError:
        return False, "output is not a room number"
    if room not in available:
        return False, "reported room is unavailable or out of range"
    return True, "reported room is available"


def _judge_cf_1256e(
    input_data: str,
    expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate any optimal team assignment for Codeforces 1256E."""
    try:
        input_values = list(map(int, input_data.split()))
        n = input_values[0]
        skills = input_values[1 : 1 + n]
        expected_cost = int(expected.split()[0])
        output_values = list(map(int, actual.split()))
    except (IndexError, TypeError, ValueError):
        return False, "input or output is not a valid integer sequence"

    if len(skills) != n or len(output_values) != n + 2:
        return False, "wrong number of skills or output tokens"
    reported_cost, team_count = output_values[:2]
    assignments = output_values[2:]
    if reported_cost != expected_cost:
        return False, "reported diversity is not the known optimum"
    if not 1 <= team_count <= n // 3:
        return False, "invalid team count"
    if set(assignments) != set(range(1, team_count + 1)):
        return False, "team identifiers must be consecutive and all used"

    computed_cost = 0
    for team_id in range(1, team_count + 1):
        members = [
            skills[index]
            for index, assigned_team in enumerate(assignments)
            if assigned_team == team_id
        ]
        if len(members) < 3:
            return False, "a team contains fewer than three students"
        computed_cost += max(members) - min(members)
    if computed_cost != reported_cost:
        return False, "assignment diversity does not match the reported value"
    return True, "valid optimal CF 1256E assignment"


def _judge_cf_1108e1(
    input_data: str,
    expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate any optimal interval selection for Codeforces 1108E1."""
    try:
        input_values = list(map(int, input_data.split()))
        n, m = input_values[:2]
        values = input_values[2 : 2 + n]
        raw_segments = input_values[2 + n :]
        segments = [
            (raw_segments[2 * index] - 1, raw_segments[2 * index + 1] - 1)
            for index in range(m)
        ]
        output_values = list(map(int, actual.split()))
        known_optimum = int(expected.split()[0])
    except (IndexError, TypeError, ValueError):
        return False, "input or output is not a valid integer sequence"

    if len(values) != n or len(segments) != m or len(output_values) < 2:
        return False, "wrong number of input or output values"
    reported_difference, selected_count = output_values[:2]
    selected = output_values[2:]
    if selected_count != len(selected):
        return False, "reported interval count does not match the witness"
    if len(set(selected)) != len(selected):
        return False, "selected interval indices are not unique"
    if any(not 1 <= index <= m for index in selected):
        return False, "selected interval index is out of range"

    transformed = values[:]
    for index in selected:
        left, right = segments[index - 1]
        for position in range(left, right + 1):
            transformed[position] -= 1
    computed_difference = max(transformed) - min(transformed)
    if computed_difference != reported_difference:
        return False, "selected intervals do not produce the reported difference"
    if reported_difference != known_optimum:
        return False, "reported difference is not the known optimum"
    return True, "valid optimal CF 1108E1 interval selection"


def _judge_cf_1006b(
    input_data: str,
    _expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate any optimal partition for Codeforces 1006B."""
    try:
        input_values = list(map(int, input_data.split()))
        n, k = input_values[:2]
        values = input_values[2 : 2 + n]
        output_values = list(map(int, actual.split()))
    except (IndexError, TypeError, ValueError):
        return False, "input or output is not a valid integer sequence"

    if len(values) != n or len(output_values) != k + 1:
        return False, "output must contain the profit and exactly k lengths"
    reported_profit = output_values[0]
    lengths = output_values[1:]
    if any(length <= 0 for length in lengths) or sum(lengths) != n:
        return False, "partition lengths must be positive and sum to n"

    offset = 0
    computed_profit = 0
    for length in lengths:
        computed_profit += max(values[offset : offset + length])
        offset += length
    optimum = sum(sorted(values, reverse=True)[:k])
    if computed_profit != reported_profit:
        return False, "partition does not produce the reported profit"
    if reported_profit != optimum:
        return False, "reported profit is not optimal"
    return True, "valid optimal CF 1006B partition"


def _judge_cf_1154a(
    input_data: str,
    _expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Accept any ordering of the reconstructed integers for Codeforces 1154A."""
    try:
        board_values = sorted(map(int, input_data.split()))
        originals = list(map(int, actual.split()))
    except (TypeError, ValueError):
        return False, "input or output is not a valid integer sequence"
    if len(board_values) != 4 or len(originals) != 3:
        return False, "expected four board values and three reconstructed integers"
    if any(value <= 0 for value in originals):
        return False, "reconstructed integers must be positive"
    a, b, c = originals
    reconstructed = sorted((a + b, a + c, b + c, a + b + c))
    if reconstructed != board_values:
        return False, "output does not reconstruct the four board values"
    return True, "valid CF 1154A reconstruction in an arbitrary order"


def _judge_cf_1433d(
    input_data: str,
    _expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate any legal spanning-tree construction for Codeforces 1433D."""
    try:
        input_tokens = input_data.split()
        output_tokens = actual.split()
        input_index = 0
        output_index = 0
        test_count = int(input_tokens[input_index])
        input_index += 1

        for _ in range(test_count):
            n = int(input_tokens[input_index])
            input_index += 1
            gangs = list(map(int, input_tokens[input_index : input_index + n]))
            input_index += n
            if output_index >= len(output_tokens):
                return False, "missing YES or NO verdict"
            verdict = output_tokens[output_index].upper()
            output_index += 1
            feasible = len(set(gangs)) > 1
            if not feasible:
                if verdict != "NO":
                    return False, "single-gang instance requires NO"
                continue
            if verdict != "YES":
                return False, "multi-gang instance requires YES"

            edges: list[tuple[int, int]] = []
            for _ in range(n - 1):
                if output_index + 1 >= len(output_tokens):
                    return False, "missing spanning-tree edge"
                left = int(output_tokens[output_index])
                right = int(output_tokens[output_index + 1])
                output_index += 2
                edges.append((left, right))

            parent = list(range(n))

            def find(node: int) -> int:
                while parent[node] != node:
                    parent[node] = parent[parent[node]]
                    node = parent[node]
                return node

            for left, right in edges:
                if not (1 <= left <= n and 1 <= right <= n):
                    return False, "edge endpoint is out of range"
                if gangs[left - 1] == gangs[right - 1]:
                    return False, "edge joins districts from the same gang"
                root_left = find(left - 1)
                root_right = find(right - 1)
                if root_left == root_right:
                    return False, "edges contain a cycle"
                parent[root_left] = root_right
            if len({find(node) for node in range(n)}) != 1:
                return False, "edges do not connect all districts"
    except (IndexError, TypeError, ValueError):
        return False, "input or output has an invalid construction format"

    if input_index != len(input_tokens) or output_index != len(output_tokens):
        return False, "unexpected extra input or output tokens"
    return True, "valid CF 1433D spanning-tree construction"


def _judge_kattis_calculatingdartscores(
    input_data: str,
    _expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Accept any sequence of at most three dart throws with the target score."""
    try:
        target = int(input_data.strip())
    except (TypeError, ValueError):
        return False, "invalid target score"

    stripped = actual.strip()
    scores = [
        multiplier * section
        for multiplier in (1, 2, 3)
        for section in range(1, 21)
    ]
    if stripped.lower() == "impossible":
        feasible = any(
            sum(throws) == target
            for dart_count in (1, 2, 3)
            for throws in product(scores, repeat=dart_count)
        )
        return (not feasible), (
            "target is unreachable" if not feasible else "target has a legal witness"
        )

    lines = [line.strip() for line in actual.splitlines() if line.strip()]
    if not 1 <= len(lines) <= 3:
        return False, "witness must contain between one and three darts"
    multiplier_by_name = {"single": 1, "double": 2, "triple": 3}
    total = 0
    for line in lines:
        parts = line.lower().split()
        if len(parts) != 2 or parts[0] not in multiplier_by_name:
            return False, "invalid dart description"
        try:
            section = int(parts[1])
        except ValueError:
            return False, "dart section is not an integer"
        if not 1 <= section <= 20:
            return False, "dart section is out of range"
        total += multiplier_by_name[parts[0]] * section
    if total != target:
        return False, "dart witness does not sum to the target"
    return True, "valid Kattis calculatingdartscores witness"


def _judge_kattis_bestrelayteam(
    input_data: str,
    _expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate any optimal relay lineup with tolerant numeric comparison."""
    try:
        tokens = input_data.split()
        n = int(tokens[0])
        runners: dict[str, tuple[Decimal, Decimal]] = {}
        offset = 1
        for _ in range(n):
            name = tokens[offset]
            runners[name] = (
                Decimal(tokens[offset + 1]),
                Decimal(tokens[offset + 2]),
            )
            offset += 3
        lines = [line.strip() for line in actual.splitlines() if line.strip()]
        reported = Decimal(lines[0])
        lineup = lines[1:]
    except (IndexError, InvalidOperation, TypeError, ValueError):
        return False, "input or output has an invalid relay format"

    if len(runners) != n or len(lineup) != 4 or len(set(lineup)) != 4:
        return False, "lineup must contain four distinct known runners"
    if any(name not in runners for name in lineup):
        return False, "lineup contains an unknown runner"
    computed = runners[lineup[0]][0] + sum(
        (runners[name][1] for name in lineup[1:]),
        Decimal(0),
    )
    optimum = min(
        first_time
        + sum(
            sorted(
                (
                    other_times[1]
                    for other_name, other_times in runners.items()
                    if other_name != first_name
                )
            )[:3],
            Decimal(0),
        )
        for first_name, (first_time, _other_time) in runners.items()
    )
    tolerance = Decimal("0.000001")
    if abs(reported - computed) > tolerance:
        return False, "reported time does not match the lineup"
    if abs(computed - optimum) > tolerance:
        return False, "lineup is not optimal"
    return True, "valid optimal Kattis bestrelayteam lineup"


def _judge_single_float(
    _input_data: str,
    expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Compare a single floating-point answer using common judge tolerances."""
    try:
        expected_tokens = expected.split()
        actual_tokens = actual.split()
        if len(expected_tokens) != 1 or len(actual_tokens) != 1:
            return False, "expected exactly one numeric output token"
        expected_value = float(expected_tokens[0])
        actual_value = float(actual_tokens[0])
    except (TypeError, ValueError):
        return False, "output is not a floating-point number"
    if not math.isfinite(expected_value) or not math.isfinite(actual_value):
        return False, "non-finite floating-point output"
    passed = math.isclose(
        actual_value,
        expected_value,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    return passed, "single-float comparison with abs/rel tolerance 1e-6"


_SEMANTIC_JUDGES: dict[int, Judge] = {
    4078: _judge_cf_1108e1,
    4091: _judge_cf_1006b,
    4105: _judge_cf_1118e,
    4377: _judge_cf_1154a,
    4451: _judge_cf_1256e,
    4518: _judge_cf_1433d,
    4763: _judge_kattis_calculatingdartscores,
    4897: _judge_kattis_bestrelayteam,
    4966: _judge_single_float,
    4974: _judge_kattis_bookingaroom,
}


def judge_output(
    *,
    problem_id: int | None,
    input_data: str,
    expected: str,
    actual: str,
) -> dict[str, Any]:
    """Compare one output and return a stable audit record."""
    judge = _SEMANTIC_JUDGES.get(problem_id) if problem_id is not None else None
    mode = "semantic_checker" if judge is not None else "token"
    try:
        passed, detail = (judge or _token_equal)(input_data, expected, actual)
    except Exception as exc:  # A checker failure is evidence, never a silent pass.
        return {
            "passed": False,
            "judge_mode": mode,
            "judge_detail": f"checker_error: {type(exc).__name__}: {exc}",
        }
    return {
        "passed": bool(passed),
        "judge_mode": mode,
        "judge_detail": detail,
    }

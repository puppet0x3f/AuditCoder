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
import json
import math
from typing import Any


Judge = Callable[[str, str, str], tuple[bool, str]]


def _token_equal(_input_data: str, expected: str, actual: str) -> tuple[bool, str]:
    passed = actual.split() == expected.split()
    return passed, "whitespace-insensitive token comparison"


def _judge_json(_input_data: str, expected: str, actual: str) -> tuple[bool, str]:
    """Compare function-style results structurally, ignoring JSON spacing."""
    try:
        expected_value = json.loads(expected.strip())
        actual_value = json.loads(actual.strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, "function result is not valid JSON"
    return expected_value == actual_value, "structural JSON result comparison"


def _judge_cf_1248d1(
    input_data: str,
    expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate any optimal swap for CF 1248D1.

    APPS stores one optimal pair, but the problem accepts every pair that
    reaches the same beauty.  In particular, swapping the same two positions
    in the opposite order is equivalent.
    """
    values = input_data.split()
    if len(values) < 2:
        return False, "invalid CF 1248D1 input"
    try:
        n = int(values[0])
        text = values[1]
        optimum = int(expected.split()[0])
        tokens = actual.split()
        reported, left, right = map(int, tokens[:3])
    except (IndexError, TypeError, ValueError):
        return False, "output must contain beauty and two swap positions"
    if len(text) != n or len(tokens) != 3:
        return False, "wrong input or output length"
    if not (1 <= left <= n and 1 <= right <= n):
        return False, "swap position is out of range"
    chars = list(text)
    chars[left - 1], chars[right - 1] = chars[right - 1], chars[left - 1]
    swapped = "".join(chars)
    if swapped.count("(") != swapped.count(")"):
        beauty = 0
    else:
        doubled = swapped + swapped
        beauty = 0
        for start in range(n):
            balance = 0
            good = True
            for char in doubled[start : start + n]:
                balance += 1 if char == "(" else -1
                if balance < 0:
                    good = False
                    break
            if good and balance == 0:
                beauty += 1
    if reported != optimum:
        return False, "reported beauty is not the known optimum"
    if beauty != reported:
        return False, "the reported swap does not attain its beauty"
    return True, "valid optimal CF 1248D1 swap"


def _judge_cf_1042c(
    input_data: str,
    _expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate any optimal operation sequence for CF 1042C."""
    try:
        values = list(map(int, input_data.split()))
        n = values[0]
        initial = values[1 : 1 + n]
    except (IndexError, TypeError, ValueError):
        return False, "invalid CF 1042C input"
    if len(initial) != n:
        return False, "input length does not match n"
    lines = [line.split() for line in actual.splitlines() if line.strip()]
    if len(lines) != n - 1:
        return False, f"expected {n - 1} operations, got {len(lines)}"
    active = {index + 1: value for index, value in enumerate(initial)}
    removed_once = False
    try:
        for operation in lines:
            kind = int(operation[0])
            if kind == 1:
                if len(operation) != 3:
                    return False, "type-1 operation has the wrong arity"
                left, right = int(operation[1]), int(operation[2])
                if left == right or left not in active or right not in active:
                    return False, "type-1 operation uses an inactive or equal index"
                active[right] = active[left] * active[right]
                del active[left]
            elif kind == 2:
                if len(operation) != 2:
                    return False, "type-2 operation has the wrong arity"
                index = int(operation[1])
                if removed_once or index not in active:
                    return False, "invalid or repeated type-2 removal"
                removed_once = True
                del active[index]
            else:
                return False, "unknown operation type"
    except (IndexError, TypeError, ValueError):
        return False, "operation contains a non-integer token"
    if len(active) != 1:
        return False, "operation sequence does not leave exactly one value"

    nonzero = [value for value in initial if value != 0]
    if not nonzero:
        optimum = 0
    else:
        product = 1
        for value in nonzero:
            product *= value
        if product > 0:
            optimum = product
        elif len(nonzero) == 1:
            optimum = 0 if len(nonzero) != n else product
        else:
            # Zeros and the unwanted negative can be merged into one group
            # and removed by the single type-2 operation.  Their presence
            # does not force the best result to be zero.
            closest_to_zero = max(value for value in nonzero if value < 0)
            optimum = product // closest_to_zero
    result = next(iter(active.values()))
    if result != optimum:
        return False, "operation sequence does not attain the maximum remaining value"
    return True, "valid optimal CF 1042C operation sequence"


def _judge_cf_1189a(
    input_data: str,
    _expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate any minimum good-substring partition for CF 1189A."""
    values = input_data.split()
    if len(values) < 2:
        return False, "invalid CF 1189A input"
    try:
        n = int(values[0])
        text = values[1]
        tokens = actual.split()
        count = int(tokens[0])
    except (IndexError, TypeError, ValueError):
        return False, "invalid partition output"
    parts = tokens[1:]
    if len(text) != n or len(parts) != count or count < 1:
        return False, "partition length does not match the witness"
    if "".join(parts) != text:
        return False, "parts do not concatenate to the input string"
    good = lambda part: part.count("0") != part.count("1")
    if any(not good(part) for part in parts):
        return False, "a partition part is not good"
    whole_good = good(text)
    minimum = 1 if whole_good else (n if len(set(text)) == 1 else 2)
    if count != minimum:
        return False, "partition is not minimum"
    return True, "valid minimum CF 1189A partition"


def _judge_cf_1395b(
    input_data: str,
    _expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate any rook Hamiltonian path for CF 1395B."""
    try:
        n, m, start_x, start_y = map(int, input_data.split())
        values = list(map(int, actual.split()))
    except (TypeError, ValueError):
        return False, "invalid CF 1395B input or output"
    if len(values) != 2 * n * m:
        return False, "wrong number of grid cells"
    path = list(zip(values[::2], values[1::2]))
    if path[0] != (start_x, start_y):
        return False, "path does not start at the requested cell"
    if len(set(path)) != n * m:
        return False, "path repeats a cell"
    if any(not (1 <= x <= n and 1 <= y <= m) for x, y in path):
        return False, "path contains an out-of-range cell"
    if any(x1 != x2 and y1 != y2 for (x1, y1), (x2, y2) in zip(path, path[1:])):
        return False, "adjacent cells are not connected by a rook move"
    return True, "valid CF 1395B rook Hamiltonian path"


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


def _judge_cf_780d(
    input_data: str,
    expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate any legal distinct short-name assignment for CF 780D."""
    try:
        values = input_data.split()
        n = int(values[0])
        clubs = [(values[1 + 2 * i], values[2 + 2 * i]) for i in range(n)]
    except (IndexError, TypeError, ValueError):
        return False, "invalid CF 780D input"

    lines = [line.strip() for line in actual.splitlines() if line.strip()]
    expected_first = expected.strip().split()[0].upper() if expected.strip() else ""
    if not lines:
        return False, "empty output"
    verdict = lines[0].upper()
    if verdict == "NO":
        return (expected_first == "NO"), "NO agrees with the reference feasibility result"
    if verdict != "YES" or expected_first == "NO" or len(lines) != n + 1:
        return False, "wrong feasibility verdict or witness length"

    options: list[tuple[str, str]] = [
        (team[:3], team[:2] + town[0]) for team, town in clubs
    ]
    chosen = lines[1:]
    if any(len(name) != 3 for name in chosen):
        return False, "short names must contain exactly three letters"
    if any(name not in options[i] for i, name in enumerate(chosen)):
        return False, "short name is not one of the two allowed options"
    if len(set(chosen)) != n:
        return False, "chosen short names are not distinct"
    # The directional rule is about which option was selected, not merely the
    # text.  A club with identical A/B options is one neutral formal choice.
    chose_first = [
        options[j][0] != options[j][1] and chosen[j] == options[j][0]
        for j in range(n)
    ]
    for i, name in enumerate(chosen):
        first, second = options[i]
        if second != first and name == second:
            if any(
                j != i and chose_first[j] and options[j][0] == first
                for j in range(n)
            ):
                return False, "second-option rule is violated"
    return True, "valid CF 780D construction"


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


def _judge_float_sequence(
    _input_data: str,
    expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Compare a whitespace-separated sequence of real values.

    Several APPS tasks, including Codeforces' ``Curling`` (problem 2148 in
    the sampled set), explicitly accept an absolute/relative error for every
    coordinate.  Comparing these outputs as exact tokens rejects equivalent
    values solely because Python printed a different number of decimal
    places.
    """
    expected_tokens = expected.split()
    actual_tokens = actual.split()
    if len(expected_tokens) != len(actual_tokens):
        return False, "numeric output has a different number of tokens"
    try:
        expected_values = [float(token) for token in expected_tokens]
        actual_values = [float(token) for token in actual_tokens]
    except (TypeError, ValueError):
        return False, "output contains a non-numeric token"
    if any(
        not math.isfinite(value)
        for value in (*expected_values, *actual_values)
    ):
        return False, "non-finite floating-point output"
    for expected_value, actual_value in zip(expected_values, actual_values):
        tolerance = 1e-6 * max(1.0, abs(expected_value))
        if abs(actual_value - expected_value) > tolerance:
            return False, "numeric sequence differs beyond abs/rel tolerance 1e-6"
    return True, "numeric sequence comparison with abs/rel tolerance 1e-6"


def _judge_cablecar(
    input_data: str,
    expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate any legal pair of monotone cable-car forests (Kattis cablecar)."""
    try:
        n, k = map(int, input_data.split()[:2])
        expected_tokens = expected.split()
        lines = actual.split()
    except (TypeError, ValueError):
        return False, "invalid cablecar input or output"
    expected_yes = bool(expected_tokens) and expected_tokens[0].upper() == "YES"
    if not expected_yes:
        return (bool(lines) and lines[0].upper() == "NO"), "infeasible instance requires NO"
    if not lines or lines[0].upper() != "YES":
        return False, "feasible instance requires YES"
    if len(lines) != 1 + 4 * k:
        return False, "wrong number of cable endpoints"
    try:
        values = list(map(int, lines[1:]))
    except ValueError:
        return False, "cable endpoint is not an integer"

    def parse(offset: int) -> list[tuple[int, int]]:
        return [(values[offset + 2*i], values[offset + 2*i + 1]) for i in range(k)]

    mobi = parse(0)
    vina = parse(2 * k)

    def valid_company(edges: list[tuple[int, int]]) -> bool:
        if any(not (1 <= left < right <= n) for left, right in edges):
            return False
        starts = [left for left, _ in edges]
        ends = [right for _, right in edges]
        if len(set(starts)) != k or len(set(ends)) != k:
            return False
        order = sorted(range(k), key=lambda i: starts[i])
        return all(ends[order[i]] < ends[order[i + 1]] for i in range(k - 1))

    if not valid_company(mobi) or not valid_company(vina):
        return False, "company endpoints violate uniqueness or monotonicity"

    def connected_pairs(edges: list[tuple[int, int]]) -> set[tuple[int, int]]:
        graph = [[] for _ in range(n + 1)]
        for left, right in edges:
            graph[left].append(right)
            graph[right].append(left)
        result: set[tuple[int, int]] = set()
        for source in range(1, n + 1):
            seen = {source}
            stack = [source]
            while stack:
                node = stack.pop()
                for nxt in graph[node]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            result.update((min(source, target), max(source, target)) for target in seen if target != source)
        return result

    if connected_pairs(mobi) & connected_pairs(vina):
        return False, "the two companies connect a common station pair"
    return True, "valid cablecar construction"


def _judge_cf_623a(
    input_data: str,
    expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate any string whose equal/neighboring letters induce the graph."""
    try:
        tokens = input_data.split()
        n, m = int(tokens[0]), int(tokens[1])
        edge_set = {tuple(sorted((int(tokens[2 + 2*i]), int(tokens[3 + 2*i])))) for i in range(m)}
        lines = actual.split()
    except (IndexError, TypeError, ValueError):
        return False, "invalid graph-string input or output"
    expected_yes = expected.split() and expected.split()[0].lower() == "yes"
    if not expected_yes:
        return (bool(lines) and lines[0].lower() == "no"), "NO agrees with the reference feasibility result"
    if len(lines) != 2 or lines[0].lower() != "yes":
        return False, "feasible instance requires YES and one string"
    assignment = lines[1]
    if len(assignment) != n or any(ch not in "abc" for ch in assignment):
        return False, "invalid assignment string"
    produced = set()
    for i in range(n):
        for j in range(i + 1, n):
            if assignment[i] == assignment[j] or abs(ord(assignment[i]) - ord(assignment[j])) == 1:
                produced.add((i + 1, j + 1))
    return produced == edge_set, "valid CF 623A graph-string witness" if produced == edge_set else "assignment induces a different graph"


def _judge_dag_edge_removal(
    input_data: str,
    _expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate a removal set leaving an acyclic directed graph."""
    try:
        tokens = list(map(int, input_data.split()))
        n, m = tokens[:2]
        edges = [(tokens[2 + 2*i], tokens[3 + 2*i]) for i in range(m)]
        output = list(map(int, actual.split()))
    except (IndexError, TypeError, ValueError):
        return False, "invalid exitsinexcess input or output"
    if not output:
        return False, "missing removal count"
    count = output[0]
    removed = output[1:]
    if count != len(removed) or count < 0 or count * 2 > m:
        return False, "invalid removal count"
    if len(set(removed)) != count or any(not 1 <= idx <= m for idx in removed):
        return False, "removal indices are invalid"
    deleted = set(removed)
    graph = [[] for _ in range(n + 1)]
    indegree = [0] * (n + 1)
    for index, (left, right) in enumerate(edges, 1):
        if index not in deleted:
            graph[left].append(right)
            indegree[right] += 1
    queue = [node for node in range(1, n + 1) if indegree[node] == 0]
    seen = 0
    for node in queue:
        seen += 1
        for nxt in graph[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return seen == n, "acyclic graph after legal removals" if seen == n else "remaining graph contains a directed cycle"


def _judge_prefix_swaps(
    input_data: str,
    expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate any minimum prefix-swap sequence for CF 1012D."""
    lines = [line.split() for line in actual.splitlines() if line.strip()]
    expected_tokens = expected.split()
    if not expected_tokens or not lines:
        return False, "empty prefix-swap output"
    try:
        s, t = input_data.split()[:2]
        optimum = int(expected_tokens[0])
        count = int(lines[0][0])
    except (IndexError, TypeError, ValueError):
        return False, "invalid prefix-swap format"
    if count != optimum or len(lines) != count + 1:
        return False, "operation count is not the known minimum"
    for row in lines[1:]:
        if len(row) != 2:
            return False, "operation must contain two prefix lengths"
        a, b = map(int, row)
        if not (0 <= a <= len(s) and 0 <= b <= len(t)):
            return False, "prefix length out of range"
        old_s, old_t = s, t
        s = old_t[:b] + old_s[a:]
        t = old_s[:a] + old_t[b:]
    valid = (set(s) <= {"a"} and set(t) <= {"b"}) or (set(s) <= {"b"} and set(t) <= {"a"})
    return valid, "valid minimum prefix-swap sequence" if valid else "final strings do not have the required uniform letters"


def _judge_cf_883g(
    input_data: str,
    _expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate both optimal orientations, including their reachable counts."""
    tokens = list(map(int, input_data.split()))
    n, m, start = tokens[:3]
    directed = []
    undirected = []
    for index in range(m):
        kind, left, right = tokens[3 + 3*index:6 + 3*index]
        (directed if kind == 1 else undirected).append((left - 1, right - 1))
    output = actual.split()
    if len(output) != 4:
        return False, "expected two counts and two orientation strings"
    reported_max, reported_min = int(output[0]), int(output[2])
    signs_max, signs_min = output[1], output[3]
    if any(len(signs) != len(undirected) or set(signs) - {"+", "-"} for signs in (signs_max, signs_min)):
        return False, "orientation strings have invalid length or symbols"

    def reachable(extra_edges: list[tuple[int, int]]) -> int:
        graph = [[] for _ in range(n)]
        for left, right in directed + extra_edges:
            graph[left].append(right)
        seen = {start - 1}
        queue = [start - 1]
        for node in queue:
            for nxt in graph[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return len(seen)

    optimum_min = reachable([])
    optimum_max = reachable(undirected + [(right, left) for left, right in undirected])
    for reported, signs, optimum in (
        (reported_max, signs_max, optimum_max),
        (reported_min, signs_min, optimum_min),
    ):
        oriented = [edge if sign == "+" else (edge[1], edge[0]) for edge, sign in zip(undirected, signs)]
        observed = reachable(oriented)
        if reported != observed:
            return False, "reported count differs from the orientation's reachable set"
        if observed != optimum:
            return False, "orientation is not optimal"
    return True, "valid optimal CF 883G orientations"


def _judge_unbalanced_substring(
    input_data: str,
    _expected: str,
    actual: str,
) -> tuple[bool, str]:
    """Validate any witness for AtCoder's unbalanced-substring task.

    The reference pair in APPS contains one witness, but the statement allows
    every valid interval.  The checker therefore validates the reported
    interval against the input instead of comparing it with that one sample
    interval.
    """
    tokens = actual.split()
    if len(tokens) != 2:
        return False, "expected two integer positions"
    try:
        left, right = (int(token) for token in tokens)
    except (TypeError, ValueError):
        return False, "positions must be integers"
    text = input_data.strip().splitlines()[0].strip() if input_data.strip() else ""
    if left == -1 and right == -1:
        # A majority substring always contains two equal occurrences at
        # distance one or two (otherwise consecutive occurrences of its
        # majority character would be at least three positions apart, which
        # cannot exceed half of the interval).  This is the standard linear
        # witness test for the task and avoids an O(|s|^2) fallback.
        if any(text[index] == text[index + 1] for index in range(len(text) - 1)):
            return False, "reported no witness although one exists"
        if any(text[index] == text[index + 2] for index in range(len(text) - 2)):
            return False, "reported no witness although one exists"
        return True, "no unbalanced substring exists"
    if not (1 <= left < right <= len(text)):
        return False, "witness positions are outside the input string"
    substring = text[left - 1 : right]
    counts: dict[str, int] = {}
    for char in substring:
        counts[char] = counts.get(char, 0) + 1
    passed = max(counts.values(), default=0) * 2 > len(substring)
    return passed, "valid unbalanced-substring witness" if passed else "substring is balanced"


_SEMANTIC_JUDGES: dict[int, Judge] = {
    583: _judge_cf_1248d1,
    829: _judge_cf_1189a,
    1260: _judge_cf_1042c,
    2036: _judge_cf_1395b,
    4078: _judge_cf_1108e1,
    4091: _judge_cf_1006b,
    4105: _judge_cf_1118e,
    4377: _judge_cf_1154a,
    4451: _judge_cf_1256e,
    4518: _judge_cf_1433d,
    4763: _judge_kattis_calculatingdartscores,
    4897: _judge_kattis_bestrelayteam,
    # Kattis janitortroubles prints a real-valued area; ``9`` and ``9.0``
    # are equivalent judge outputs even when the sampled reference uses an
    # integer-looking token.
    4791: _judge_single_float,
    4966: _judge_single_float,
    # Codeforces Curling prints one real-valued coordinate per disk.  The
    # official checker applies the tolerance independently to every value.
    2148: _judge_float_sequence,
    # These tasks admit multiple valid witnesses or use floating-point
    # output; token equality would reject correct solutions.
    3065: _judge_single_float,
    3203: _judge_single_float,
    3448: _judge_cablecar,
    3745: _judge_cf_623a,
    3014: _judge_dag_edge_removal,
    3812: _judge_prefix_swaps,
    3988: _judge_cf_883g,
    # The AtCoder task accepts any interval with a strict majority character,
    # rather than the single interval stored in the APPS reference output.
    2886: _judge_unbalanced_substring,
    4974: _judge_kattis_bookingaroom,
    1630: _judge_cf_780d,
    # APPS function-style records return JSON arrays/booleans.  Their spacing
    # is immaterial, so compare the parsed values rather than raw tokens.
    2365: _judge_json,
    2883: _judge_json,
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

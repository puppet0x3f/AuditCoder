"""Regression tests for problem-specific output comparison."""

from auditcoder.common.output_judging import judge_output


def test_janitortroubles_accepts_integer_looking_float():
    result = judge_output(
        problem_id=4791,
        input_data="3 3 3 3\n",
        expected="9\n",
        actual="9.0\n",
    )
    assert result["passed"]
    assert result["judge_mode"] == "semantic_checker"


def test_janitortroubles_rejects_wrong_float():
    result = judge_output(
        problem_id=4791,
        input_data="3 3 3 3\n",
        expected="9\n",
        actual="9.1\n",
    )
    assert not result["passed"]


def test_curling_accepts_equivalent_float_sequence():
    result = judge_output(
        problem_id=2148,
        input_data="6 2\n5 5 6 8 3 12\n",
        expected="2 6.0 9.87298334621 13.3370849613 12.5187346573 13.3370849613\n",
        actual="2.0 6.0000000000 9.872983346207416 13.337084961299999 12.518734657300001 13.3370849613\n",
    )
    assert result["passed"]
    assert result["judge_mode"] == "semantic_checker"


def test_curling_rejects_wrong_coordinate():
    result = judge_output(
        problem_id=2148,
        input_data="2 1\n2 20\n",
        expected="1 1\n",
        actual="1 1.01\n",
    )
    assert not result["passed"]


def test_unbalanced_substring_accepts_any_valid_witness():
    result = judge_output(
        problem_id=2886,
        input_data="needed\n",
        expected="2 5\n",
        actual="2 3\n",
    )
    assert result["passed"]
    assert result["judge_mode"] == "semantic_checker"


def test_unbalanced_substring_rejects_balanced_witness():
    result = judge_output(
        problem_id=2886,
        input_data="needed\n",
        expected="2 5\n",
        actual="1 6\n",
    )
    assert not result["passed"]


def test_unbalanced_substring_accepts_no_witness_when_none_exists():
    result = judge_output(
        problem_id=2886,
        input_data="abc\n",
        expected="-1 -1\n",
        actual="-1 -1\n",
    )
    assert result["passed"]


def test_cf1248d1_accepts_any_optimal_swap():
    result = judge_output(
        problem_id=583,
        input_data="6\n()()()\n",
        expected="3\n1 1\n",
        actual="3 2 2\n",
    )
    assert result["passed"]
    assert result["judge_mode"] == "semantic_checker"


def test_cf1189a_rejects_nonminimum_partition():
    result = judge_output(
        problem_id=829,
        input_data="6\n100011\n",
        expected="2\n100 011\n",
        actual="3 1 000 11\n",
    )
    assert not result["passed"]


def test_cf1042c_accepts_valid_operation_witness():
    result = judge_output(
        problem_id=1260,
        input_data="5\n5 -2 0 1 -3\n",
        expected="2 3\n1 1 2\n1 2 4\n1 4 5\n",
        actual="1 2 5\n2 3\n1 1 5\n1 5 4\n",
    )
    assert result["passed"]


def test_cf1395b_accepts_any_rook_hamiltonian_path():
    result = judge_output(
        problem_id=2036,
        input_data="2 2 1 1\n",
        expected="1 1\n1 2\n2 2\n2 1\n",
        actual="1 1\n1 2\n2 2\n2 1\n",
    )
    assert result["passed"]


def test_cablecar_accepts_another_valid_witness():
    result = judge_output(
        problem_id=3448,
        input_data="3 1\n",
        expected="YES\n1 2\n1 3\n",
        actual="YES\n1 2\n2 3\n",
    )
    assert result["passed"]


def test_cf623a_accepts_label_symmetry():
    result = judge_output(
        problem_id=3745,
        input_data="2 1\n1 2\n",
        expected="Yes\naa\n",
        actual="Yes\nbb\n",
    )
    assert result["passed"]


def test_exitsinexcess_accepts_any_acyclic_removal_set():
    result = judge_output(
        problem_id=3014,
        input_data="2 2\n1 2\n2 1\n",
        expected="1\n2\n",
        actual="1\n1\n",
    )
    assert result["passed"]


def test_cf883g_checks_orientation_and_reachability():
    result = judge_output(
        problem_id=3988,
        input_data="2 2 1\n1 1 2\n2 2 1\n",
        expected="2\n-\n2\n+\n",
        actual="2\n+\n2\n+\n",
    )
    assert result["passed"]


def test_prefix_swaps_rejects_invalid_duplicate_operations():
    result = judge_output(
        problem_id=3812,
        input_data="bab\nbb\n",
        expected="2\n1 0\n1 3\n",
        actual="2\n1 0\n1 0\n",
    )
    assert not result["passed"]

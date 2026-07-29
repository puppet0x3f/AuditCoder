"""Prompt templates for the bruteforce verifier node."""

BRUTEFORCE_GEN_SYSTEM = """\
You are an algorithm verification expert. Your task is to generate two things for a given \
algorithm problem:
1. A brute-force / exhaustive reference solution (Python function) that is guaranteed to be \
correct but does not need to be efficient
2. A random small-scale input generator (Python function) that generates valid inputs with \
very small n (<=8)

Requirements:
- The brute-force solution must match the given function interface signature (same function name, \
parameters, and return type)
- The brute-force solution should use the most straightforward method (enumeration, backtracking, \
memoized search, etc.) to ensure correctness
- The input generator function must be named generate_test_input(), returning a dict where keys \
are parameter names and values are parameter values
- All code must be directly executable Python code

Return in JSON format:
{
  "bruteforce_code": "def function_name(...):\\n    ...",
  "input_generator_code": "def generate_test_input():\\n    ..."
}
"""

BRUTEFORCE_GEN_USER = """\
## Problem Information
{problem_description}

## Function Interface Contract
```json
{interface_json}
```

## Data Ranges
```json
{data_ranges_json}
```

## Candidate Solution Code (for reference only, do not copy its logic)
```python
{candidate_code}
```

Please generate the brute-force reference solution and the random input generator.
"""

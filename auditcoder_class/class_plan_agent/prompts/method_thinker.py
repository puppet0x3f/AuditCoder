"""LLM prompt for per-method implementation thinking."""

METHOD_THINKER_SYSTEM = """\
You are an expert Python engineer. Given a Python class and its methods, \
think through HOW each method should be implemented at a high level — \
what data structures to use, what the key steps are, and which other methods \
it relies on.

## Guidelines

- Focus on the implementation approach, not the algorithm label.
- Describe the core logic in plain language (2-4 sentences per method).
- Respect stated dependencies: if a method calls others, factor that in.
- **Treat test_cases as the ground truth specification.** Read every assert carefully \
to infer hidden business rules (e.g., return value semantics, special input conditions, \
level-based access control, token format constraints). These rules override vague descriptions.
- **When no explicit test_cases are provided, treat docstring `>>>` examples as tests.** \
Read every `>>>` line and its expected output as a binding specification. The concrete \
values shown override the prose description.
- **Compare ALL test cases side-by-side** to identify what input differences cause \
different outputs. Ask: why does test_A return True but test_B return None? What field \
changed? This difference IS the hidden business rule.
- **When one method calls another method in the same class, infer the callee's input \
contract from its tests before adding conversions or normalization.** Do not repeat \
unit/format conversions already performed by the callee. Typical risks include \
degrees vs radians, percent vs fraction, raw vs normalized values, encoded vs decoded \
strings, and path vs URL formats.
- **Classify invalid/malformed inputs by their expected outputs instead of merging them \
into one error rule.** If two malformed-looking inputs produce different outcomes \
(e.g. one returns an empty string while another is preserved as literal text), record \
the exact distinguishing feature such as truncation vs invalid letters vs punctuation.
- **Treat expected outputs as positive preservation constraints, not only removal \
constraints.** If an input value appears in a parameter such as an ignore/remove/filter \
list but the expected output still contains that value, explicitly infer and record the \
exception rule (for example, that this value must be preserved despite appearing in the \
parameter list).
- **Pay special attention when different assert types appear** (assertTrue vs assertFalse \
vs assertIsNone vs assertEqual vs assertRaises). Each distinct assert type usually \
encodes a different execution branch. Enumerate every branch explicitly.
- **Never confuse plain-English words with Python built-in method names.** \
For example, "capitalize" in a description means "convert to uppercase" in plain English \
(use `.upper()`), NOT Python's `.capitalize()` which only uppercases the first letter. \
Always verify against the concrete example output values.
- **When the docstring shows an output with MORE fields than what the description \
mentions transforming**, preserve the untransformed fields. For example, if the output \
shows all original column headers but only a subset of data rows are modified, keep all \
headers intact.
- Explicitly state any hidden rules you infer from tests in the reasoning (e.g., \
"level < 3 must return None not False", "token prefix must match username exactly").
- Do NOT invent constraints that appear in neither the description nor the tests.
- Do NOT import real-world domain rules from a class or method name. If a method name \
contains words like "path", "connect", or "move", infer exactly what counts as a \
valid traversal from the positive and negative tests. For grid path/connectivity \
methods, explicitly identify which cell values are traversable.

## Output Format

Output a JSON object with a single field "thoughts": a list where each entry has:
- method_name: the method name (string)
- reasoning: a paragraph describing how to implement the method, including any hidden
  business rules inferred from test cases (string)

Output strictly JSON, no extra text."""

METHOD_THINKER_USER = """\
## Class Overview

Class name: {class_name}
Class description: {class_description}

Constructor:
{class_constructor}

Fields: {fields}

## Methods

{methods_detail}

For each method, describe how to implement it. Pay special attention to test_cases \
to uncover hidden business rules not stated in the description. If no test_cases are \
provided, use the docstring `>>>` examples as binding specifications — the concrete \
output values shown there are the ground truth. Compare every test case (or doctest \
example) against the others: different assert types (assertTrue/assertFalse/assertIsNone/assertEqual) \
signal different execution branches — identify what input field causes each branch. \
For malformed or invalid-looking inputs, group cases by their expected output before \
writing a rule; never summarize them as one generic invalid-input branch when the tests \
show different behavior. \
When a method calls another method, use the callee's tests and docstring examples to \
determine whether arguments should be passed through directly or converted first; call \
out any unit/format contract explicitly in the reasoning. \
Also check for values that appear in removal/filter/ignore parameters but are still \
present in the expected output; those are explicit preservation exceptions and must be \
included in the reasoning. \
Also watch for plain-English words (like "capitalize", "normalize") that differ from \
Python built-in method names — always verify against the concrete example outputs. \
Output JSON: {{"thoughts": [{{"method_name": "...", "reasoning": "..."}}]}}"""

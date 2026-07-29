"""Shared prompt templates for all method agents."""

CLASS_METHOD_CODING_STANDARDS = """\
## Class Method Standards

Before writing code, examine the Constructor shown in Class Context:
- List every `self.*` attribute and its initial type/value
- Identify attributes that are logically linked (e.g. a list and its length counter)

Rules:
1. Use ONLY `self.*` names established in `__init__` — never invent new instance attributes \
unless the method's explicit purpose is to introduce new state
2. `__init__` must NOT have an explicit `return` statement
3. Non-`__init__` methods must NOT re-initialize attributes already set by `__init__`, \
unless resetting is the method's stated purpose
4. When mutating state, update ALL logically linked attributes together to maintain class invariants
5. Never alter the method signature (parameters, decorators, return annotation)
6. Access all instance state via `self`; call sibling instance/class methods as `self.method_name(...)`; \
call sibling @staticmethod methods as `ClassName.method_name(...)` — never as bare `method_name(...)`
7. Do not use module-level (global) variables
8. Output ONLY the method body — indented with 8 spaces (2 levels); do NOT include the `def` line. \
Every non-blank top-level line in the method body, including opening/closing docstring quotes \
and docstring text, must use the same 8-space base indentation. Do not mix 5/7/9-space \
docstring indentation.
9. **String escape sequences**: when writing Python string literals that contain `\n`, `\t`, or \
other escape sequences, always write them with the backslash notation (e.g. `'\\n'.join(...)`). \
Never embed a literal newline or tab character inside a string literal in your output — \
that produces a SyntaxError in the assembled file."""

METHOD_AGENT_USER = """\
## Class Context

{class_context}

## Method to Implement

{method_interface}

Method description:
{method_description}

## Algorithm Hints

{hints}

## Instructions

Before writing code, scan the Constructor shown above: confirm which `self.*` attributes \
exist and note any that are logically linked. Use only those attribute names — \
never introduce new `self.*` names unless this method's purpose is to add state.

If ground-truth tests are present in the class context, treat them as the highest-priority \
behavioral specification. If the docstring, hints, or implementation reasoning conflict \
with a test assertion, follow the test assertion.

Before calling a sibling method, verify that method's expected input representation from \
its tests, docstring examples, and any shown implementation. Do not repeat conversions \
that the sibling method already performs. This is especially important for unit/format \
pairs such as degrees/radians, percent/fraction, raw/normalized values, encoded/decoded \
strings, and path/URL forms.

When implementing filtering/removal/ignore-list logic, do not assume every element in a \
parameter list must be removed. Compare the asserted expected output against the input: \
if a value appears in the removal/filter/ignore parameter but is still present in the \
expected output, preserve that value and encode the exception explicitly in the code.

When implementing parsers, scanners, unescapers, validators, or other malformed-input \
handling, classify each malformed-looking test by its asserted output. Do not collapse \
all malformed inputs into one behavior. If one malformed case returns an empty value but \
another malformed case is preserved as literal text or partially processed, encode the \
distinguishing condition explicitly.

Algorithm hints are advisory and may be wrong. Discard any hint that conflicts with \
ground-truth tests, especially hints that add unstated domain-specific rules.

If "Implementation reasoning" is present in the class context, use it as helpful guidance, \
but verify it against the ground-truth tests before coding.

Do not import real-world domain rules from class or method names. For example, a class name \
that resembles a game, parser, or data structure does not imply full official rules. Implement \
only the simplified behavior demonstrated by the tests and method contract.

Generate ONLY the method body lines (indented with 8 spaces, i.e., 2 levels of indentation). \
Ensure docstring opening quotes, docstring contents, closing quotes, and executable statements \
all compile when inserted under the existing method definition. \
Do NOT include the `def method_name(self, ...):` line — that is already in the class skeleton.
Output JSON: {{"method_body": "..."}}"""

"""Prompt templates for the weak signal evaluation node."""

WEAK_SIGNAL_SYSTEM = """\
You are a code review expert. Please evaluate the given algorithm code along the following three \
dimensions and return the result in JSON format.

Evaluation dimensions:
1. **Readability** (readability_score: 1-5)
   - 1: Extremely difficult to read; no naming conventions, no structure
   - 3: Average; basically readable but with room for improvement
   - 5: Excellent; clear naming, well-structured, logic is easy to follow

2. **Code Style** (style_score: 1-5)
   - 1: Completely non-compliant with Python conventions
   - 3: Mostly compliant; minor issues present
   - 5: Fully compliant with PEP 8; consistent style throughout

3. **Explanation Consistency** (explanation_consistent: bool)
   - Whether the Agent's explanation of its own code is consistent with the code's actual behavior
   - If no explanation is provided, default to true

Return in JSON format:
{
  "readability_score": <1-5>,
  "style_score": <1-5>,
  "explanation_consistent": <true/false>,
  "notes": ["note1", "note2"]
}
"""

WEAK_SIGNAL_USER = """\
## Code
```python
{source_code}
```

## Function Interface Contract
```json
{interface_json}
```

## Agent's Explanation
{explanation}

Please evaluate the above code.
"""

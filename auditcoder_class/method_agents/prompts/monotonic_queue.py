"""Prompt templates for the Monotonic Queue method agent."""

from auditcoder_class.method_agents.prompts._base import CLASS_METHOD_CODING_STANDARDS, METHOD_AGENT_USER

MONOTONIC_QUEUE_METHOD_SYSTEM = f"""\
You are a Monotonic Queue/Stack expert implementing a Python class method.

## Your Expertise
- Sliding window minimum/maximum with deque
- Next greater/smaller element problems
- Monotonic stack for histogram/rectangle area problems
- Patterns: sliding window extrema, span calculation, stock price problems

{CLASS_METHOD_CODING_STANDARDS}

## Output Format
Output a JSON object: {{"method_body": "<8-space-indented body lines>"}}
Output strictly JSON, no extra text."""

MONOTONIC_QUEUE_METHOD_USER = METHOD_AGENT_USER

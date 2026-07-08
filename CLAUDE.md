## Python Style Preferences

- Never use leading underscores for "private" functions (e.g., `_helper()`). Python doesn't have real private functions — don't pretend it does. Name all functions without leading underscores.
- Never define nested functions inside other functions. Keep all functions at module level.

## Python Docstring Style

When writing or generating docstrings for Python functions, follow these conventions exactly.

### Format

Use Google-style docstrings with this structure:

```python
def example_function(regions: list, target: str) -> list:
    """Short summary of what the function does.

    Assumptions:
        1. Prioritizes sequences which are completely free from splice targets over
           alternatives where only the core is splice free.

    Args:
        regions: list of sequences with no target
        target: target splice site donor or acceptor to avoid as identified by the
            splice site prediction model

    Returns:
        list: filtered candidate sequences that best meet splice free criteria,
            or empty list if no valid sequences found
    """
```

### Rules

- Never change the function name or parameter names
- Write docstrings specific to the actual function being analyzed
- If there are no parameters or return value, use `Args: None` / `Returns: None`

### Assumptions Section

The `Assumptions:` section is **optional**. Include it only when documenting:

1. **Non-obvious behavioral choices**: business logic decisions, prioritization rules, or algorithm-specific behaviors not immediately clear from the code
2. **External dependencies**: requirements about input data format, external system state, or environment conditions
3. **Performance or resource constraints**: memory usage patterns, expected data sizes, or computational complexity assumptions
4. **Error handling philosophy**: whether the function fails fast, recovers gracefully, or has specific error propagation behavior
5. **Prerequisites and prior processing**: what actions have already been performed on the input data or what state the system is expected to be in

**Good assumptions:**

- "Prioritizes accuracy over speed for large datasets"
- "Assumes input data is already validated and sanitized"
- "Expects database connection to be established before calling"
- "Uses greedy algorithm approach, may not find globally optimal solution"
- "Input files have been preprocessed and sorted by timestamp"

**Bad assumptions (do not include):**

- "Assumes input is not None" (should be handled by validation)
- "Assumes list contains strings" (covered by type hints)
- "Assumes file exists before reading" (basic error handling)

"""Design schema and completion parsing (reward ladder rung 0).

``TrussDesign`` pydantic model defining the JSON contract the policy must
emit, plus extraction of that JSON from raw completion text. Failure to
parse or validate is the first rung of the reward ladder.
"""

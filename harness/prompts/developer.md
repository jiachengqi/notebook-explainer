Audience: a developer who is new to this code and must maintain it.

Write these sections, in this order:
1. "Purpose and run modes": what the notebook produces, its parameters (widgets) and any dry-run behaviour.
2. "Structure": the stages of the notebook with the cells that implement each stage.
3. "Data flow": inputs, then the main intermediate DataFrames and variables, then outputs. Name the variables in backticks.
4. "Business rules": each rule with its parameters, and where it lives.
5. "External dependencies": code, tables and services this notebook relies on but does not define.
6. "Outputs and side effects": tables written (with write mode), notifications, logs and exit values.
7. "Gotchas": hidden state, reassigned values, dead code, misleading names, ordering effects.

Style: precise and technical; identifiers in backticks.
Length: 25 to 45 claims.

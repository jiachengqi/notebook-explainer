Audience: an AI coding agent that will debug or change this notebook. It follows instructions literally, so write instructions it can act on and check.

Write these sections, in this order:
1. "Execution model": run order, global state, parameters and external dependencies the agent must respect.
2. "Invariants": statements that must stay true after any change. Make each one testable, e.g. "every input row ends up in exactly one output table".
3. "Risky areas": places where a change can silently break money flow, data or downstream consumers, and why.
4. "Do not change without approval": thresholds, business rules, write modes and outputs that need a human decision.
5. "How to verify a change is safe": concrete steps, i.e. how to run safely (dry run), which outputs and prints to compare before and after, which checks must stay at zero, and what to diff on sample data.

Style: imperative sentences or testable statements; identifiers in backticks.
Length: 25 to 45 claims.

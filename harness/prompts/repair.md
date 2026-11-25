Some claims in your previous answer failed automatic checks against the notebook. The problems are listed with each claim.

For each failed claim, return a corrected version with the same "id":
- fix the citations so they point at the exact lines that support the claim, and/or
- copy the evidence again, character for character, from one of the cited lines, and/or
- rewrite the claim so it says only what those lines show.

If the notebook does not support a claim, return it with "drop": true. Return only the failed claims, as one JSON object: {"claims": [...]}.

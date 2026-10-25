# Verification

Locally verified on 23 October 2026:

- Five automated tests passed, including a complete synthetic four-pool/two-target study, matching source prediction/metric reproduction, inner selection-objective reproduction, all four named baseline comparisons, decision freezing, checkpoint reuse and tamper rejection.
- Algebra tests verify exact cancellation, the blend-error identity and deterministic selection on the original grid.
- Lock tests reject an existing lock and cross-host stale-lock recovery.
- The synthetic integration test used the original four alphas plus one neighboring value and the five original weights; production uses all nine alphas and 13 weights. The full production grid is checked by the plan stage.
- Local NetCDF import emitted the existing `numpy.ndarray size changed` warning; tests completed successfully.
- ZIP integrity, all packaged SHA-256 hashes and the extracted planning command are checked by the bundle builder.
- The optional plotting command was not executed successfully locally because Matplotlib is absent from the available local environments. The scientific runner and tests do not require Matplotlib. The plotting script is supplied separately and requires the server's plotting dependency.

Real native-grid follow-up fits have **not** been run locally. Server runtime and the actual scientific outcomes are unknown. The production runner stops if original full-model candidate objectives, LPS scores, blend scores or archived blend predictions cannot be reproduced within the declared numeric tolerances. Synthetic tests are correctness evidence, not climate results.

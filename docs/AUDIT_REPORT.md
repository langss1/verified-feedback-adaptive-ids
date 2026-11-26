# Paper–GitHub audit, 26 November 2026

The saved numerical evidence agrees with the manuscript. The audit found and
repaired presentation/reproduction defects; no model was retuned or result changed.

| Finding | Repair and evidence |
| --- | --- |
| Paper described 1,024 balanced source samples as an exact count. | R11 specifies a 512/class cap, at most 1,024 total; actual 684–706 samples are listed in `source_memory_audit.json`. |
| The 82-feature notation could be read as applying to NSL/UNSW development. | R11 restricts it to corrected CIC; per-seed actual dimensions are in `data_map.csv`. |
| AP score semantics were not explicit enough. | R11 adds the existing odds-shift equation and maps `raw`, `p`, `p1`, `p2`, and `threshold` in the guide. |
| Public fresh replay required nonexistent README.txt. | Wrapper now copies the documentation file that exists; an actual fresh Botnet replay matches both target/source hashes. |
| D5 raw preparer imports were missing from the root public source. | Original `remote_zip.py` and `prepare_external.py` helpers are included; import tested. Full raw reacquisition is outside this audit. |
| GitHub highlighted R7 while the local reviewed manuscript was R10. | R11 and its source are published together with the audit/evidence maps in v1.2.0; historical releases remain intact. |
| Data/evidence locations were scattered and easy to confuse. | One reviewer guide maps 60 prepared files, array schemas, code symbols, result cells, provenance and releases. |

Validation: 111 public git blobs and seven release assets verified; all seven
assets reachable anonymously; all 870 original and 240 additional saved
evaluations pass their audits; 70 Table II cells and 96 paired intervals agree
with independent recomputation; 120 routing reconstructions pass; two selected
fresh training replays match exact target/source prediction hashes. The original
86-file D4 lock is unchanged. See `audit_receipt.json` for scope and counts.

Limits remain explicit: prepared-input reproduction is supported; full raw-level
ingestion of all legacy development datasets is not fully packaged. All feedback
is simulated; same-testbed captures/seeds are not independent populations;
simple maximum remains stronger in mean BA; gate gains miss their practical
target. This audit improves consistency and traceability, not the measured
novelty, statistical generalization, or acceptance probability.

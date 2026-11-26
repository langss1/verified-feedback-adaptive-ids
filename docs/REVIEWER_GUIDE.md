# Reviewer guide: manuscript, source code, and data

Start with the [audited six-page English paper](https://github.com/langss1/verified-feedback-adaptive-ids/blob/v1.2.0/manuscripts/r11/paper.pdf).
The experimental implementation is frozen at commit
`3a64384e756c4847976b2919a14e44d03bca177f`. Release `v1.2.0` adds this guide,
paper clarification, data/evidence maps and packaging repairs; it does not change
the frozen model, inputs, selection, or any measured experimental result.

## What the paper does

The detector (ML1, HGB) learns from arrived training labels. The local scorer
(ML2, weighted 7-NN) supports query selection and a restricted positive decision
path. ML2 does not assess a human's competence. A separate audit label pool
selects the operating threshold; those labels never train either classifier.
The selected configuration uses no pseudo-label training. This is an evaluation
of controlled integration, not a new HGB/NN algorithm or a real-human study.

Read the introduction as problem → attribution gap → RQ1/RQ2/RQ3 → objectives →
contributions. RQ1 tests the adaptive package against frozen detection; RQ2
tests the incremental gate/controller path under shared acquisition; RQ3 relates
failures to feedback arrival and post-fit attack opportunity.

## Navigate from a claim to evidence

- [Claim → section → code function → result map](claim_evidence_map.csv).
- [All 70 Table II cells recomputed from prediction arrays](table_ii_audit.csv).
- [96 paired metric intervals: 288 means and endpoints](paired_interval_audit.csv).
- [Independent saved-run and fresh-replay receipts](audit_receipt.json).
- [Per-file data inventory](data_map.csv) and [per-array schema](data_schema.json).
- [Public release assets, sizes and SHA256](release_asset_map.csv).

The [paper-result verifier](https://github.com/langss1/verified-feedback-adaptive-ids/blob/v1.2.0/verify_paper_results.py) recomputes all Table II
cells and paired intervals from the restored prediction arrays, and checks the
issued-score equation against every stored system score. It does not train models.

The paper's original 11 equations are retained; its additional issued-score
equation makes the existing implementation explicit. In a result NPZ, `p1` is
raw HGB, `p2` is raw NN, `raw` is the selected routing score q, `threshold` is
the threshold at issuance, and `p` is the odds-shifted decision score. Metrics
classify `p >= 0.5`, equivalent to `raw >= threshold`. System AP ranks `p` over
the stream; raw-model AP ranks `p1` or `p2`. Because thresholds vary over time,
system AP is not generally the AP of the raw combined scores. None is claimed
to be a calibrated posterior. Source scores use the same final controller.

Source replay/prototypes are selected without replacement, at most 512 per
class and 1,024 total. Actual sizes by seed are 703, 692, 706, 684 and 704;
the smaller attack class prevents equal class counts. See
[source_memory_audit.json](source_memory_audit.json). These are fitting-memory
caps, not bounds on stored flow rows, ledgers, or ARF correction replay.

## Data map and access

The public release URLs are in [release_asset_map.csv](release_asset_map.csv).
All seven existing assets were reachable anonymously and matched their
published sizes and locally recomputed SHA256 values during this audit.

| Archive | Release | Exact role and scope |
| --- | --- | --- |
| `prepared_inputs.zip` | v1.0.0 | 50 numeric NPZ files: historical splits, development streams, final retention probes, and four original external cohort types, across five seeds. |
| `saved_predictions.zip` | v1.0.0 | All 870 development/confirmation run predictions, compressed request/feedback/fit ledgers, and result CSVs. |
| `example_models.zip` | v1.0.0 | Illustrative final Botnet model and synthetic workflow demo; not aggregate research evidence. |
| `D5_Adaptive_Baseline_Extension.zip` | v1.1.0 | Ten new-capture NPZ files, initial-state caches, 240 additional system evaluations, logs, protocols and complete extension analysis. |

The GitHub source ZIP by itself excludes large NPZ/prediction assets. Download
the release archives; there is no need to request private Drive access.

The per-file map records release, ZIP, canonical workspace directory, relative
path, SHA256, seed, role, actual feature count, stored row count, selected stream
row count and binary support. Its 60 rows describe files, not 60 independent
populations. The schema identifies each array's shape, dtype and information
role. `y_test` and family/timestamp/raw-row metadata are evaluator/simulator
information; they are not passed as unrestricted labels to the learner.

| Cohort | Role | Selected flow support |
| --- | --- | --- |
| Corrected CICIDS2017 Tuesday | Historical source BASE/META/initial CAL | BASE 8,192; META and initial CAL 1,024 each per seed. |
| Final S3 Tuesday probe | Confirmation source retention only | 1,024 balanced labels per seed; disjoint sessions/fingerprints from used historical splits and archived S2. |
| NSL-KDD / UNSW-NB15 | Development only | 4,000 / 6,000 selected stream rows; stored arrays are larger. Encodings vary by seed (116–118 / 157–170 features). |
| CICIDS2017 Wednesday / Friday | Development only | 12,000 flows each; 82 corrected flow features. |
| DoS15 / Bot02 | Original primary confirmation | 12,000 each; 81 / 334 attacks. |
| Web challenge | Separate case-control challenge | 12,158 total: 12,000 benign and 158 attacks; prevalence is constructed. |
| Web main | Descriptive only | 12,000 total with one attack; not a supported primary test. |
| Hulk16 / DDoS20 | Additional previously unscored captures | 12,000 each; 3,920 / 843 attacks. |

During confirmation, `run_confirmation.py`/D5 `history()` replace the initial
historical CAL with final S3. Initial CAL in `inputs/history_<seed>.npz` is not
the final confirmation probe. Source transformation parameters are frozen from
historical BASE. Seed-specific target matrices share cohort identities but use
their corresponding source transformation; they are not newly sampled networks.

The 2018 binary label maps attempted attacks to benign following the corrected
dataset convention. Use `y_test`, not a naive comparison of the family string to
BENIGN, when recomputing binary metrics. Family strings preserve original attack
descriptions, including attempted categories.

## Fresh prepared-input reproduction

Use Python 3.11. These are the canonical sibling directories:

```text
workspace/
  adaptive_feedback_d4_20261126/  # repository at tag v1.2.0
  adaptive_baseline_d5_20261126/  # extension ZIP, not extensions/d5 in place
```

```bash
git clone --branch v1.2.0 https://github.com/langss1/verified-feedback-adaptive-ids adaptive_feedback_d4_20261126
cd adaptive_feedback_d4_20261126
python -m pip install -r requirements.txt
python download_artifacts.py --results
python audit_results.py
python fresh_replay.py --dataset bot_main --seed 11 --scenario delay80
```

Download `D5_Adaptive_Baseline_Extension.zip` from release v1.1.0 and extract it
into `workspace/`. Use a separate environment with the D5 requirements, then:

```bash
cd ../adaptive_baseline_d5_20261126
python -m pip install -r requirements.txt
python check_adapter.py
python audit_extension.py
python analyze_extension.py
python fresh_replay_check.py
python ../adaptive_feedback_d4_20261126/verify_paper_results.py --d5 .
```

D4 uses pandas 2.1.3. D5 execution uses its pinned pandas 2.2.3/River 0.22.0;
raw fingerprint preparation uses pandas 2.1.3 for parity with D4. The audit here
used the existing matching installed dependencies for D5; they are third-party
requirements, not redistributed as project code. Saved prediction arithmetic
was audited for all 870 D4 runs and 240 added evaluations. Training was freshly
rerun for two selected cases, not every seed/condition. A full rerun needs more
CPU time; see the D4 and D5 READMEs and preserve original locks/results.

## Raw sources, provenance, and limits

- Corrected CICIDS2017: https://intrusion-detection.distrinet-research.be/WTMC2021/
- Corrected CSE-CICIDS2018: https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html
- Provider download page: https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html

Prepared-array replay is the complete supported reproduction path. This release
does not claim one-command reconstruction from all original NSL/UNSW/CIC raw
archives. Legacy development ingestion/splitting context is not fully packaged.
Its prepared matrices and hashes are public, and the dimensions actually used
are recorded in the data map. No raw-data rebuilding was newly performed here.

For D5 raw rebuilding, `prepare_new.py` streams the pinned 2018 ZIP members.
It also needs the corrected 2017 ZIP in
`workspace/dual_verifier_d2_20261125/data/CICIDS2017_improved.zip`, plus restored
D4 inputs and `deployment/transform_seed_<seed>.json`. Use pandas 2.1.3 for
fingerprint parity. `remote_zip.py` and the `HashReader`/`priority` helpers from
`prepare_external.py` are now included in the root repository. Their import path
was tested; the large archives were not reacquired for this audit. The legacy
`prepare_external.py` main routine has extra legacy dependencies and is not the
prepared replay command.

Receipts retain source ZIP/member names, raw row IDs, timestamps, fingerprint
keys, member CRCs and decompressed SHA256. Only selected 2018 members were
streamed: no full-archive SHA256 is claimed. Historical aggregate HTTPS counters
marked unknown remain unknown. Dataset rights stay with their providers; MIT
applies to this implementation, not to underlying datasets or River.

## Interpret the evidence correctly

- 870 original runs = 48 screening + 432 checking + 390 confirmation.
- D5 adds 80 ARF runs and 40 shared HGB/NN trajectories, yielding 240 additional
  system evaluations. Reusing 160 D4 evaluations gives a 400-row comparison
  matrix. Shared routing and reused cohorts are not independent training runs.
- Five seeds measure algorithmic variation, not five organizations. New captures
  share the historical testbed and 63 predictor fingerprints; overlap with earlier
  sampled 2018 cohorts is zero. A post-hoc overlap-exclusion sensitivity is not a
  clean new training experiment.
- GDT improves over frozen detection, but gate increments remain below the
  one-point practical target. Simple maximum has higher mean BA while meeting
  the study FPR limit. ARF shares queries/labels/audits but differs in weighting,
  native updates, correction replay and memory; its lower score does not isolate
  gate novelty or prove HGB universally superior.
- All feedback is simulated. SQL overall recall is zero; post-fit recall is
  undefined where no later positive exists. Neither human performance nor
  production readiness is measured.

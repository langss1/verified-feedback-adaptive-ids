# Verified Feedback Adaptive IDS

Reproducible two-classifier adaptive intrusion-detection study. ML1 is histogram
gradient boosting. ML2 is a weighted seven-neighbor inspector. The selected
positive-evidence gate uses arrived verified labels, and pseudo-label training
is disabled. Models are refitted periodically; this is not per-sample partial_fit.

## Start the paper review

Read the [latest audited six-page English manuscript](manuscripts/r11/paper.pdf)
and [reviewer guide](docs/REVIEWER_GUIDE.md). The guide links a 60-file data map,
array schemas, claim-to-code evidence, result recomputation, exact release assets
and reproduction commands. [Audit findings and repairs](docs/AUDIT_REPORT.md)
are explicit; the frozen experimental results are unchanged.

Release [v1.2.0](https://github.com/langss1/verified-feedback-adaptive-ids/releases/tag/v1.2.0)
contains the synchronized manuscript/source and audit maps. Experimental data
remain in v1.0.0/v1.1.0; source caps are not guaranteed class balance, and the
82-feature space applies to corrected CIC confirmation. All labels are simulated.

## D5 extension: matched native ARF and additional captures

The frozen extension adds 80 native-ARF runs and 40 shared HGB/NN trajectories,
yielding 240 additional system evaluations. Two previously unscored capture
members contain 12000 flows each. The detailed protocol, audit and analysis are
in [extensions/d5](extensions/d5). Download the reproducible extension, updated
historical five-page R7 manuscript, editable source and supplement from
[release v1.1.0](https://github.com/langss1/verified-feedback-adaptive-ids/releases/tag/v1.1.0).

| New capture | GDT BA | Native ARF BA | Simple maximum BA | GDT - verified BA |
| --- | ---: | ---: | ---: | ---: |
| Hulk16 | 96.46% | 89.10% | 97.15% | +0.29 pp |
| DDoS20 | 91.45% | 78.08% | 93.52% | +0.55 pp |

The gate still misses the one-point incremental target, and simple maximum
remains higher in mean BA. Native ARF shares acquisition/labels/audits but differs
in weighting, memory and update cadence; this is a package comparison. Both new
captures belong to the same historical benchmark and share 63 fingerprints.
All annotations are simulated. No model was retuned using these outcomes.
The original D4 source/input lock and v1.0.0 artifacts remain unchanged.

## Original D4 findings and scope

The fixed confirmation consists of 390 runs and 4,699,750 flow decisions.
The complete development and confirmation archive has 870 audited runs.
Five seeds reuse the same capture cohorts, rather than five independent sites.

| Primary cohort | Balanced accuracy | Attack recall | False-positive rate |
| --- | ---: | ---: | ---: |
| DoS | 64.74% | 29.57% | 0.09% |
| Botnet | 94.31% | 88.85% | 0.22% |

These values average four feedback conditions within each seed, then five seeds.
The gate's BA gain over matched verified-only learning is 0.69 percentage points
on DoS and 0.13 on Botnet, below the prespecified one-point practical criterion.
The simple maximum-score control performs better in BA on these captures.
SQL recall on the separate Web challenge is zero. The Web challenge is a
case-control sample; the original one-attack Web sample is descriptive only.
This repository is a research prototype, not a production-readiness claim.

All experimental feedback is SIMULATED from benchmark labels. Zero real-human
participant results are claimed. The CSV review workflow can accept actual
independent reviews later. Source/recent memory and audit labels have distinct
roles; audit labels never train either classifier.

## Setup

Use Python 3.11 and `pip install -r requirements.txt`. The recorded experiment
used Windows; bit-for-bit replay was checked on that environment.

Run `python download_artifacts.py` to fetch and hash-verify prepared inputs and
the illustrative model from the public release. Add `--results` for all saved
prediction sequences and compressed fit/query/correction ledgers. ZIP paths are
extracted relative to the repository root. No dataset is fetched implicitly by
the training code.

## Replay and independent audit

```bash
python download_artifacts.py --results
python audit_results.py
python fresh_replay.py --dataset bot_main --seed 11 --scenario delay80
```

The audit checks 870 saved runs, arithmetic, label availability, held-label
exclusion, fit-input hashes, matched query/inspector histories and score rules.
It does not prove universal software correctness. `fresh_replay.py` writes a
NEW workspace and checks one selected run's target and source prediction hashes.

To recompute all 390 confirmation runs in a new workspace:

```bash
python fresh_replay.py --all --workers 6
```

This requires substantial CPU time. `run_confirmation.py --workers 6` validates
the original freeze and skips cases that are already complete in its workspace.
It is a resume command, not a request to overwrite archived results.

## Analyze the archived or replayed outputs

```bash
python finalize_analysis.py
python describe_feedback_support.py
python make_figures.py
```

`finalize_analysis.py` preserves the frozen analysis source and removes unmatched
aggregate comparisons with the clean1-only full-label reference. The correction
is documented in `provenance/postlock_analysis_reporting.json`; primary verdict
criteria and predictions are unchanged.

The primary causal comparison for the decision gate is `matched_verified`:
same detector, labels, requests, inspector evolution, weights and update cadence.
This comparator still uses ML2 for sample acquisition; it does not remove every
role of ML2. `frozen_detector` freezes ML1 and its output threshold while sharing
ML2-based acquisition. `budget_full_history` changes source memory and regular
cadence; the 100-percent-label reference has a different annotation budget and
is not a mathematically optimal upper bound. See protocol files for exact roles.

## Human review and fixed inference

```bash
python human_workflow.py new --dataset wednesday --seed 11 --session human.pkl --until 100
python human_workflow.py export --session human.pkl --csv reviews.csv
python human_workflow.py selftest
```

Fill `human_label` only after independently inspecting evidence. A transformed
flow row may not provide sufficient incident context. The self-test uses clearly
marked synthetic labels and does not measure human annotation accuracy.
See `HUMAN_FEEDBACK_GUIDE.txt` for correction, availability and session resumption.

The example model is one final Botnet seed-11 delay80 run trained with SIMULATED
feedback, not an aggregate of all experiments. Fixed CSV inference:

```bash
python inference.py --snapshot model_examples/bot_main_11_delay80_simulated_snapshot.pkl --transformer deployment/transform_seed_11.json --input-csv deployment/example_raw_features.csv --output predictions.csv
```

The input must match the same 82 corrected flow-feature definitions and units.
This command does not extract features from PCAPs or retrain on predictions.

## Data provenance and integrity

Source: corrected CICIDS2017, with later corrected CSE-CICIDS2018 capture cohorts.
Original author documentation and downloads:

- https://intrusion-detection.distrinet-research.be/WTMC2021/
- https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html
- https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html

Attempted flows are mapped to benign following the authors' guidance. Addresses,
ports, labels and timestamps are excluded from model predictors. Prepared arrays,
source split identities, deduplication receipts and source-only transformations
are supplied for exact replay. Data rights remain with the original authors;
the code license does not relicense the underlying datasets.

The 2018 archive was streamed by selected ZIP members, not fully downloaded.
Member CRCs and decompressed SHA256 values are recorded. No full 2018 archive
SHA256 is claimed. An acquisition merge lost exact aggregate HTTPS counters;
the receipt marks them unknown. An S2 retention probe used by a smoke test was
archived and replaced by a group/fingerprint-disjoint S3 probe before confirmation.

`protocol/confirmation_lock.json` hashes model sources and inputs before external
prediction. Development data are explicitly not untouched tests. `artifact_manifest.json`
checks release assets. Scores after threshold shift are decision scores, not
calibrated posterior probabilities. Stored ledgers retain all observations, so
capped training memory does not imply constant overall session storage.

## License

Project implementation code is MIT licensed. Dataset and third-party rights
remain with their respective authors. No external paper's native result is
represented as a locally rerun baseline or proof of global algorithmic novelty.

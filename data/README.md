# Data provenance

Nothing in `data/raw/` or `data/processed/` is committed to this repository.
This file records where the data came from, so the experiment can be reproduced
from the original source rather than from a copy hosted here.

## Dataset

**CHAOS - Combined (CT-MR) Healthy Abdominal Organ Segmentation Challenge
Data.** Only the **CT** portion is used by this project. The MR portion is not
downloaded and not used.

| Field | Value |
| --- | --- |
| Official challenge site | <https://chaos.grand-challenge.org/> |
| Data repository record | <https://zenodo.org/records/3362845> |
| Version DOI | [10.5281/zenodo.3362845](https://doi.org/10.5281/zenodo.3362845) |
| Concept DOI | [10.5281/zenodo.3362844](https://doi.org/10.5281/zenodo.3362844) |
| Zenodo publication date | 2019-04-11 |
| Zenodo record created | 2019-08-08 |
| License (Zenodo record metadata) | `cc-by-nc-sa-4.0`, Creative Commons Attribution Non Commercial Share Alike 4.0 International |
| Challenge event | IEEE International Symposium on Biomedical Imaging (ISBI), 11 April 2019, Venice, Italy |
| Accessed for this project | 2026-09-10 |

Provenance fields above were read from the Zenodo REST API record
(`https://zenodo.org/api/records/3362845`) and from the official challenge
site, not from secondary summaries.

### Public availability before Summer 2025

The Zenodo record carries a publication date of 2019-04-11 and was created on
2019-08-08, and the challenge itself was held at ISBI in April 2019. The
dataset was therefore publicly available for roughly six years before Summer
2025. The challenge description paper, Kavur et al., "CHAOS Challenge -
combined (CT-MR) healthy abdominal organ segmentation", *Medical Image
Analysis* vol. 69, 2021, predates Summer 2025 as well.

This repository is being written now. The only historical claim being made is
that this dataset was already publicly available and could have supported
independent medical-imaging exploration by Summer 2025. No script, metric or
output in this repository was created in 2025.

### Access mechanism

The archives are downloaded directly over anonymous HTTPS from the Zenodo
record. No account, credential, or terms click-through was needed. Registering
on grand-challenge.org is required only to submit results to the challenge
leaderboard, which this project does not do.

## Archives

Both archives on the record contain CT **and** MR data mixed together.

Both archives on the record mix CT and MR together, and **both are used**, for
their CT content only.

| Archive | Size | MD5 (published, and verified locally) | CT subjects |
| --- | --- | --- | --- |
| `CHAOS_Train_Sets.zip` | 890,778,407 bytes (0.89 GB) | `3f6c9f1dbb95e91cb807b44bf4aceba9` | 20 |
| `CHAOS_Test_Sets.zip` | 1,091,803,676 bytes (1.09 GB) | `da6efc17118d7375654ab268473a5555` | 20 |

Reading each ZIP central directory remotely, before downloading, showed CT to
be 88.8 and 87.9 percent of the respective compressed archives. Fetching only
the CT members through HTTP range requests would therefore have saved about a
tenth of the transfer while adding a custom extraction path, so each archive
was downloaded whole and checked against its published MD5 instead.

`Train_Sets` and `Test_Sets` are the **challenge's** split of its own
segmentation task. This project does not perform segmentation and does not use
the masks, so the absence of public ground truth in `Test_Sets` costs it
nothing, while those 20 CT cases double the number of independent patients
available for a patient-level split. The two trees are read as one pool of
reference CT studies. Each subject keeps a record of the archive it came from,
as provenance only.

**This project's own train/validation/test split is a separate decision, is not
yet made, and does not reuse the challenge's split.** The expanded 40-subject
cohort supports a substantially stronger patient-level split than 20 subjects
would; the exact assignment will be designed in the next milestone, because the
observed acquisition parameter groups hold 16, 21, 2 and 1 subjects and the two
rare groups cannot be represented in every split.

The CT subject folder names are disjoint between the archives, together
covering 1 to 40 with no repeats, and the two `definitions.txt` files in the
archives state the same lists. Discovery refuses to pool a subject id that
appears in both trees, because one patient under two identities could end up on
both sides of a split.

### What was extracted

Downloaded on 2026-09-10 and 2026-09-12, each verified against the record's
published MD5 before use; both matched. Only the `CT/**` subtrees plus the
small archive text files were extracted into `data/raw/chaos/`, 5811 of 9180
members from the training archive and 3575 of 5668 from the test archive. The
5462 MR members were left inside the archives.

Measured layout:

```
data/raw/chaos/
    CHAOS_Train_Sets.zip                  verified archives, kept for re-extraction
    CHAOS_Test_Sets.zip
    Train_Sets/
        NOTES_PLEASE_READ.txt
        definitions.txt                   the official CT/MR train case ids
        CT/<subject id>/
            DICOM_anon/*.dcm              the CT slices
            Ground/*.png                  liver masks, not used by this project
    Test_Sets/
        definitions.txt                   the official CT/MR test case ids
        CT/<subject id>/
            DICOM_anon/*.dcm              the CT slices, no public masks
```

| Measured fact | Value |
| --- | --- |
| CT subjects | 40, pooled from both archives |
| Subject ids, `Train_Sets` | 1, 2, 5, 6, 8, 10, 14, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 |
| Subject ids, `Test_Sets` | 3, 4, 7, 9, 11, 12, 13, 15, 17, 20, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 |
| CT DICOM slices | 6407, of which 2874 train and 3533 test |
| Distinct series / studies | 40 / 40, exactly one series per subject |
| Slices per subject | 78 to 294 |
| Liver mask PNGs | 2874, train subjects only, unused here |
| Slices failing HU conversion | 0 |

The subject ids match each archive's own `definitions.txt` exactly. Full
measured metadata is in `outputs/audit/`, produced by `scripts/audit_chaos.py`,
which can be re-run after any fresh download. Both archives are kept on disk
after extraction, so the CT trees can be rebuilt without downloading again;
together they occupy about 5.1 GB.

## Important characteristics

**CHAOS is reference CT data. It is not paired low-dose data.** The archive
contains one acquisition per subject. Every degraded image in this project is
produced by synthetic, image-domain degradation applied by this repository. No
real low-dose acquisition is involved, and no claim about real low-dose CT
reconstruction follows from results computed on it.

The CT series are contrast-enhanced upper-abdomen scans acquired in the portal
venous phase, from potential liver donors described by the challenge as having
healthy livers.

## Handling rules

* Raw DICOM files, extracted archives, and derived image arrays are excluded
  from Git by `.gitignore`.
* This data is not re-hosted. Reproducing the work means downloading it from
  the Zenodo record above under its own license terms.
* **Dataset licensing and source-code licensing are separate questions.** Use
  or redistribution of CHAOS data and of CHAOS-derived artifacts is subject to
  the CHAOS dataset license, which is non-commercial and share-alike. That
  license governs the data and things derived from it; it is not the license of
  this repository's source code, and nothing here places the code under it.
* The files in `outputs/audit/` are derived technical summaries computed from
  CHAOS data. They contain measurements and counts, no pixel data and no
  patient-identifying elements, but they are CHAOS-derived artifacts and the
  dataset license applies to them.
* Subjects are identified by the CHAOS folder name, which is the dataset's own
  anonymized case label. No DICOM patient attribute is read for identity, and
  no patient-identifying element is written into any audit output.
* Audit figures rendered from the images stay in `outputs/audit/figures/`,
  which is git-ignored, pending a decision about publishing dataset-derived
  images under the share-alike license.

# Releases and Zenodo archival

The package version `0.1.0rc1` corresponds to GitHub tag `v0.1.0rc1`.
The [GitHub releases page](https://github.com/DrCanD/spikescr-local-event-geometry/releases)
is the authoritative source for publication status.

## Repository metadata

Use the following description in GitHub's **About** settings:

> Code, frozen checkpoint and data for local event geometry and activation-replacement analysis of SpikeSCR on Spiking Speech Commands.

Suggested topics:

`spiking-neural-networks`, `speech-recognition`, `spiking-speech-commands`,
`spikescr`, `neuromorphic-computing`, `activation-replacement`,
`reproducible-research`

After Zenodo has created the archival record, use its DOI landing-page URL for
the repository's website field.

## Publish in this order

1. **Check the license scope.** Original project software uses MIT; research
   artifacts, documentation and model weights use CC BY 4.0 within the author's
   rights. Keep [LICENSE_NOTICE.md](../LICENSE_NOTICE.md), citation metadata and
   the [dataset and upstream terms](THIRD_PARTY_NOTICES.md) consistent.
2. **Connect GitHub to Zenodo and enable this repository.** In Zenodo's profile
   menu, open **GitHub**, synchronize repositories if needed, and enable
   `DrCanD/spikescr-local-event-geometry`.
3. **Finalize the release commit.** Keep `CITATION.cff` and `pyproject.toml`
   versions consistent, validate citation metadata, and refresh both checksum
   files after any tracked-file change. Require a passing file-integrity check
   and the repository's two CI jobs for the selected commit.
4. **Publish the GitHub release.** Create the tag from the verified commit and
   use the release notes below. For `v0.1.0rc1`, mark the release as a pre-release.
   A saved draft is preparation only; publish after the Zenodo repository
   integration is enabled and the metadata is final.
5. **Verify the Zenodo record.** Wait for processing, check that the archived
   version, files, author, license and citation metadata are correct, and retain
   the assigned DOI.
6. **Add the DOI links.** Add the DOI to `CITATION.cff`, a DOI badge to the README,
   and the DOI landing-page URL to the About website field. Refresh checksums
   for this follow-up documentation commit. Keep the published release tag
   fixed; later changes belong to subsequent commits or releases.

`CITATION.cff` supplies the citation metadata. An additional `.zenodo.json` is
unnecessary unless Zenodo-specific fields are needed; when both files exist,
Zenodo gives `.zenodo.json` precedence.

Official guidance: [enable a repository](https://help.zenodo.org/docs/github/enable-repository/),
[archive a GitHub release](https://help.zenodo.org/docs/github/archive-software/github-upload/),
[citation metadata](https://help.zenodo.org/docs/github/describe-software/citation-file/),
and [licenses](https://help.zenodo.org/docs/deposit/describe-records/licenses/).

## Release notes for v0.1.0rc1

**Title:** `v0.1.0rc1 — Technical reproduction package`

This release candidate accompanies *Stable predictions can hide substantial
temporal changes in a high-accuracy spiking speech classifier* by İsmail Can
Dikmen. It supports reproduction of the technical experiment at the supplied
frozen SpikeSCR checkpoint on Spiking Speech Commands.

### Included

- Frozen checkpoint, transformed 100-utterance validation panel, clean scores
  and recorded benchmark predictions.
- All 725,070 candidate records, six activation-change metrics at seven network
  boundaries, and 180,740 complete 35-class activation-replacement score vectors.
- Statistical regeneration, model-replay tools, a hash-verified upstream-source
  preparation script, pinned dependencies and numerical conformance checks.
- Data schemas, provenance, reproduction instructions, SHA-256 manifests and
  recorded validation results.

### Licenses

Original project software is licensed under MIT. Project-authored research
outputs, documentation and the frozen model weights are licensed under
CC BY 4.0, within the author's rights. The transformed SSC panel retains the
dataset's CC BY 4.0 attribution; upstream source and dependencies retain their
own terms. See `LICENSE_NOTICE.md` for the file-level scope.

### Validation scope

Recorded-result integrity and statistical regeneration checks pass. The recorded
real-model CPU preflight covers four fixed probes and 64 forward passes, with a
resume check. A full conformance report for all 725,070 candidates in the
reference CUDA environment is not yet included. Full CPU and cross-device
equivalence have not been established; see the validation notes for the scope
of recorded diagnostics.

The package includes candidate predictions and scalar score descriptors, but
not every full unpatched candidate score vector or raw hidden-state tensor.
Clean and activation-replacement score vectors are included in full. Raw SSC
event files are obtained separately for the workflows that require them.

### Getting started

Follow the repository README's **Quick start** to verify the files and
recalculate the statistics. Follow **Model replay** and the reproduction protocol
to execute the frozen model. Preserve the release tag and commit identifier
with each reproduction run.

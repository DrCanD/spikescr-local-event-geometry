# Releases

## Archived versions

| Version | Tag / commit | Archive |
| --- | --- | --- |
| 0.1.0rc1 | `v0.1.0rc1`, `v1.0` → `a225e8b` | [10.5281/zenodo.22936265](https://doi.org/10.5281/zenodo.22936265): checkpoint, panel, census, internal metrics, replacement scores and the first reproduction tools |
| 1.1.0 | `v1.1.0` → `ea15b24` | [10.5281/zenodo.23089436](https://doi.org/10.5281/zenodo.23089436) (2026-10-01): the manuscript submission package — this layout, the search, replica and execution-control records, and the `paper` cross-check. The archive was downloaded and checked: all 104 file hashes match `provenance/SHA256SUMS.txt`, and `paper` reproduces the same `report.json` (434 cells, 0 failures). |

The concept DOI [10.5281/zenodo.22936264](https://doi.org/10.5281/zenodo.22936264) resolves to the latest version.
Existing tags stay fixed; a new release uses a new tag at its verified commit. `provenance/records.json` maps every
file of the 0.1.0rc1 archive to its current name and hash, so the two archives can be compared file by file.

## Release checklist

1. `python scripts/update_manifest.py` after the last content change; commit `provenance/manifest.json` and
   `provenance/SHA256SUMS.txt`.
2. `python -m ssc_geometry verify`, `python -m unittest discover -s tests` and `python -m ssc_geometry paper --out outputs/release`
   on a clean clone: 0 missing/modified/unexpected files, all tests passing, `report.json` with 0 failed cells.
   Run the same on a second clone at a different path and confirm that `report.json` is byte-identical.
3. Both GitHub Actions jobs green on the release commit (analysis job and CPU model preflight).
4. `pyproject.toml`, `src/ssc_geometry/__init__.py`, `CITATION.cff` and `.zenodo.json` carry the same version. Zenodo
   reads `.zenodo.json` for the archive's metadata (title, author with ORCID and affiliation, description, license,
   keywords); `CITATION.cff` serves the GitHub citation box.
5. Tag (`git tag -a v1.1.0`) and publish the GitHub release (not a draft). The repository's GitHub–Zenodo link archives
   every published release automatically under the concept DOI, usually within minutes; nothing is uploaded by hand.
   Then write the version DOI into `CITATION.cff` and `docs/release.md` in a follow-up commit.
6. In the manuscript's data-availability statement cite the version DOI, the concept DOI and the commit.

## What a release does not contain

The raw SSC event files, the SpikeSCR source, the raw hidden tensors and unpatched score vectors of the census, the
full all-target gradient tensors, the replica weights and raw candidate maps, and the per-source arrays of the padding
decomposition and pair controls; see the README section "What is not included".

## Licenses

Project software: [MIT](../LICENSE). Project-authored research outputs, documentation and the trained checkpoint:
[CC BY 4.0](../LICENSES/CC-BY-4.0.txt), within the author's rights. SSC attribution and upstream software terms
remain in force; see [LICENSE_NOTICE.md](../LICENSE_NOTICE.md) and [third_party.md](third_party.md).

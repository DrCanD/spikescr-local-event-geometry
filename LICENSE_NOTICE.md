# Licenses and scope

Copyright (c) 2026 İsmail Can Dikmen for original project contributions.

This repository contains separately licensed software and research artifacts.
The licenses apply by file category as follows; they are not alternative
licenses that can be selected for every file.

| Material | Paths | License |
| --- | --- | --- |
| Original project source code, scripts, tests, configuration and build/CI files | `src/`, `scripts/`, `tests/`, `configs/`, `.github/`, `pyproject.toml`, `requirements-*.txt`, `.gitattributes`, `.gitignore` | [MIT](LICENSE) |
| Project-trained frozen model weights | `data/model/frozen_checkpoint.pt` | [CC BY 4.0](LICENSES/CC-BY-4.0.txt), for the rights held by the project author |
| Project-authored research outputs, transformed panel, provenance, documentation, citation metadata and validation records | `data/` other than the checkpoint above, `docs/`, `validation/`, `checksums/`, `README.md`, `CITATION.cff`, this notice | [CC BY 4.0](LICENSES/CC-BY-4.0.txt), subject to the retained dataset attribution below |

## Attribution

For the project-authored CC BY 4.0 material, credit İsmail Can Dikmen and
*Local event geometry in a frozen spiking speech classifier*, link to this
repository and the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/),
and indicate any changes. Retain the supplied third-party attributions when
sharing material derived from the dataset.

Repository: https://github.com/DrCanD/spikescr-local-event-geometry

For software, retain the copyright and permission notices required by MIT.
The requested academic citation in `CITATION.cff` does not add conditions to MIT.

## Third-party material

The Spiking Speech Commands panel derives from the dataset by Benjamin Cramer,
Yannik Stradmann, Johannes Schemmel and Friedemann Zenke, released under
CC BY 4.0. Their attribution and the documented transformations remain part of
the distributed panel's provenance.

Separately downloaded SpikeSCR source and other dependencies retain their own
terms. The project's MIT and CC BY 4.0 grants do not relicense third-party
software or grant rights the project author does not hold. See
[third-party notices](docs/THIRD_PARTY_NOTICES.md) for source references and scope.

The license texts in `LICENSE` and `LICENSES/` retain their original terms.

# Sources and permissions

## Spiking Speech Commands

The panel is derived from the public Spiking Speech Commands dataset released with the Heidelberg Spiking Data Sets. Attribution belongs to Benjamin Cramer, Yannik Stradmann, Johannes Schemmel and Friedemann Zenke. The dataset page states a Creative Commons Attribution 4.0 license.

Source: https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/

Reference: Cramer et al., The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks, IEEE Transactions on Neural Networks and Learning Systems, 2022. DOI 10.1109/TNNLS.2020.3044364.

This package distributes only the transformed 100-source validation panel, with source indices and hashes. It does not distribute raw audio or the complete raw train, validation or test event files. Its transform, grouping, horizon convention and retained count units are documented in the source and reproduction protocol.

## SpikeSCR

The upstream model is by Jiaqi Wang and colleagues, Efficient Speech Command Recognition Leveraging Spiking Neural Networks and Progressive Time-Scaled Curriculum Distillation, Neural Networks 195, 108253, 2026. DOI 10.1016/j.neunet.2025.108253.

Source: https://github.com/JackieWang9811/SpikeSCR

Pinned commit: 095f418f53b3b24c21caf558225c65ad674d44b1.

The model source is not bundled here. A separate program retrieves the required files and verifies their recorded Git blob hashes. This package does not assign an upstream license, assert additional redistribution permissions, or claim that public visibility alone grants them. Respect the upstream authors' terms.

## Project-specific artifacts

The checkpoint was trained for the reported reproduction. Its bytes and tensor-state digest are retained. The study's recorded numerical outputs are renamed without changing their contents, except for documented descriptive metadata and the boolean representation in a CSV.

The package intentionally does not select a new public license on the author's behalf. A project-code and checkpoint distribution license must be confirmed before public release. The provided publication helper creates a private repository and does not apply a license template automatically.

# The checkpoint

The analysis target is the released checkpoint (`data/model/checkpoint.pt`), not a distribution over independently
trained models. The file is the selected training output as saved; its stored zero-based epoch is 281 (epoch 282 in
the manuscript). `configs/experiment.json` holds its file hash and tensor-state digest; `python -m ssc_geometry checkpoint`
checks both and counts 3,302,416 parameters (3,302,400 trainable, 16 fixed rotary frequencies).

Training setup (Supplementary S1): seed 312, 300 epochs, AdamW at learning rate 0.005 with weight decay 0.01, cosine
schedule with T_max 40, the upstream time-neuron masking augmentation, 5 ms input bins (the pinned public
configuration defaults to 10 ms), five adjacent input channels combined into one of 140 features, two blocks of width
256 with 16 attention heads and local kernel size 31, batch size 256.

The upstream file `Training/main_former_v2_ssc_spikescr.py` is one of the eight hash-checked files that
`scripts/prepare_upstream.py` downloads; it documents the public training implementation but is not a drop-in
reproduction of the wrappers, preprocessing cache and random-number state used to obtain this checkpoint. No
retraining was run for this repository, and no tested from-scratch training command is included; independent
training reproducibility should be reported separately from the fixed-checkpoint audit.

The checkpoint was optimised with batch size 256 through the public q/k reshape path, so its weights belong to that
coupled function. Native singleton evaluation (the audit path) and the q/k source-isolation diagnostic are their own
execution contracts (Appendix A); a conventional head/time layout would be an unevaluated third function.

The official test split was evaluated once, with batch size 256, for Table 2; it was not used for training,
checkpoint selection or tuning, and no command in this repository reads it.

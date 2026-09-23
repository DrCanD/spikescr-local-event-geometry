# Recorded training setup

The analysis target is the released checkpoint, not a distribution over independently trained models. The checkpoint file is byte-identical to the selected training output. Its stored zero-based epoch is 281, corresponding to epoch 282 in the manuscript.

The recorded setup uses seed 312, 300 epochs, AdamW at learning rate 0.005 and weight decay 0.01, cosine scheduling with T_max 40, and the upstream time-neuron masking augmentation. The input override is 5 ms instead of the pinned public configuration's 10 ms default. Five adjacent input channels are combined into one of 140 features. The network uses two blocks, width 256, 16 attention heads and local kernel size 31.

The upstream source file `Training/main_former_v2_ssc_spikescr.py` is one of the eight hash-checked files downloaded by the source preparation program. It documents the public training implementation. It is not presented as a drop-in reproduction of every historical wrapper, preprocessing cache and random-number state used to obtain this checkpoint.

No clean-room retraining experiment was run during packaging. This release does not include a tested from-scratch training command. Independent training reproducibility should be reported separately from the complete fixed-checkpoint neighborhood audit. The official test split must not be used for training, checkpoint selection or tuning.

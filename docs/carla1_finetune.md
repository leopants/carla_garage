# Fine-tuning TransFuser++ for CARLA 1.0

The released TF++ checkpoints were trained on CARLA 0.9.x (UE4) and drive poorly in CARLA 1.0 (UE5). A short fine-tune on
PDM-Lite expert data collected in CARLA 1.0 recovers most of the lost driving score.

## Results

Driving score, Leaderboard 2.0 metric, TensorRT inference, Low quality.

| Model | Fine-tune data | Town10 (15 runs) | Town15, held out (10 runs) |
|---|---|---|---|
| `all_towns/model_0030_0.pth` | none | 27.1 | 46.8 |
| Fine-tuned | Town10, 3,078 frames | 77.2 | 49.8 |
| **Fine-tuned** | **Town10 + Town12, 8,442 frames** | 70.3 | **72.5** |
| PDM-Lite expert | - | 93.9 | - |

- Town10: 5 routes × 3 seeds. Town15: 5 Bench2Drive routes × 2 seeds; Town15 was never used for training.
- The Town10 test routes share roads with the Town10 data, so Town15 is the honest generalization test: Town10 data alone
  barely transfers (+3), while adding Town12 data gives +26.
- Run-to-run noise is large. Rerunning the same model on the same 30 routes and seeds moved the mean by 6 points (90% CI up to
  17), so differences under about 10 points on this many runs are not meaningful.

## Recipe

1. Collect data with the PDM-Lite expert (`leaderboard_autopilot`) in CARLA 1.0, as for the original training data. We used
   7 Town10 routes with different weathers and 75 Town12 routes from `data/50x38_Town12`, balanced over scenario types.
2. Fine-tune with `train.py`, starting from the `all_towns` arguments (`args.txt`) with these changes:
   - `--load_file pretrained_models/all_towns/model_0030_0.pth --continue_epoch 0`
   - `--epochs 3 --lr 3e-5 --batch_size 2 --use_amp 0 --use_color_aug 0`
   - `--schedule_reduce_epoch_01 100 --schedule_reduce_epoch_02 100` (no learning-rate drop)
   - environment variable `FREEZE_BN=1`, which keeps the pretrained BatchNorm statistics (batch size 2 is too small to
     re-estimate them).

   On an 8 GB GPU this takes about 18 minutes per epoch for 8,442 frames.
3. Evaluate the last checkpoint with `sensor_agent.py` as usual (the agent's config is unchanged).

## Code changes

- `team_code/data.py`: the semantic converter is padded to 256 ids, because CARLA 1.0 maps emit ids beyond the 29 known
  ones (id 29 on Town12); unknown ids become unlabeled. The image and LiDAR augmenters are skipped when `imgaug` is not
  installed.
- `team_code/train.py`: `FREEZE_BN=1` puts every BatchNorm layer in eval mode during training (affine parameters still train).

## What did not help

Tried on top of the recipe; none beat it on Town15 beyond the noise above:
- **Camera augmentation:** 70.6.
- **RL fine-tuning of the speed head** against a log-replay verifier: 60.3. Replayed traffic does not react, so the policy
  learned to push through junctions.
- **JEPA-style latent prediction loss** (predict the BEV features 1 s ahead): 68.4. It learned the dynamics but only
  shapes features, not the driving decision.

The fine-tuned model's main remaining failure on unseen routes is freezing behind or yielding to other actors, not
collisions.

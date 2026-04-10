# Kitchen Object Pose Generation Manual

This manual explains how to use `scripts/generate_object_poses_kitchen.py` for the kitchen stacking task, especially when you want to control how much the cup positions vary around the ArUCO tag.

## What The Script Generates

The script writes a new `object_poses.json`-compatible file for the two-cup kitchen task.

- It keeps the existing JSON schema used by the loader and replay-buffer code.
- It preserves each sampled template episode's metadata, object `rvec`, and cup `z` values.
- It only changes the cups' XY positions in the ArUCO-tag frame.

The most important mode is:

- `anchored-relative`
  - Keep one cup near a nominal XY location.
  - Place the other cup relative to it with a chosen distance and angle.
  - This is the recommended mode for fixed-grasp experiments.

The older mode is:

- `template-distance`
  - Keep each template episode's midpoint and direction.
  - Only enforce the cup-to-cup distance.

## Coordinate Frame

All XY parameters in this script are in the same ArUCO-tag frame used in `video/example_kitchen/demos/mapping/object_poses.json`.

They are not world-frame positions from Isaac Sim.

From the current reference data:

- empirical `pink_cup` mean XY is about `(-0.027, -0.109)`
- empirical `blue_cup` mean XY is about `(0.004, -0.095)`
- empirical cup distance mean/std is about `0.053 / 0.039 m`

If you omit `--anchor-x` and `--anchor-y`, the script uses the empirical mean of the chosen anchor cup from the reference file.

## Recommended Mental Model

For a fixed-grasp experiment, use `pink` as the anchor:

- keep `pink` nearly fixed
- move `blue` around `pink`
- keep orientations fixed
- keep `z` fixed

This isolates pick generalization better than sampling both cups independently.

If you vary both cups a lot at the same time, you mix:

- pick difficulty
- place difficulty
- pair-distance variance
- workspace drift

That makes the learning result harder to interpret.

## Main Parameters

### `--mode`

- `anchored-relative`
  - Best for studying position variance with a fixed grasp setup.
- `template-distance`
  - Best if you want to stay close to the original example data structure.

### `--mean-distance` or `--distance-mean`

The average XY distance between the two cups in meters.

- Increase it to separate the cups more.
- Keep it fixed if you want to isolate angle or anchor-position variance.

### `--distance-std`

The standard deviation of the cup-to-cup distance in meters.

- `0.0` means every sample has the same distance.
- Higher values add radial variation.
- If this gets too large relative to `--distance-mean`, the sampled distribution becomes truncated because distance must stay positive.

Use this carefully if you want a clean ablation.

### `--anchor-object`

Which cup stays near the nominal location in `anchored-relative` mode.

- `pink` is recommended for the kitchen task.
- `blue` is only useful if you want to anchor the source cup instead.

### `--anchor-x`, `--anchor-y`

Nominal XY location of the anchor cup in the tag frame.

- If omitted, the script uses the empirical mean from the reference data.
- This is usually the right default.

### `--anchor-std-x`, `--anchor-std-y`

How much the anchor cup itself moves.

- Small values keep the placement target stable.
- Larger values introduce workspace drift and therefore place variance.

Rule of thumb:

- `0.003 to 0.005 m`: nearly fixed target
- `0.005 to 0.010 m`: moderate target drift
- `0.010 to 0.020 m`: high target drift

### `--angle-min-deg`, `--angle-max-deg`

The angular arc for placing the non-anchor cup around the anchor cup.

- Narrow arc: low positional variance
- Wide arc: high positional variance

This is often the cleanest way to increase source-cup position variance while keeping distance fixed.

Rule of thumb:

- `[-20, 20]`: low variance
- `[-90, 90]`: medium variance
- `[-150, 150]`: high variance

## How To Set Values For Different Experiments

### 1. Fixed grasp, low position variance

Use this as a baseline.

```bash
python scripts/generate_object_poses_kitchen.py \
  --num-entries 200 \
  --mode anchored-relative \
  --anchor-object pink \
  --distance-mean 0.08 \
  --distance-std 0.0 \
  --anchor-std-x 0.003 \
  --anchor-std-y 0.003 \
  --angle-min-deg -20 \
  --angle-max-deg 20 \
  --seed 0
```

What this means:

- pink is almost fixed
- blue stays about `8 cm` away
- blue only appears in a small arc around pink

This mostly tests whether the policy can learn a nearly nominal setup.

### 2. Fixed grasp, medium source position variance

```bash
python scripts/generate_object_poses_kitchen.py \
  --num-entries 200 \
  --mode anchored-relative \
  --anchor-object pink \
  --distance-mean 0.08 \
  --distance-std 0.0 \
  --anchor-std-x 0.003 \
  --anchor-std-y 0.003 \
  --angle-min-deg -90 \
  --angle-max-deg 90 \
  --seed 1
```

What changed:

- target is still stable
- distance is still fixed
- only the blue cup angle varies much more

This is a good next step if you want to test translation and viewpoint generalization without changing task scale.

### 3. Fixed grasp, high source position variance

```bash
python scripts/generate_object_poses_kitchen.py \
  --num-entries 200 \
  --mode anchored-relative \
  --anchor-object pink \
  --distance-mean 0.08 \
  --distance-std 0.0 \
  --anchor-std-x 0.003 \
  --anchor-std-y 0.003 \
  --angle-min-deg -150 \
  --angle-max-deg 150 \
  --seed 2
```

Why this is useful:

- grasp pose relative to the blue cup can remain conceptually fixed
- blue appears over a much wider set of XY locations
- pink stays stable enough that you are still mostly studying pick generalization

If the policy fails here but succeeds in the low-variance setup, that is a much cleaner result than varying both cups heavily.

### 4. Add distance variance on top of position variance

```bash
python scripts/generate_object_poses_kitchen.py \
  --num-entries 200 \
  --mode anchored-relative \
  --anchor-object pink \
  --distance-mean 0.08 \
  --distance-std 0.015 \
  --anchor-std-x 0.003 \
  --anchor-std-y 0.003 \
  --angle-min-deg -120 \
  --angle-max-deg 120 \
  --seed 3
```

Now you are no longer only testing XY position variation.

You are also testing:

- how much cup separation changes
- whether the policy is sensitive to different source-target distances

Use this only after the fixed-distance experiments, otherwise the result is harder to interpret.

### 5. High full-scene variance

```bash
python scripts/generate_object_poses_kitchen.py \
  --num-entries 200 \
  --mode anchored-relative \
  --anchor-object pink \
  --distance-mean 0.08 \
  --distance-std 0.015 \
  --anchor-std-x 0.015 \
  --anchor-std-y 0.010 \
  --angle-min-deg -150 \
  --angle-max-deg 150 \
  --seed 4
```

This adds both:

- source-cup position variance
- target-cup drift

Use this only if you want a harder, less isolated study.

## Recommended Experiment Order

If your question is:

"With a fixed grasp pose, will the policy still learn when cup positions vary a lot?"

then a good progression is:

1. low angle variance, fixed distance
2. medium angle variance, fixed distance
3. high angle variance, fixed distance
4. high angle variance plus distance variance
5. high angle variance plus anchor drift

That order helps you identify which kind of variance actually breaks learning.

## How To Read The Script Output

After generation, the script prints summary statistics such as:

- `Distance mean/std`
- `Blue XY mean/std`
- `Pink XY mean/std`
- `Anchor XY mean/std`
- `Relative angle min/mean/max`

Use these to verify that the dataset you generated matches the experiment you intended.

Examples:

- If `Anchor XY std` is large, you are drifting the target a lot.
- If `Distance std` is near zero, separation is effectively fixed.
- If `Relative angle min/max` spans a wide range, source position variance is high.

## Practical Recommendations

- For fixed-grasp studies, vary angle first.
- Keep `distance-std = 0.0` first.
- Keep `anchor-std-x` and `anchor-std-y` small first.
- Do not vary cup orientation until you finish the position-variance experiments.

The cleanest first high-variance experiment is:

```bash
python scripts/generate_object_poses_kitchen.py \
  --num-entries 200 \
  --mode anchored-relative \
  --anchor-object pink \
  --distance-mean 0.08 \
  --distance-std 0.0 \
  --anchor-std-x 0.003 \
  --anchor-std-y 0.003 \
  --angle-min-deg -150 \
  --angle-max-deg 150 \
  --seed 42
```

That keeps the target stable, keeps distance fixed, and makes the source cup appear over a broad set of positions.

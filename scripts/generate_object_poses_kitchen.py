import argparse
import copy
import json
import math
import random
import statistics
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_PATH = (
    REPO_ROOT / "video" / "example_kitchen" / "demos" / "mapping" / "object_poses.json"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive float")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be a non-negative float")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a kitchen object_poses.json variant. The recommended "
            "anchored-relative mode keeps one cup near a nominal location and "
            "samples the other cup relative to it, which is useful for fixed-"
            "grasp variance studies."
        )
    )
    parser.add_argument(
        "--num-entries",
        type=positive_int,
        required=True,
        help="Number of generated episode entries to write.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["anchored-relative", "template-distance"],
        default="anchored-relative",
        help=(
            "Generation strategy. anchored-relative is recommended for fixed-grasp "
            "experiments. template-distance preserves each template midpoint and "
            "only enforces pair distance."
        ),
    )
    parser.add_argument(
        "--mean-distance",
        "--distance-mean",
        dest="distance_mean",
        type=positive_float,
        required=True,
        help="Target mean XY distance between the two cups in meters.",
    )
    parser.add_argument(
        "--distance-std",
        type=non_negative_float,
        default=0.0,
        help=(
            "Standard deviation of the XY cup-to-cup distance in meters. "
            "Use 0.0 to keep distance fixed."
        ),
    )
    parser.add_argument(
        "--anchor-object",
        type=str,
        choices=["pink", "blue"],
        default="pink",
        help=(
            "Cup to keep near a nominal location in anchored-relative mode. "
            "Use pink if you want the placement target to stay stable."
        ),
    )
    parser.add_argument(
        "--anchor-x",
        type=float,
        default=None,
        help=(
            "Nominal anchor cup X position in the ArUCO-tag frame. "
            "Defaults to the empirical mean from the reference file."
        ),
    )
    parser.add_argument(
        "--anchor-y",
        type=float,
        default=None,
        help=(
            "Nominal anchor cup Y position in the ArUCO-tag frame. "
            "Defaults to the empirical mean from the reference file."
        ),
    )
    parser.add_argument(
        "--anchor-std-x",
        type=non_negative_float,
        default=0.005,
        help="Standard deviation for the anchor cup X position in meters.",
    )
    parser.add_argument(
        "--anchor-std-y",
        type=non_negative_float,
        default=0.005,
        help="Standard deviation for the anchor cup Y position in meters.",
    )
    parser.add_argument(
        "--angle-min-deg",
        type=float,
        default=-120.0,
        help=(
            "Minimum relative angle in degrees for the non-anchor cup in "
            "anchored-relative mode."
        ),
    )
    parser.add_argument(
        "--angle-max-deg",
        type=float,
        default=120.0,
        help=(
            "Maximum relative angle in degrees for the non-anchor cup in "
            "anchored-relative mode."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to a new file beside the reference data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible template sampling.",
    )
    args = parser.parse_args()

    if args.angle_min_deg > args.angle_max_deg:
        parser.error("--angle-min-deg must be less than or equal to --angle-max-deg")

    return args


def load_reference_entries(reference_path: Path) -> list[dict]:
    with reference_path.open("r") as f:
        data = json.load(f)

    episodes = data if isinstance(data, list) else [data]
    templates = []
    for episode in episodes:
        if episode.get("status") != "full":
            continue

        cup_names = {obj.get("object_name") for obj in episode.get("objects", [])}
        if {"blue_cup", "pink_cup"}.issubset(cup_names):
            templates.append(episode)

    if not templates:
        raise ValueError(
            f"No usable kitchen cup templates found in reference file: {reference_path}"
        )

    return templates


def sanitize_token(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")


def resolve_output_path(
    output: Path | None,
    mode: str,
    distance_mean: float,
    num_entries: int,
) -> Path:
    if output is not None:
        return output

    distance_token = sanitize_token(distance_mean)
    mode_token = "anchored" if mode == "anchored-relative" else "template"
    return DEFAULT_REFERENCE_PATH.parent / (
        f"object_poses_{mode_token}_n{num_entries}_dist{distance_token}.json"
    )


def get_object_indices(objects: list[dict]) -> tuple[int, int]:
    blue_idx = None
    pink_idx = None
    for idx, obj in enumerate(objects):
        if obj.get("object_name") == "blue_cup":
            blue_idx = idx
        elif obj.get("object_name") == "pink_cup":
            pink_idx = idx

    if blue_idx is None or pink_idx is None:
        raise ValueError("Template episode is missing blue_cup or pink_cup")

    return blue_idx, pink_idx


def compute_reference_means(templates: list[dict]) -> dict[str, tuple[float, float]]:
    blue_x = []
    blue_y = []
    pink_x = []
    pink_y = []

    for template in templates:
        blue_idx, pink_idx = get_object_indices(template["objects"])
        blue_tvec = template["objects"][blue_idx]["tvec"]
        pink_tvec = template["objects"][pink_idx]["tvec"]
        blue_x.append(float(blue_tvec[0]))
        blue_y.append(float(blue_tvec[1]))
        pink_x.append(float(pink_tvec[0]))
        pink_y.append(float(pink_tvec[1]))

    return {
        "blue": (statistics.mean(blue_x), statistics.mean(blue_y)),
        "pink": (statistics.mean(pink_x), statistics.mean(pink_y)),
    }


def sample_gaussian(mean: float, std: float, rng: random.Random) -> float:
    if std == 0.0:
        return mean
    return rng.gauss(mean, std)


def sample_positive_gaussian(mean: float, std: float, rng: random.Random) -> float:
    if std == 0.0:
        return mean

    for _ in range(1000):
        sampled = rng.gauss(mean, std)
        if sampled > 0.0:
            return sampled

    raise ValueError(
        "Unable to sample a positive distance. Reduce --distance-std or increase "
        "--distance-mean."
    )


def enforce_xy_distance(
    blue_tvec: list[float],
    pink_tvec: list[float],
    target_distance: float,
    rng: random.Random,
) -> tuple[list[float], list[float]]:
    blue_x, blue_y = float(blue_tvec[0]), float(blue_tvec[1])
    pink_x, pink_y = float(pink_tvec[0]), float(pink_tvec[1])
    midpoint_x = (blue_x + pink_x) / 2.0
    midpoint_y = (blue_y + pink_y) / 2.0

    dx = pink_x - blue_x
    dy = pink_y - blue_y
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        angle = rng.uniform(0.0, 2.0 * math.pi)
        unit_x = math.cos(angle)
        unit_y = math.sin(angle)
    else:
        unit_x = dx / norm
        unit_y = dy / norm

    half_distance = 0.5 * target_distance
    offset_x = half_distance * unit_x
    offset_y = half_distance * unit_y

    blue_new = [midpoint_x - offset_x, midpoint_y - offset_y, float(blue_tvec[2])]
    pink_new = [midpoint_x + offset_x, midpoint_y + offset_y, float(pink_tvec[2])]
    return blue_new, pink_new


def generate_template_distance_entries(
    templates: list[dict],
    num_entries: int,
    distance_mean: float,
    distance_std: float,
    rng: random.Random,
) -> list[dict]:
    generated = []
    for _ in range(num_entries):
        template = copy.deepcopy(rng.choice(templates))
        blue_idx, pink_idx = get_object_indices(template["objects"])
        blue_obj = template["objects"][blue_idx]
        pink_obj = template["objects"][pink_idx]

        target_distance = sample_positive_gaussian(distance_mean, distance_std, rng)
        blue_tvec, pink_tvec = enforce_xy_distance(
            blue_obj["tvec"], pink_obj["tvec"], target_distance, rng
        )
        blue_obj["tvec"] = blue_tvec
        pink_obj["tvec"] = pink_tvec
        generated.append(template)

    return generated


def generate_anchored_relative_entries(
    templates: list[dict],
    num_entries: int,
    distance_mean: float,
    distance_std: float,
    anchor_object: str,
    anchor_x: float,
    anchor_y: float,
    anchor_std_x: float,
    anchor_std_y: float,
    angle_min_deg: float,
    angle_max_deg: float,
    rng: random.Random,
) -> list[dict]:
    generated = []
    for _ in range(num_entries):
        template = copy.deepcopy(rng.choice(templates))
        blue_idx, pink_idx = get_object_indices(template["objects"])
        blue_obj = template["objects"][blue_idx]
        pink_obj = template["objects"][pink_idx]

        sampled_anchor_x = sample_gaussian(anchor_x, anchor_std_x, rng)
        sampled_anchor_y = sample_gaussian(anchor_y, anchor_std_y, rng)
        sampled_distance = sample_positive_gaussian(distance_mean, distance_std, rng)
        sampled_angle_deg = rng.uniform(angle_min_deg, angle_max_deg)
        sampled_angle_rad = math.radians(sampled_angle_deg)
        dx = sampled_distance * math.cos(sampled_angle_rad)
        dy = sampled_distance * math.sin(sampled_angle_rad)

        if anchor_object == "pink":
            pink_obj["tvec"] = [
                sampled_anchor_x,
                sampled_anchor_y,
                float(pink_obj["tvec"][2]),
            ]
            blue_obj["tvec"] = [
                sampled_anchor_x + dx,
                sampled_anchor_y + dy,
                float(blue_obj["tvec"][2]),
            ]
        else:
            blue_obj["tvec"] = [
                sampled_anchor_x,
                sampled_anchor_y,
                float(blue_obj["tvec"][2]),
            ]
            pink_obj["tvec"] = [
                sampled_anchor_x + dx,
                sampled_anchor_y + dy,
                float(pink_obj["tvec"][2]),
            ]

        generated.append(template)

    return generated


def describe_entries(entries: list[dict], anchor_object: str) -> dict[str, float]:
    blue_x = []
    blue_y = []
    pink_x = []
    pink_y = []
    distances = []
    relative_angles_deg = []

    for entry in entries:
        blue_idx, pink_idx = get_object_indices(entry["objects"])
        blue_tvec = entry["objects"][blue_idx]["tvec"]
        pink_tvec = entry["objects"][pink_idx]["tvec"]

        bx = float(blue_tvec[0])
        by = float(blue_tvec[1])
        px = float(pink_tvec[0])
        py = float(pink_tvec[1])

        blue_x.append(bx)
        blue_y.append(by)
        pink_x.append(px)
        pink_y.append(py)

        dx = bx - px
        dy = by - py
        distances.append(math.hypot(dx, dy))
        relative_angles_deg.append(math.degrees(math.atan2(dy, dx)))

    anchor_x_values = pink_x if anchor_object == "pink" else blue_x
    anchor_y_values = pink_y if anchor_object == "pink" else blue_y

    def mean(values: list[float]) -> float:
        return statistics.mean(values)

    def std(values: list[float]) -> float:
        return statistics.pstdev(values) if len(values) > 1 else 0.0

    return {
        "blue_x_mean": mean(blue_x),
        "blue_x_std": std(blue_x),
        "blue_y_mean": mean(blue_y),
        "blue_y_std": std(blue_y),
        "pink_x_mean": mean(pink_x),
        "pink_x_std": std(pink_x),
        "pink_y_mean": mean(pink_y),
        "pink_y_std": std(pink_y),
        "anchor_x_mean": mean(anchor_x_values),
        "anchor_x_std": std(anchor_x_values),
        "anchor_y_mean": mean(anchor_y_values),
        "anchor_y_std": std(anchor_y_values),
        "distance_mean": mean(distances),
        "distance_std": std(distances),
        "angle_min_deg": min(relative_angles_deg),
        "angle_max_deg": max(relative_angles_deg),
        "angle_mean_deg": mean(relative_angles_deg),
    }


def main() -> None:
    args = parse_args()
    reference_path = DEFAULT_REFERENCE_PATH
    output_path = resolve_output_path(
        args.output, args.mode, args.distance_mean, args.num_entries
    )
    rng = random.Random(args.seed)

    templates = load_reference_entries(reference_path)
    reference_means = compute_reference_means(templates)
    anchor_x = (
        args.anchor_x
        if args.anchor_x is not None
        else reference_means[args.anchor_object][0]
    )
    anchor_y = (
        args.anchor_y
        if args.anchor_y is not None
        else reference_means[args.anchor_object][1]
    )

    if args.mode == "anchored-relative":
        generated_entries = generate_anchored_relative_entries(
            templates=templates,
            num_entries=args.num_entries,
            distance_mean=args.distance_mean,
            distance_std=args.distance_std,
            anchor_object=args.anchor_object,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            anchor_std_x=args.anchor_std_x,
            anchor_std_y=args.anchor_std_y,
            angle_min_deg=args.angle_min_deg,
            angle_max_deg=args.angle_max_deg,
            rng=rng,
        )
    else:
        generated_entries = generate_template_distance_entries(
            templates=templates,
            num_entries=args.num_entries,
            distance_mean=args.distance_mean,
            distance_std=args.distance_std,
            rng=rng,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(generated_entries, f, indent=2)

    stats = describe_entries(generated_entries, args.anchor_object)
    print(f"Wrote {len(generated_entries)} entries to {output_path}")
    print(f"Reference templates: {reference_path}")
    print(f"Mode: {args.mode}")
    print(
        f"Distance mean/std: {stats['distance_mean']:.6f} / "
        f"{stats['distance_std']:.6f} m"
    )
    print(
        f"Blue XY mean/std: ({stats['blue_x_mean']:.6f}, {stats['blue_y_mean']:.6f}) / "
        f"({stats['blue_x_std']:.6f}, {stats['blue_y_std']:.6f})"
    )
    print(
        f"Pink XY mean/std: ({stats['pink_x_mean']:.6f}, {stats['pink_y_mean']:.6f}) / "
        f"({stats['pink_x_std']:.6f}, {stats['pink_y_std']:.6f})"
    )
    if args.mode == "anchored-relative":
        print(
            f"Anchor {args.anchor_object} nominal XY: ({anchor_x:.6f}, {anchor_y:.6f})"
        )
        print(
            f"Anchor XY mean/std: ({stats['anchor_x_mean']:.6f}, {stats['anchor_y_mean']:.6f}) / "
            f"({stats['anchor_x_std']:.6f}, {stats['anchor_y_std']:.6f})"
        )
        print(
            f"Relative angle min/mean/max: {stats['angle_min_deg']:.2f} / "
            f"{stats['angle_mean_deg']:.2f} / {stats['angle_max_deg']:.2f} deg"
        )
    if args.seed is not None:
        print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
Utility script to generate coordinate transformation matrix between ArUco tag frame and robot base frame.

This script helps create the transformation file needed by run_dataset_pose_publisher.py
to convert poses from ArUco tag coordinates to robot base coordinates.

Usage:
    python create_coordinate_transform.py --session_dir /path/to/session --output tx_robot_tag.json

The transformation matrix can be determined by:
1. Physical measurement of robot base relative to ArUco tag board
2. Using a calibration procedure where the robot touches known points on the tag board
3. Using the robot's forward kinematics with a known pose relative to the tag

Output format:
{
    "tx_robot_tag": [[4x4 transformation matrix]]
}

Where tx_robot_tag transforms points from ArUco tag frame to robot base frame:
    point_robot = tx_robot_tag @ point_tag
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def create_identity_transform():
    """Create an identity transformation matrix."""
    return np.eye(4)


def create_translation_transform(dx, dy, dz):
    """Create a translation transformation matrix."""
    transform = np.eye(4)
    transform[:3, 3] = [dx, dy, dz]
    return transform


def create_rotation_transform(rx, ry, rz, dx=0, dy=0, dz=0):
    """Create a rotation + translation transformation matrix (Euler angles in radians)."""
    transform = np.eye(4)

    # Create rotation matrix from Euler angles
    rotation = Rotation.from_euler('xyz', [rx, ry, rz])
    transform[:3, :3] = rotation.as_matrix()

    # Add translation
    transform[:3, 3] = [dx, dy, dz]
    return transform


def create_example_transform():
    """
    Create an example transformation matrix.

    This is a placeholder that should be replaced with actual calibration data.
    The example assumes:
    - Robot base is 0.5m in front of ArUco tag board
    - Robot base is 0.2m below the center of ArUco tag board
    - No rotation between frames
    """
    return create_translation_transform(0.5, 0.0, -0.2)


def save_transform_matrix(transform, output_path):
    """Save transformation matrix to JSON file."""
    transform_dict = {
        "tx_robot_tag": transform.tolist()
    }

    with open(output_path, 'w') as f:
        json.dump(transform_dict, f, indent=2)

    print(f"Transformation matrix saved to: {output_path}")
    print(f"Matrix shape: {transform.shape}")
    print(f"Matrix:\n{transform}")


def main():
    """Main function to generate coordinate transformation."""
    parser = argparse.ArgumentParser(
        description='Generate coordinate transformation matrix for ArUco tag to robot base frame'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output JSON file path for transformation matrix'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['identity', 'example', 'translation', 'rotation'],
        default='example',
        help='Method to generate transformation matrix (default: example)'
    )
    parser.add_argument(
        '--translation',
        type=float,
        nargs=3,
        metavar=['DX', 'DY', 'DZ'],
        help='Translation values [dx, dy, dz] in meters (for translation method)'
    )
    parser.add_argument(
        '--rotation',
        type=float,
        nargs=3,
        metavar=['RX', 'RY', 'RZ'],
        help='Rotation values [rx, ry, rz] in radians (for rotation method)'
    )

    args = parser.parse_args()

    # Generate transformation matrix based on method
    if args.method == 'identity':
        transform = create_identity_transform()
        print("Using identity transformation (no transformation)")

    elif args.method == 'example':
        transform = create_example_transform()
        print("Using example transformation (placeholder values)")
        print("WARNING: This is just an example! Replace with actual calibration data.")

    elif args.method == 'translation':
        if args.translation is None:
            print("Error: --translation required for translation method")
            return 1
        transform = create_translation_transform(*args.translation)
        print(f"Using translation transformation: {args.translation}")

    elif args.method == 'rotation':
        if args.rotation is None:
            print("Error: --rotation required for rotation method")
            return 1
        dx, dy, dz = args.translation if args.translation else [0, 0, 0]
        transform = create_rotation_transform(*args.rotation, dx, dy, dz)
        print(f"Using rotation+translation transformation: rot={args.rotation}, trans={args.translation}")

    # Save transformation matrix
    output_path = Path(args.output)
    save_transform_matrix(transform, output_path)

    print("\nTo use this transformation with run_dataset_pose_publisher.py:")
    print(f"python run_dataset_pose_publisher.py dataset.zarr.zip --coord_transform {output_path}")

    return 0


if __name__ == '__main__':
    exit(main())
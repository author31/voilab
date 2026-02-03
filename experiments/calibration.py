import numpy as np
import cv2
import json
import glob
import os

# --- Configuration ---
# Path to the folder containing calibration images
# IMAGES_FOLDER = 'charuco_gopro_normal_lens'
VIDEO_PATH = "calibration_video.mp4"
# Output file for calibration data
CALIBRATION_FILE = 'calibration.json'
# Dimensions of the ChAruco board
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 5
IMAGE_EXTS= ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG')
# Display settings for detected boards
DISPLAY_SCALE = 0.5 # Scale images down for display to fit screen

# --- ChAruco Board Setup ---
# Define the ArUco dictionary.
# You can choose different dictionaries (e.g., DICT_4X4_50, DICT_6X6_250)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
ALLOW_EXPORT = False

# Create the ChAruco board object
# squareLength and markerLength should be in the same units (e.g., meters)
CHARUCO_BOARD = cv2.aruco.CharucoBoard(
    size=(CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT),
    squareLength=0.04,
    markerLength=0.02,
    dictionary=ARUCO_DICT)

def main():
    """
    Main function to perform camera calibration from a .mp4 video.
    """
    # --- SETTINGS ---
    VIDEO_PATH = "/home/hcis-s17/author_workdir/voilab/1124_calib_vid/GX010405.MP4"
    FRAME_STEP = 10

    # Arrays to store ChAruco detections
    corners_all = []
    ids_all = []
    image_size = None

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{VIDEO_PATH}'")
        return

    print(f"Processing video: {VIDEO_PATH}")

    frame_index = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        # Frame sampling
        if frame_index % FRAME_STEP != 0:
            frame_index += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)

        if ids is not None and len(ids) > 0:

            img_display = cv2.aruco.drawDetectedMarkers(frame.copy(), corners)

            # Interpolate ChAruco corners
            response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, CHARUCO_BOARD
            )

            if response is not None and response > 20:
                corners_all.append(charuco_corners)
                ids_all.append(charuco_ids)

                img_display = cv2.aruco.drawDetectedCornersCharuco(
                    img_display, charuco_corners, charuco_ids
                )

                if image_size is None:
                    image_size = gray.shape[::-1]

                # Show detection
                h, w = img_display.shape[:2]
                resized = cv2.resize(img_display,
                                     (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
                cv2.imshow("Charuco Board Detection", resized)
                cv2.waitKey(1)

                processed_frames += 1

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    if not image_size:
        print("Calibration failed: No ChAruco boards detected.")
        return

    print(f"\nDetected boards in {processed_frames} frames. Calibrating...")

    # Calibration
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        corners_all, ids_all, CHARUCO_BOARD, image_size, None, None
    )

    # --- Save and print JSON (unchanged from your version) ---
    if ret:
        dist_coeffs_flat = distCoeffs.flatten()
        calibration_data = {
            "final_reproj_error": ret,
            "fps": 0.0,
            "image_height": image_size[1],
            "image_width": image_size[0],
            "intrinsic_type": "FISHEYE",
            "intrinsics": {
                "aspect_ratio": cameraMatrix[0, 0] / cameraMatrix[1, 1],
                "focal_length": cameraMatrix[0, 0],
                "principal_pt_x": cameraMatrix[0, 2],
                "principal_pt_y": cameraMatrix[1, 2],
                "radial_distortion_1": dist_coeffs_flat[0] if len(dist_coeffs_flat) > 0 else 0.0,
                "radial_distortion_2": dist_coeffs_flat[1] if len(dist_coeffs_flat) > 1 else 0.0,
                "radial_distortion_3": dist_coeffs_flat[2] if len(dist_coeffs_flat) > 2 else 0.0,
                "radial_distortion_4": dist_coeffs_flat[3] if len(dist_coeffs_flat) > 3 else 0.0,
                "skew": cameraMatrix[0, 1],
            },
            "nr_calib_images": len(corners_all),
            "stabelized": False,
        }

        print(json.dumps(calibration_data, indent=4))

        if ALLOW_EXPORT:
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(calibration_data, f, indent=4)

        print(f"Calibration data saved to '{CALIBRATION_FILE}'")
    else:
        print("Calibration failed.")

if __name__=="__main__":
  main()

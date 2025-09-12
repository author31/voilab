import numpy as np
import cv2
import json
import glob
import os

# --- Configuration ---
# Path to the folder containing calibration images
IMAGES_FOLDER = 'charuco_gopro_normal_lens'
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
    Main function to perform camera calibration.
    """
    # Arrays to store object points and image points from all images
    corners_all = []  # Corners discovered in all images processed
    ids_all = []      # Aruco ids corresponding to corners discovered
    image_size = None # Determined at runtime from the first valid image

    # Check if the images folder exists
    if not os.path.isdir(IMAGES_FOLDER):
        print(f"Error: The specified folder '{IMAGES_FOLDER}' does not exist.")
        print("Please create it and add your calibration images.")
        return

    # Find all image files in the specified folder
    # Supports multiple common image formats
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(glob.glob(os.path.join(IMAGES_FOLDER, ext)))

    # Exit if no images found
    if not image_paths:
        print(f"Calibration failed: No images found in the '{IMAGES_FOLDER}' folder.")
        return

    print(f"Found {len(image_paths)} images. Starting detection...")

    # Loop through all found images
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find ArUco markers in the image
        corners, ids, _ = cv2.aruco.detectMarkers(image=gray, dictionary=ARUCO_DICT)

        # If markers are found, draw them and interpolate for ChAruco corners
        if ids is not None and len(ids) > 0:
            img_display = cv2.aruco.drawDetectedMarkers(image=img.copy(), corners=corners)

            # Get ChAruco corners
            response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=CHARUCO_BOARD)

            # If enough corners are found, store them and draw the detected board
            # The threshold (e.g., > 20) ensures a reliable detection
            if response > 20:
                corners_all.append(charuco_corners)
                ids_all.append(charuco_ids)

                img_display = cv2.aruco.drawDetectedCornersCharuco(
                    image=img_display,
                    charucoCorners=charuco_corners,
                    charucoIds=charuco_ids)

                # Set image size on the first successful detection
                if image_size is None:
                    image_size = gray.shape[::-1]

                # Display the image with detected markers and corners
                h, w = img_display.shape[:2]
                resized_img = cv2.resize(img_display, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
                cv2.imshow('Charuco Board Detection', resized_img)
                cv2.waitKey(50) # Show image for a short duration (50ms)
            else:
                print(f"Not enough corners found in {image_path}")
        else:
            print(f"No markers detected in {image_path}")

    cv2.destroyAllWindows()

    # --- Perform Calibration ---
    # Ensure we have enough data to proceed
    if not image_size:
        print("\nCalibration failed. No ChAruco boards were detected in any of the images.")
        print("Things to check:")
        print("1. Are the board dimensions (ROWCOUNT, COLCOUNT) correct?")
        print("2. Are the images clear and well-lit?")
        print("3. Is the board fully visible in most images?")
        return

    print("\nBoard detection complete. Calibrating camera...")

    # Calibrate the camera
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)

    # --- Save Results ---
    if ret:
        print("Calibration successful!")
        
        # --- Create JSON Output ---
        # Flatten the distortion coefficients array for easier access
        dist_coeffs_flat = distCoeffs.flatten()

        # Build the dictionary for JSON output
        calibration_data = {
            "final_reproj_error": ret,
            "fps": 0.0,  # Placeholder, not calculated in this script
            "image_height": image_size[1],
            "image_width": image_size[0],
            "intrinsic_type": "FISHEYE", # Set based on requested format
            "intrinsics": {
                "aspect_ratio": cameraMatrix[0, 0] / cameraMatrix[1, 1],
                "focal_length": cameraMatrix[0, 0], # Using fx from camera matrix
                "principal_pt_x": cameraMatrix[0, 2],
                "principal_pt_y": cameraMatrix[1, 2],
                # Note: Mapping the first 4 distortion coeffs from the output vector.
                # Standard model order is (k1, k2, p1, p2, k3, ...).
                "radial_distortion_1": dist_coeffs_flat[0] if len(dist_coeffs_flat) > 0 else 0.0,
                "radial_distortion_2": dist_coeffs_flat[1] if len(dist_coeffs_flat) > 1 else 0.0,
                "radial_distortion_3": dist_coeffs_flat[2] if len(dist_coeffs_flat) > 2 else 0.0,
                "radial_distortion_4": dist_coeffs_flat[3] if len(dist_coeffs_flat) > 3 else 0.0,
                "skew": cameraMatrix[0, 1]
            },
            "nr_calib_images": len(corners_all),
            "stabelized": False
        }

        # Print the formatted JSON to the console
        print("\n--- Calibration Data (JSON) ---")
        print(json.dumps(calibration_data, indent=4))

        if ALLOW_EXPORT:
            # Save calibration data to a JSON file
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump(calibration_data, f, indent=4)
            
        print(f"\nCalibration data saved to '{CALIBRATION_FILE}'")
    else:
        print("\nCalibration failed. Could not compute camera parameters.")

if __name__=="__main__":
  main()
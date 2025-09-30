#!/usr/bin/env python3

"""
Simple script to run ROS2 subscription using ROS2INFRASTRUCTURE.
"""

import time
import sys
import os
import cv2
import numpy as np
from sensor_msgs.msg import Image, JointState

# Add the source path to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages/diffusion_policy/src'))

from diffusion_policy.infrastructure.ros2_infrastructure import ROS2Manager
from std_msgs.msg import Float32, Float64
from cv_bridge import CvBridge


def test_callback(topic, data):
    """Test callback function."""
    print(f"Callback called for topic {topic}")
    print(f"Data type: {type(data)}")
    print(f"Data keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")

    # Convert ROS Image to OpenCV format
    bridge = CvBridge()
    try:
        # Check if data is a dictionary (likely from get_data method)
        if isinstance(data, dict):
            # Extract image data from dictionary
            if 'data' in data:
                # Assuming the dictionary contains raw image data
                img_data = data['data']
                if isinstance(img_data, np.ndarray):
                    cv_image = img_data
                else:
                    # Convert to numpy array if needed
                    cv_image = np.frombuffer(img_data, dtype=np.uint8)
                    cv_image = cv_image.reshape(data.get('height', 480), data.get('width', 640), -1)
            else:
                print("No 'data' key in dictionary")
                return
        else:
            # Data is likely a ROS Image message
            cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

        # Save image to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"received_image_{timestamp}.jpg"
        cv2.imwrite(filename, cv_image)
        print(f"Saved image to {filename}")

    except Exception as e:
        print(f"Error converting/saving image: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Test ROS2 subscription."""
    print("Starting ROS2 subscription test...")

    # Create manager
    manager = ROS2Manager()

    try:
        # Initialize infrastructure
        print("Initializing ROS2 infrastructure...")
        infrastructure = manager.initialize(node_name='test_subscriber')

        # Give some time for initialization
        time.sleep(1)

        # Create subscriber for a test topic
        test_topic = '/rgb'
        print(f"Creating subscriber for topic: {test_topic}")

        subscription_id = infrastructure.create_subscriber(
            topic=test_topic,
            msg_type=Image,
            callback=test_callback
        )

        print(f"Subscriber created with ID: {subscription_id}")

        # Wait for messages
        print("Waiting for messages (press Ctrl+C to stop)...")

        # Wait for some time to see if we get messages
        for i in range(10):
            print(f"Waiting... {i+1}/10")
            time.sleep(1)

            # Check if we have data
            data = infrastructure.get_data(test_topic)
            if data:
                print(f"Received data: {data}")
            else:
                print("No data received yet")

    except KeyboardInterrupt:
        print("Test interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Shutting down...")
        manager.shutdown()
        print("Test completed")


if __name__ == '__main__':
    main()

import omni.graph.core as og
from typing import Dict, Any


class IsaacSimActionGraph:
    """Action Graph for Isaac Sim simulation pipeline with ROS2 integration"""

    def __init__(self, task_name: str, usd_path: str):
        self.task_name = task_name
        self.usd_path = usd_path
        self.graph_handle = None
        self.nodes = {}

    def create_action_graph(self) -> None:
        """Create simulation pipeline action graph with ROS2 integration"""

        # Define node creation configurations
        keys = og.Controller.Keys

        (graph_handle, nodes, _) = og.Controller.edit(
            {
                "graph_path": "/World/ROS_JointStates",
                "evaluator_name": "execution",
            },
            {
                keys.CREATE_NODES: [
                    # Core timing and context nodes
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),

                    # ROS2 joint state nodes
                    ("PublisherJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("SubscriberJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),

                    # Robot control
                    ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),

                    # End-effector pose reading
                    ("end_effector_translate", "omni.graph.nodes.ReadPrimAttribute"),
                    ("end_effector_rotate", "omni.graph.nodes.ReadPrimAttribute"),
                    ("break_3_vector", "omni.graph.nodes.BreakVector3"),
                    ("break_4_vector", "omni.graph.nodes.BreakVector4"),

                    # ROS2 pose publisher
                    ("ros2_publisher", "isaacsim.ros2.bridge.ROS2Publisher"),

                    # Gripper width calculation
                    ("read_prim_attribute", "omni.graph.nodes.ReadPrimAttribute"),
                    ("read_prim_attribute_01", "omni.graph.nodes.ReadPrimAttribute"),
                    ("subtract", "omni.graph.nodes.Subtract"),
                    ("magnitude", "omni.graph.nodes.Magnitude"),
                    ("ros2_publisher_01", "isaacsim.ros2.bridge.ROS2Publisher"),

                    # Camera rendering and publishing
                    ("isaac_run_one_simulation_frame", "isaacsim.core.nodes.OmnIsaacRunOneSimulationFrame"),
                    ("isaac_create_render_product", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("ros2_camera_helper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ],

                keys.SET_VALUES: [
                    # Context configuration
                    ("Context.inputs:domain_id", 0),
                    ("Context.inputs:useDomainIDEnvVar", True),

                    # Joint state publisher
                    ("PublisherJointState.inputs:topicName", "/joint_states"),
                    ("PublisherJointState.inputs:nodeNamespace", ""),
                    ("PublisherJointState.inputs:targetPrim", "/World/panda/root_joint"),

                    # Joint state subscriber
                    ("SubscriberJointState.inputs:topicName", "/joint_command"),
                    ("SubscriberJointState.inputs:nodeNamespace", ""),

                    # Articulation controller target
                    ("ArticulationController.inputs:targetPrim", "/World/panda"),

                    # End-effector translate reader
                    ("end_effector_translate.inputs:name", "xformOp:translate"),
                    ("end_effector_translate.inputs:prim", "/World/panda/panda_link7"),
                    ("end_effector_translate.inputs:usePath", False),

                    # End-effector rotation reader
                    ("end_effector_rotate.inputs:name", "xformOp:orient"),
                    ("end_effector_rotate.inputs:prim", "/World/panda/panda_link7"),
                    ("end_effector_rotate.inputs:usePath", False),

                    # Pose publisher
                    ("ros2_publisher.inputs:topicName", "/eeff_states"),
                    ("ros2_publisher.inputs:messageName", "Pose"),
                    ("ros2_publisher.inputs:messagePackage", "geometry_msgs"),

                    # Gripper width readers
                    ("read_prim_attribute.inputs:name", "xformOp:translate"),
                    ("read_prim_attribute.inputs:prim", "/World/panda/panda_leftfinger"),
                    ("read_prim_attribute.inputs:usePath", False),

                    ("read_prim_attribute_01.inputs:name", "xformOp:translate"),
                    ("read_prim_attribute_01.inputs:prim", "/World/panda/panda_rightfinger"),
                    ("read_prim_attribute_01.inputs:usePath", False),

                    # Gripper width publisher
                    ("ros2_publisher_01.inputs:topicName", "/gripper_width"),
                    ("ros2_publisher_01.inputs:messageName", "Float64"),
                    ("ros2_publisher_01.inputs:messagePackage", "std_msgs"),

                    # Camera render product
                    ("isaac_create_render_product.inputs:cameraPrim", "/World/panda/panda_link7/gopro_link/camera"),
                    ("isaac_create_render_product.inputs:width", 224),
                    ("isaac_create_render_product.inputs:height", 224),

                    # Camera helper
                    ("ros2_camera_helper.inputs:topicName", "/camera/rgb/image_raw"),
                    ("ros2_camera_helper.inputs:type", "rgb"),
                    ("ros2_camera_helper.inputs:frameSkipCount", 0),
                ],

                keys.CONNECT: [
                    # Execution flow
                    ("OnPlaybackTick.outputs:tick", "PublisherJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "SubscriberJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "end_effector_translate.inputs:usdTimecode"),
                    ("OnPlaybackTick.outputs:tick", "end_effector_rotate.inputs:usdTimecode"),
                    ("OnPlaybackTick.outputs:tick", "read_prim_attribute.inputs:usdTimecode"),
                    ("OnPlaybackTick.outputs:tick", "read_prim_attribute_01.inputs:usdTimecode"),
                    ("OnPlaybackTick.outputs:tick", "isaac_run_one_simulation_frame.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ros2_publisher.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ros2_publisher_01.inputs:execIn"),

                    # Context propagation
                    ("Context.outputs:context", "PublisherJointState.inputs:context"),
                    ("Context.outputs:context", "SubscriberJointState.inputs:context"),
                    ("Context.outputs:context", "ros2_publisher.inputs:context"),
                    ("Context.outputs:context", "ros2_publisher_01.inputs:context"),
                    ("Context.outputs:context", "ros2_camera_helper.inputs:context"),

                    # Time data
                    ("ReadSimTime.outputs:simulationTime", "PublisherJointState.inputs:timeStamp"),

                    # Joint control flow
                    ("SubscriberJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                    ("SubscriberJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                    ("SubscriberJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                    ("SubscriberJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),

                    # End-effector pose processing
                    ("end_effector_translate.outputs:value", "break_3_vector.inputs:tuple"),
                    ("end_effector_rotate.outputs:value", "break_4_vector.inputs:tuple"),

                    # Position components
                    ("break_3_vector.outputs:x", "ros2_publisher.inputs:position:x"),
                    ("break_3_vector.outputs:y", "ros2_publisher.inputs:position:y"),
                    ("break_3_vector.outputs:z", "ros2_publisher.inputs:position:z"),

                    # Orientation components
                    ("break_4_vector.outputs:x", "ros2_publisher.inputs:orientation:x"),
                    ("break_4_vector.outputs:y", "ros2_publisher.inputs:orientation:y"),
                    ("break_4_vector.outputs:z", "ros2_publisher.inputs:orientation:z"),
                    ("break_4_vector.outputs:w", "ros2_publisher.inputs:orientation:w"),

                    # Gripper width calculation
                    ("read_prim_attribute.outputs:value", "subtract.inputs:a"),
                    ("read_prim_attribute_01.outputs:value", "subtract.inputs:b"),
                    ("subtract.outputs:difference", "magnitude.inputs:input"),
                    ("magnitude.outputs:magnitude", "ros2_publisher_01.inputs:data"),

                    # Camera pipeline
                    ("isaac_run_one_simulation_frame.outputs:step", "isaac_create_render_product.inputs:execIn"),
                    ("isaac_create_render_product.outputs:renderProductPath", "ros2_camera_helper.inputs:renderProductPath"),
                    ("isaac_create_render_product.outputs:execOut", "ros2_camera_helper.inputs:execIn"),
                ],
            },
        )

        self.graph_handle = graph_handle
        self.nodes = nodes
        print(f"Created action graph for task: {self.task_name}")

    def set_pipeline_stage(self) -> None:
        """Set graph's pipeline stage to simulation"""
        if self.graph_handle:
            self.graph_handle.set_pipeline_stage(og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION)
            print(f"Set pipeline stage to simulation for {self.task_name}")

    def start(self) -> None:
        """Start action graph"""
        if self.graph_handle:
            self.graph_handle.play()
            print(f"Started action graph for {self.task_name}")

    def stop(self) -> None:
        """Stop action graph"""
        if self.graph_handle:
            self.graph_handle.stop()
            print(f"Stopped action graph for {self.task_name}")

    def cleanup(self) -> None:
        """Clean up action graph resources"""
        if self.graph_handle:
            og.Controller.remove(self.graph_handle)
            self.graph_handle = None
            print(f"Cleaned up action graph for {self.task_name}")
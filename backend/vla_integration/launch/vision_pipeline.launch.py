import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Launch file for VLA vision processing components."""

    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='vla',
        description='Namespace for vision processing nodes'
    )

    # Get launch configurations
    namespace = LaunchConfiguration('namespace')

    # Vision processor node
    vision_processor_node = Node(
        package='vla_integration',
        executable='vision_processor',
        name='vision_processor',
        namespace=namespace,
        parameters=[
            os.path.join(
                get_package_share_directory('vla_integration'),
                'config',
                'object_detection_config.yaml'
            )
        ],
        remappings=[
            ('/camera/rgb/image_rect_color', '/camera/rgb/image_rect_color'),
            ('/camera/depth/image_rect_raw', '/camera/depth/image_rect_raw'),
            ('/camera/rgb/camera_info', '/camera/rgb/camera_info'),
            ('/vla/vision_data', 'vla/vision_data'),
        ],
        output='screen'
    )

    # Vision grounding node
    vision_grounding_node = Node(
        package='vla_integration',
        executable='vision_grounding',
        name='vision_grounding',
        namespace=namespace,
        parameters=[
            os.path.join(
                get_package_share_directory('vla_integration'),
                'config',
                'object_detection_config.yaml'
            )
        ],
        remappings=[
            ('/vla/vision_data', 'vla/vision_data'),
            ('/vla/object_detection', 'vla/object_detection'),
        ],
        output='screen'
    )

    # Object localizer node
    object_localizer_node = Node(
        package='vla_integration',
        executable='object_localizer',
        name='object_localizer',
        namespace=namespace,
        output='screen'
    )

    # Vision service node
    vision_service_node = Node(
        package='vla_integration',
        executable='vision_service',
        name='vision_service',
        namespace=namespace,
        parameters=[
            os.path.join(
                get_package_share_directory('vla_integration'),
                'config',
                'object_detection_config.yaml'
            )
        ],
        remappings=[
            ('/vision/localize_object', 'vla/vision/localize_object'),
        ],
        output='screen'
    )

    # Scene understanding node
    scene_understanding_node = Node(
        package='vla_integration',
        executable='scene_understanding',
        name='scene_understanding',
        namespace=namespace,
        output='screen'
    )

    return LaunchDescription([
        namespace_arg,
        vision_processor_node,
        vision_grounding_node,
        object_localizer_node,
        vision_service_node,
        scene_understanding_node,
    ])
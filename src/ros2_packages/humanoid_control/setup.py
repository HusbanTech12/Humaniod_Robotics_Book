from setuptools import setup
from glob import glob
import os

package_name = 'humanoid_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        # Include all URDF files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='husban12',
    maintainer_email='husban12@example.com',
    description='Package for controlling humanoid robot systems',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_command_publisher = humanoid_control.joint_command_publisher:main',
            'sensor_subscriber = humanoid_control.sensor_subscriber:main',
            'config_service = humanoid_control.config_service:main',
            'config_client = humanoid_control.config_client:main',
            'behavior_action_server = humanoid_control.behavior_action_server:main',
            'behavior_action_client = humanoid_control.behavior_action_client:main',
            'sensor_processing_node = humanoid_control.sensor_processing_node:main',
            'state_estimation_node = humanoid_control.state_estimation_node:main',
            'behavior_manager_node = humanoid_control.behavior_manager_node:main',
            'physics_validation_test = humanoid_control.physics_validation_test:main',
            'joint_validation_test = humanoid_control.joint_validation_test:main',
            'sensor_data_publisher = humanoid_control.sensor_data_publisher:main',
            'sensor_verification_test = humanoid_control.sensor_verification_test:main',
            'sensor_quality_test = humanoid_control.sensor_quality_test:main',
        ],
    },
)
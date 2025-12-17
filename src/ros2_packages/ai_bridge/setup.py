from setuptools import setup
import os
from glob import glob

package_name = 'ai_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='husban12',
    maintainer_email='husban12@example.com',
    description='Package for bridging AI agents to ROS 2 controllers',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ai_bridge = ai_bridge.ai_bridge:main',
        ],
    },
)
from setuptools import find_packages, setup

package_name = 'aegis_autonomy'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aegis Flight Systems',
    maintainer_email='engineer@aegis.ai',
    description='Aegis Autonomous Flight Pilot OS - Core ROS 2 Nodes',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = aegis_autonomy.perception_node:main',
            'sensor_fusion_node = aegis_autonomy.sensor_fusion_node:main',
            'cognitive_planner_node = aegis_autonomy.cognitive_planner_node:main',
            'flight_control_node = aegis_autonomy.flight_control_node:main',
            'mission_management_node = aegis_autonomy.mission_management_node:main',
            'hardware_interface_node = aegis_autonomy.hardware_interface_node:main',
            'atc_node = aegis_autonomy.atc_node:main',
            'computer_vision_node = aegis_autonomy.computer_vision_node:main',
            'shadow_mode_node = aegis_autonomy.shadow_mode_node:main',
            'envelope_protection_node = aegis_autonomy.envelope_protection_node:main',
            'visual_navigation_node = aegis_autonomy.visual_navigation_node:main',
            'emergency_ai_node = aegis_autonomy.emergency_ai_node:main',
            'depth_perception_node = aegis_autonomy.depth_perception_node:main',
        ],
    },
)

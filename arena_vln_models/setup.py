from setuptools import setup
from glob import glob


package_name = 'arena_vln_models'


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name, ['README.md', 'arena_vln_models/tracking_config.py']),
    ],
    install_requires=['setuptools', 'PyYAML'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='dev@example.com',
    description='Arena native FollowPath interface and benchmark lifecycle.',
    license='MIT',
    entry_points={'console_scripts': [
        'benchmark = arena_vln_models.runner:main',
        'example_client = arena_vln_models.example_client:main',
        'timing_observer = arena_vln_models.timing:main',
    ]},
)

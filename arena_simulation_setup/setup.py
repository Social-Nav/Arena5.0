import os
from setuptools import setup, find_namespace_packages

package_name = 'arena_simulation_setup'
python_root = 'src'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_namespace_packages(
        where=python_root,
    ),
    package_dir={'': python_root},
    data_files=[
        ('share/' + package_name, ['package.xml']),
        # Will recursively track all .yaml files in the entities/robots
        # directory and its subdirectories.
        *[
            (
                os.path.join('share', package_name, base),
                [os.path.join(base, file)]
            )
            for dir in ['configs', 'launch', 'resource', 'worlds']
            for base, dirs, files in os.walk(dir)
            if '__pycache__' not in base.split(os.sep)
            for file in files
            if not file.endswith(('.pyc', '.pyo'))
        ],
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ],
    install_requires=[
        'setuptools',
        'aiofiles',
        'attrs',
        'cattrs',
        'numpy',
        'Pillow',
        'pydantic',
        'PyYAML',
        'shapely',
    ],
    zip_safe=True,
    maintainer='voshch',
    maintainer_email='dev@voshch.dev',
    description='arena_simulation_setup.',
    license='MIT',
    scripts=[
        'scripts/download_assets',
        'scripts/model_staging',
        'scripts/touch_world',
    ],
    entry_points={
        'console_scripts': [
            f'generate_world = {package_name}.utils.generative.world_generator:main',
            f'world_generator = {package_name}.utils.generative.world_generator_ros:main',
            f'laser_scan_to_cloud = {package_name}.laser_scan_to_cloud:main',
        ],
    },
)

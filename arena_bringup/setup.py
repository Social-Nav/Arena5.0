import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arena_bringup'


def recursive_walk(base_dir, *, destination=None, relative_to=None):
    if destination is None:
        destination = os.path.join('share', package_name)
    if relative_to is None:
        relative_to = ''

    def process(base, files):
        adjusted_base = os.path.relpath(base, relative_to)
        return (
            os.path.normpath(os.path.join(destination, adjusted_base)),
            [
                os.path.join(base, file)
                for file in files
                if not file.endswith(('.pyc', '.pyo'))
            ]
        )

    return [
        process(base, files)
        for base, _, files in os.walk(base_dir)
        if '__pycache__' not in base.split(os.sep)
    ]


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, f'{package_name}.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *recursive_walk('scripts', destination=os.path.join('lib', package_name), relative_to='scripts'),
        *recursive_walk('launch'),
        *recursive_walk('configs'),
    ],
    install_requires=['setuptools', 'PyYAML', 'numpy', 'Pillow', 'imageio'],
    zip_safe=True,
    maintainer='voshch',
    maintainer_email='dev@voshch.dev',
    description='Arena bringup package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'test = arena_bringup.test:main',
            'internnav_eval = arena_bringup.internnav_eval:main',
            'dual_vln_eval = arena_bringup.internnav_eval:main',
            'social_nav_validation = arena_bringup.social_nav_validation:main',
            'social_nav_scenario_validate = arena_bringup.social_nav_scenario:main',
            'social_nav_scenario_eval = arena_bringup.social_nav_scenario:eval_main',
            'social_nav_metrics_aggregate = arena_bringup.social_nav_metrics_aggregate:main',
            'grscenes_episode_to_eval_config = arena_bringup.grscenes_episode_to_eval_config:main',
        ],
        'launch_ros.node_action': [
            'NodeLogLevelExtension = arena_bringup.extensions.NodeLogLevelExtension:NodeLogLevelExtension',
        ],

    },
)

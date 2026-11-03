from glob import glob

from setuptools import find_packages, setup


package_name = 'arena_vln_models'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, package_name + '.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name, ['arena_vln_models/tracking_config.py']),
    ],
    install_requires=['setuptools', 'numpy', 'Pillow', 'requests', 'PyYAML'],
    extras_require={
        # colcon inspects standard testing extras to select its pytest test
        # extension.  Modern setuptools no longer accepts tests_require.
        'test': ['pytest'],
        'internnav': [
            'PyYAML',
            'torch',
            'transformers',
        ],
    },
    zip_safe=True,
    maintainer='dev',
    maintainer_email='dev@example.com',
    description='Arena model adapters plus native Nav2 FollowPath benchmark lifecycle.',
    license='MIT',
    entry_points={'console_scripts': [
        'benchmark = arena_vln_models.runner:main',
        'timing_observer = arena_vln_models.timing:main',
        'model_executor = arena_vln_models.executor:main',
        'cmd_vel_trace = arena_vln_models.cmd_vel_trace:run_node',
    ]},
)

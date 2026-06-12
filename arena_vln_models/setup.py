from setuptools import setup


package_name = 'arena_vln_models'


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'Pillow', 'requests'],
    extras_require={
        'internnav': [
            'PyYAML',
            'torch',
            'transformers',
        ],
    },
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='dev@example.com',
    description='Arena model-sim adapters and InternNav HTTP client helpers.',
    license='MIT',
    entry_points={'console_scripts': []},
)

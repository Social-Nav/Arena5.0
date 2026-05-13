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
    install_requires=['setuptools', 'numpy', 'Pillow'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='dev@example.com',
    description='Consolidated Arena VLN model wrapper and benchmark server package.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'dual_vln_server = arena_vln_models.dual_vln_server:main',
        ],
    },
)

from setuptools import find_packages, setup

package_name = 'smart_palletizer_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Usama Sattar',
    maintainer_email='usamasattar.3347@gmail.com',
    description='Python nodes for smart_palletizer project',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'detection = smart_palletizer_py.detection:main',
            'post_processing = smart_palletizer_py.post_processing:main'
        ],
    },
)

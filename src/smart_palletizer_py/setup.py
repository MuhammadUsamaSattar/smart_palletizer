from setuptools import find_packages, setup
from glob import glob
import os

package_name = "smart_palletizer_py"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.launch.py")),
        ),
        (
            os.path.join("share", package_name, "rviz2"),
            glob(os.path.join("rviz2", "*.rviz")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Usama Sattar",
    maintainer_email="usamasattar.3347@gmail.com",
    description="Python nodes for smart_palletizer project",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "box_detection = smart_palletizer_py.box_detection:main",
            "post_processing = smart_palletizer_py.post_processing:main",
            "pose_detection = smart_palletizer_py.pose_detection:main",
            "box_spawn = smart_palletizer_py.box_spawn:main",
        ],
    },
)

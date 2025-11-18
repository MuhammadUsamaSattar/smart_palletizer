from setuptools import setup, find_packages

# python dependencies listed here will be automatically installed with the package
install_deps = [
    "numpy",
    "open3d>=0.18.0",
    "opencv-python",
    "scipy",
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="smart_palletizer",
    version="0.0.0",
    author="todo",
    author_email="todo@neura-robotics.com",
    maintainer="todo",
    maintainer_email="todo@neura-robotics.com",
    description=(
        "This package does: todo."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_deps,
    python_requires=">=3.6",
)

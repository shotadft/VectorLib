"""パッケージセットアップスクリプト"""
from setuptools import setup, find_packages

exclude_patterns = [
    "tests*",
    "venv*",
    "build*",
    "dist*",
    "*.egg-info*",
    ".vscode*",
    "__pycache__*",
    ".mypy_cache*",
    ".pytest_cache*",
    ".coverage*",
    ".tox*",
    ".nox*",
    "htmlcov*",
    ".hypothesis*",
    ".cache*",
    ".ipynb_checkpoints*",
    ".python-version*",
    ".dmypy.json*",
    "Thumbs.db*",
    "Desktop.ini*",
    "*.env*",
    "*.local*",
]

import os
if os.path.exists("README.md"):
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = ""

setup(
    name="VectorLib",
    version="1.0.2",
    description="This is a library that enables vector calculations in Python.",
    author="Shotadft",
    author_email="98450322+shotadft@users.noreply.github.com",
    url="https://github.com/shotadft/VectorLib",
    packages=find_packages(where="package", exclude=exclude_patterns),
    package_data={
        "package": ["py.typed"],
    },
    include_package_data=True,
    install_requires=[
        "numpy>=1.26.0,<2.3",
        "numba>=0.61.2"
    ],
    extras_require={
        "cupy": ["cupy>=10.0"],
    },
    python_requires=">=3.13",
    classifiers=[
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Typing :: Typed",
        "License :: OSI Approved :: MIT License",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)

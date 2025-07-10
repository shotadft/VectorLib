from setuptools import setup, find_packages

setup(
    name="VectorLib",
    version="1.0.0",
    description="This is a library that enables vector calculations in Python.",
    author="Shotadft",
    author_email="98450322+shotadft@users.noreply.github.com",
    url="https://github.com/shotadft/VectorLib",
    packages=find_packages(),
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
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Typing :: Typed",
        "License :: OSI Approved :: MIT License",
    ],
    long_description=open("README.md", encoding="utf-8").read() if __import__('os').path.exists("README.md") else "",
    long_description_content_type="text/markdown",
) 
import os
from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
        # Filter out comments and empty lines, and strip quotes if present
        return [line.strip().strip("'").strip('"') for line in lines if line.strip() and not line.strip().startswith('#')]

setup(
    name="WeTok",
    version="0.1.0",
    # Include 'src' package so 'src.WeTok' imports work
    packages=find_packages(), 
    # Include generate_wetok.py as a top-level module
    py_modules=["generate_wetok"],
    install_requires=parse_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "image_wetok=generate_wetok:main",
        ],
    },
    author="WeTok Contributors",
    description="WeTok generation and reconstruction tool",
    python_requires=">=3.8",
)

"""Setup script for ASR Data Augmentation Pipeline."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="asr-data-augmentation-pipeline",
    version="1.0.0",
    author="k_nurf",
    description="A configurable pipeline for converting Label Studio ASR exports into Whisper-ready datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BITZ-IT-Consulting-LTD/asr-data-augmentation-pipeline",
    license="GPL-3.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "asr-pipeline=asr_pipeline.pipeline:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    keywords="asr speech-recognition audio-augmentation whisper machine-learning",
)

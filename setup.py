from setuptools import setup, find_packages

setup(
    name="rant_to_rock",
    version="0.1.0",
    description="Transform audio recordings into semantically organized Markdown files for Obsidian",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "librosa>=0.9.0",
        "soundfile>=0.10.3",
        "openai>=0.27.0",
        "torch>=1.9.0",
        "transformers>=4.15.0",
        "sentence-transformers>=2.2.0",
    ],
    tests_require=[
        "pytest>=7.0.0",
        "pytest-mock>=3.10.0",
    ],
    python_requires=">=3.8",
) 
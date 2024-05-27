from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    description = f.read()

setup(
    name='krezi',
    version='1.3',
    python_requires='>= 3.9',
    author="Aadil Zikre",
    author_email="aadilzikre@gmail.com",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.23.0',
        'pandas>=1.4.2',
        'itables>=1.5.2',
        'tqdm'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3'
    ],
    description="Array of Utilities for Lazy but Efficient Programmers",
    long_description=description,
    long_description_content_type='text/markdown'
)
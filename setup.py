from setuptools import setup
import bisemantic

setup(
    name="bisemantic",
    version=bisemantic.__version__,
    packages=["bisemantic"],
    url="",
    license="",
    entry_points={
        "console_scripts": ["bisemantic=bisemantic.main:main"],
    },
    author="W.P. McNeill",
    author_email="billmcn@gmail.com",
    description="Text pair equivalence detector", install_requires=['pandas']
)

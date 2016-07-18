import os
from setuptools import setup

_packages = ["polara",
            "polara/recommender",
            "polara/evaluation",
            "polara/lib",
            "polara/tools",
            "polara/tools/mymedialite"]

_libs = ["numpy", "scipy", "pandas" ,"numba", "requests", "matplotlib", "seaborn"]

opts = dict(name="polara",
            description="Fast and flexible recommender framework",
            keywords = "recommender system",
            version = "0.1",
            license="MIT",
            author="Evgeny Frolov",
            platforms=["any"],
            packages=_packages,
            requires=_libs)


if __name__ == '__main__':
    setup(**opts)

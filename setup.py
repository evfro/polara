import os
from setuptools import setup

_packages = ["polara",
            "polara/recommender",
            "polara/recommender/coldstart",
            "polara/evaluation",
            "polara/datasets",
            "polara/lib",
            "polara/tools",
            "polara/tools/mymedialite",
            "polara/tools/graphlab"]


opts = dict(name="polara",
            description="Fast and flexible recommender framework",
            keywords = "recommender system",
            version = "0.3.5",
            license="MIT",
            author="Evgeny Frolov",
            platforms=["any"],
            packages=_packages)


if __name__ == '__main__':
    setup(**opts)

import os
from setuptools import setup

_packages = ["polara",
            "polara/recommender",
            "polara/evaluation",
            "polara/datasets",
            "polara/lib",
            "polara/tools",
            "polara/recommender/coldstart",
            "polara/recommender/hybrid",
            "polara/recommender/contextual",
            "polara/recommender/external",
            "polara/recommender/external/mymedialite",
            "polara/recommender/external/turi",
            "polara/recommender/external/implicit",
            "polara/recommender/external/lightfm"]


opts = dict(name="polara",
            description="Fast and flexible recommender system framework",
            keywords = "recommender system",
            version = "0.6.4",
            license="MIT",
            author="Evgeny Frolov",
            platforms=["any"],
            packages=_packages)

# opts.update(extras)

if __name__ == '__main__':
    setup(**opts)

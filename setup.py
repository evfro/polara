from setuptools import setup

_packages = ["polara",
             "polara/recommender",
             "polara/recommender/coldstart",
             "polara/evaluation",
             "polara/datasets",
             "polara/lib",
             "polara/tools",
             "polara/recommender/external",
             "polara/recommender/external/mymedialite",
             "polara/recommender/external/graphlab",
             "polara/recommender/external/implicit",
             "polara/recommender/external/lightfm"]


opts = dict(name="polara",
            description="Fast and flexible recommender system framework",
            keywords="recommender system",
            version="0.6.2.dev",
            license="MIT",
            author="Evgeny Frolov",
            platforms=["any"],
            packages=_packages)

extras = dict(install_requires=['futures; python_version=="2.7"'])

opts.update(extras)

if __name__ == '__main__':
    setup(**opts)

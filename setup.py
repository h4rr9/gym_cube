from setuptools import setup

setup(
    name="gymcube",
    version="0.0.1",
    install_requires=["gym>=0.17.3", "numpy>=1.17.3"],
    test_suite="nose2.collector.collector",
)

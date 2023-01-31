from setuptools import setup

setup(
    name="forecasting_model",
    author="Evgeny Grigorenko",
    packages=["src"],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "run_forecasting=src.run_forecasting:main"
        ]
    },
    install_requires=[
        "pandas",
        "numpy",
        "catboost",
        "h5py",
        "loguru",
    ]
)
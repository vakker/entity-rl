# pylint: disable=line-too-long


from setuptools import find_packages, setup

install_requires = [
    "numpy",
    "matplotlib",
    "pandas",
    "scikit-image",
    "tqdm",
    "python-multipart",
    "rich",
    "ray[rllib]",
    "torch",
    "torchvision",
    "transformers",
    "tensorboard",
    "attrdict",
    "yacs",
    "scikit-image",
    "imagecodecs",
    "gymnasium[atari,box2d]",
    "autorom[accept-rom-license]",
    "aim",
    "torch-scatter",
    "torch-sparse",
    "torch-geometric",
    "simple-playgrounds",
    "gputil",
    "mmengine",
]

test_requires = [
    "flake8",
    "pylint",
    "pytest",
    "pytest-cov",
    "pytest-env",
    "pytest-sugar",
]

dev_requires = test_requires + [
    "pre-commit",
    "ipdb",
]


setup(
    name="entity-rl",
    version="0.0.1",
    description="",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    dependency_links=[
        "git+https://github.com/gaorkl/simple-playgrounds.git@legacy-v1#egg=simple-playgrounds",
    ],
    extras_require={
        "test": test_requires,
        "dev": dev_requires,
    },
)

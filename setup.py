from setuptools import setup

setup(
    name="modelscope",
    version="0.1.0",
    description="Yet another PyTorch summary tool.",
    url="https://github.com/trsvchn/modelscope",
    author="Taras Savchyn",
    author_email="trsvchn@gmail.com",
    license="MIT",
    packages=["modelscope"],
    install_requires=["torch"],
)

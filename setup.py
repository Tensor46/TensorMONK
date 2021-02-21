from setuptools import setup, find_packages

# version info
version = "0.4"
package_name = "tensormonk"
print("Building wheel {}-{}".format(package_name, version))

with open("README.md", "r") as txt:
    long_description = txt.read()


# requirements
requirements = [
    "torch",
    "torchvision",
    "visdom",
    "numpy",
    "pandas",
    "seaborn",
    "matplotlib",
    "pillow>=6.2.0",
    "opencv-python",
    "imageio",
    "wget",
    "scikit-image"]


setup(name="tensormonk",
      version=version,
      author="Vikas Gottemukkula",
      author_email="vikasgottemukkula@gmail.com",
      url="https://github.com/Tensor46/TensorMONK",
      description="TensorMONK - A collection of deep learning architectures",
      long_description=long_description,
      keywords=["pytorch", "cnn", "gan", "objectdetection"],
      license="MIT",
      python_requires=">=3.6",
      packages=find_packages(exclude=("test", "unittests")),
      install_requires=requirements,
      zip_safe=True,
      extras_require={},
      classifiers=["Programming Language :: Python :: 3 :: Only"])

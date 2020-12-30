from setuptools import setup, find_packages

setup(name     = "anet",
      version  = "0.1",
      license  = "GPL-3.0",
      keywords = "reinforcement learning, language emergence, multi-agent systems, language grounding, unsupervised semantics, deep learning, transfer learning",
      packages = find_packages(),
      install_requires = [
                          "matplotlib<=3.1.3"
                         ]
     )

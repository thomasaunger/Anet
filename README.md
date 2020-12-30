# Anet
This is a package for the study of [*cultural evolution*](https://en.wikipedia.org/wiki/Cultural_evolution), in particular the emergence of discrete communication. This package generalizes the [BabyAI S/R](https://github.com/thomasaunger/babyai_sr) package to arbitrary tasks.

## Installation

First, install [this BabyAI fork](https://github.com/thomasaunger/babyai) and the corresponding [MiniGrid fork](https://github.com/thomasaunger/gym-minigrid), which ease compatibility. (If you’re feeling lucky, you can instead try installing the [original BabyAI repo](https://github.com/mila-iqia/babyai) (and the corresponding [original MiniGrid repo](https://github.com/maximecb/gym-minigrid)).)

Then, clone this repository and install it with `pip3`:

```
git clone https://github.com/thomasaunger/anet.git
cd anet
pip3 install --editable .
```

In order to use the plotting and visualization scripts as is, you’ll also need to install [LaTeX](https://www.latex-project.org).

## Usage

This package is organized similarly to the original BabyAI repo. Models can be trained and tested using the scripts in the appropriate folder within the `scripts` folder.

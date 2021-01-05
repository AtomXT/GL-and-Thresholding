
# Graphical Model and Thresholding


## Table of Contents

- [Background](#background)
- [Install](#install)
- [Requirements](#requirements)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background

Consider a random verctor $\mathbf{x} = (x_1, x_2, ...,x_d)$ with a multivariate normal distribution. Let $\Sigma_{*} \in \mathbb{S}_{+}^d$ denote the covariance matrix associate with the vector $\mathbf{x}$.  The inverse of the covariance matrix can be used to determine the conditional independence between the random variables.

The sparsity graph of $\Sigma_{*}^{-1}$ represents a graphical model capturing the condittional independence between the elements of $\mathbf{x}$.

Graphical lasso (GL) is one of the most commonly used techniques for estimating the inverse covariance matrix. It is known that GL is computationally expensive for large-scale problems. Therefore, we developed an explicit closed-form solution that can serve either as an approximate solution of the GL or the optimal slution of the GL with a perturbed sample covariance matix. 


## Install

The source code is currently hosted on GitHub at: https://github.com/AtomXT/GL-and-Thresholding.


## Requirements

For MATLAB users, our code is implemented with MATLAB R2020a.

For Python users, our code is implemeted with Python 3.6 and you will need:
 - NumPy & SciPy


## Usage

Let's say $x$ is a $n$ by $m$ sample data matrix. $n$ is the dimension of each sample and $m$ is the number of samples.

$\mathrm{Sigma}$ is the sample covariance matrix of $x$.

$\mathrm{lambda}$ is the regularization parameter for thresholding.

You can use the following code:

- Python

```sh
closed_form(x, lambda)

```

- MATLAB

```sh
Closed_form(x, lambda)
```
If the dimension of sample data is small, this function will only return the closed form solution of GL. Otherwise, it will return three matrices: 

- $\mathrm{A}$ : The closed form soluton.
- $\mathrm{S}$ : The thresholded $\mathrm{Sigma}$ and the diagonals remain unchanged.
- $\mathrm{Sigma\_res}$ : It has the same values as $\mathrm{Sigma}$ at positions picked by thresholding, and the same diagonals as $\mathrm{Sigma}$.

### Example codes and test data

There is a 12000*6000 test data, and corresponding scripts to run the test.

Here is the [dataset](https://drive.google.com/file/d/1AV26CgaNr0z7x-hdgMPCWbdX5VLigHHH/view?usp=sharing).

With the test data, run the following test code:

- [Python](https://github.com/AtomXT/GL-and-Thresholding/tree/main/code/Python)

    run closed_form.py

- [MATLAB](https://github.com/AtomXT/GL-and-Thresholding/tree/main/code/MATLAB)

    run test_closed_form.m


## Maintainers

[@AtomXT](https://github.com/AtomXT).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/AtomXT/GL-and-Thresholding/issues/new) or submit PRs.

### Contributors

[@Tong Xu](https://github.com/AtomXT)    [@Salar Fattahi]()


## License

MIT © Tong Xu

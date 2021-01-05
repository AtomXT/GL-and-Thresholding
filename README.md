
# Graphical Lasso and Thresholding


## Table of Contents

- [Background](#background)
- [Install](#install)
- [Requirements](#requirements)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background

Consider a random verctor x with a multivariate normal distribution. Let Sigma denote the covariance matrix associate with the vector x.  The inverse of the covariance matrix can be used to determine the conditional independence between the random variables.

The sparsity graph of inverse of Sigma represents a graphical model capturing the condittional independence between the elements of x.

Graphical lasso (GL) is one of the most commonly used techniques for estimating the inverse covariance matrix. It is known that GL is computationally expensive for large-scale problems. Therefore, we developed an explicit closed-form solution that can serve either as an approximate solution of the GL or the optimal slution of the GL with a perturbed sample covariance matix. 


## Install

The source code is currently hosted on GitHub at: https://github.com/AtomXT/GL-and-Thresholding.


## Requirements

For MATLAB users, our code is implemented with MATLAB R2020a.

For Python users, our code is implemeted with Python 3.6 and you will need:
 - NumPy & SciPy


## Usage

Let's say x is a n by m sample data matrix. n is the dimension of each sample and m is the number of samples.

Sigma is the sample covariance matrix of x, and lambda is the regularization parameter for thresholding.

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

- A : The closed form soluton.
- S : The thresholded Sigma and the diagonals remain unchanged.
- Sigma_res : It has the same values as Sigma at positions picked by thresholding, and the same diagonals as Sigma.

### Example codes and test data

There is a 12000*6000 test data, and corresponding scripts to run the test.

Here is the [dataset](https://drive.google.com/file/d/1AV26CgaNr0z7x-hdgMPCWbdX5VLigHHH/view?usp=sharing).

With the test data, run the following test code:

- [Python](https://github.com/AtomXT/GL-and-Thresholding/tree/main/code/Python)

    run closed_form.py

- [MATLAB](https://github.com/AtomXT/GL-and-Thresholding/tree/main/code/MATLAB)

    run test_closed_form.m


## Maintainers

[@Tong Xu](https://github.com/AtomXT).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/AtomXT/GL-and-Thresholding/issues/new) or submit PRs.

### Contributors

Salar Fattahi: fattahi@umich.edu

Tong Xu: xutong@umich.edu


## License

MIT © Tong Xu

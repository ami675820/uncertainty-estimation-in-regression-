This repository contains implementations of two probabilistic regression methods applied to the Auto MPG dataset:

Gaussian Mixture Model (GMM)

Mixture Density Network (MDN)

Both methods were developed and analyzed as part of my MS Thesis work:

Amish Anand (2025). Uncertainty Estimation in Regression Using Additive Models and Gaussian Processes. Master’s Thesis, Department of Data Science and Engineering, IISER Bhopal.

📌 Gaussian Mixture Model (GMM) Framework
This approach begins by using a base model (GP-NAM) to make an initial prediction. The difference between the true target value and this initial prediction is calculated as the residual. The core idea is to model the relationship between the input features and these residuals.

A Gaussian Mixture Model is used to represent the joint probability distribution of the inputs and their corresponding residuals. The number of Gaussian components in the mixture is a key hyperparameter, which is selected by finding the model that minimizes the Bayesian Information Criterion (BIC).

Each component in the GMM has its own set of parameters: a mixing weight, a mean vector, and a covariance matrix. For a new, unseen input, the framework uses the rules of Gaussian conditioning to derive a conditional probability distribution for the residual. The final predicted residual is a weighted average based on how much "responsibility" each component takes for the given input. The variance is also calculated to quantify the uncertainty.

The final prediction is then the sum of the original prediction from the base model and the predicted residual from the GMM. The predictive uncertainty is derived from the variance calculated by the model.

📌 Mixture Density Network (MDN) Framework
The Mixture Density Network models the distribution of residuals using a neural network. Similar to the GMM approach, it starts by calculating the residuals from a base prediction model.

The MDN consists of a standard neural network architecture (e.g., with ReLU activations in its hidden layers), but with a specialized output layer. Instead of outputting a single value, the network outputs the parameters for a mixture of Gaussian distributions. For a mixture of m Gaussians, the network will output three sets of parameters for each input:

Means: The center of each Gaussian component.

Standard Deviations: The width or spread of each component (typically output as logarithms for numerical stability).

Mixing Coefficients: The weight for each component, processed through a softmax layer to ensure they all sum to one.

The network is trained to find the optimal parameters of this mixture model that best describe the distribution of the residuals given the input features. The final predicted residual is the expected value (the mean) of the resulting mixture distribution. This predicted residual is then added to the base model's prediction to get the final result.

📖 Citation
If you use this work, please cite the thesis and foundational references:

Thesis

Amish Anand. Uncertainty Estimation in Regression Using Additive Models and Gaussian Processes. MS Thesis, IISER Bhopal, 2025.

Foundational References

Douglas A Reynolds et al. Gaussian mixture models. Encyclopedia of Biometrics, 741(659-663):3, 2009.

Alexander Fabisch. gmr: Gaussian mixture regression. JOSS, 6(62):3054, 2021.

Zoubin Ghahramani and Michael Jordan. Supervised learning from incomplete data via an EM approach. NeurIPS, 1993.

Christopher M Bishop. Mixture density networks. Technical Report, 1994.

Axel Brando. Mixture density networks (MDN) for distribution and uncertainty estimation. Master’s Thesis, 2017.

Axel Brando. Mixture density networks (MDN) for distribution and uncertainty estimation. GitHub repository, 2017.

Wei Zhang, Brian Barr, and John Paisley. Gaussian process neural additive models. arXiv:2402.12518, 2024.

📬 Contact
📧 amish6202@gmail.com

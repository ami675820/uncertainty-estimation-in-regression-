# Probabilistic Regression on Auto MPG

This repository provides implementations of two probabilistic regression methods applied to the **Auto MPG dataset**: a **Gaussian Mixture Model (GMM)** and a **Mixture Density Network (MDN)**. Both methods focus on modeling the error distribution of a base model to provide not only accurate predictions but also robust uncertainty estimates.

This work is part of my Master's thesis:

> **Amish Anand (2025). _Uncertainty Estimation in Regression Using Additive Models and Gaussian Processes._ Master’s Thesis, Department of Data Science and Engineering, IISER Bhopal.**

---

## 📂 Dataset

The project uses the classic **[Auto MPG Dataset](https://archive.ics.uci.edu/ml/datasets/auto+mpg)** from the UCI Machine Learning Repository. 
## 📌 Core Methodologies

Both methods share a common strategy: instead of modeling the target variable directly, they model the **residuals** (errors) of a powerful base model, in this case, a **Gaussian Process Neural Additive Model (GP-NAM)**.

### 1. Gaussian Mixture Model (GMM) Framework

The GMM approach models the joint probability distribution of the input features and their corresponding residuals.

* **Residual Calculation:** First, the difference between the true target and the base model's prediction is calculated. This error term is the residual.
* **Mixture Modeling:** A Gaussian Mixture Model is fitted to the combined data (inputs and residuals). This captures complex, multi-modal relationships that a single error distribution might miss.
* **Component Selection:** The optimal number of Gaussian components for the mixture is determined by minimizing the **Bayesian Information Criterion (BIC)**, preventing overfitting.
* **Conditional Prediction:** For a new data point, the model uses the rules of Gaussian conditioning to predict the conditional distribution of the residual. The final predicted residual is a weighted average based on each component's "responsibility" for the new input.
* **Final Output:** The final prediction is the sum of the base model's output and the predicted residual. The model also provides a robust uncertainty estimate from the residual's variance.

### 2. Mixture Density Network (MDN) Framework

The MDN uses a neural network to learn the parameters of a mixture model, providing a highly flexible and powerful alternative to the GMM.

* **Neural Network Core:** The model is a neural network that takes input features and, instead of outputting a single prediction, outputs the parameters of a Gaussian mixture distribution.
* **Dynamic Parameter Estimation:** For any given input, the MDN outputs:
    * **Means:** The centers of the Gaussian components.
    * **Standard Deviations:** The spread or width of each component.
    * **Mixing Coefficients:** The weight of each component in the final mixture, calculated via a softmax layer.
* **Conditional Distribution:** The network effectively learns to map input features to a specific, customized probability distribution for the residual.
* **Final Output:** The final predicted residual is the mean of this output distribution. This is added back to the base model's prediction to produce the final result, complete with a learned uncertainty profile.

---

## 📖 Citation

If you use this work, please cite the thesis and foundational references:

**Thesis**
- Amish Anand. *Uncertainty Estimation in Regression Using Additive Models and Gaussian Processes.* MS Thesis, IISER Bhopal, 2025.

**Foundational References**
- Douglas A Reynolds et al. Gaussian mixture models. Encyclopedia of Biometrics, 741(659-663):3, 2009.

- Alexander Fabisch. gmr: Gaussian mixture regression. JOSS, 6(62):3054, 2021.

- Zoubin Ghahramani and Michael Jordan. Supervised learning from incomplete data via an EM approach. NeurIPS, 1993.

- Christopher M Bishop. Mixture density networks. Technical Report, 1994.

- Axel Brando. Mixture density networks (MDN) for distribution and uncertainty estimation. Master’s Thesis, 2017.

- Axel Brando. Mixture density networks (MDN) for distribution and uncertainty estimation. GitHub repository, 2017.

- Wei Zhang, Brian Barr, and John Paisley. Gaussian process neural additive models. arXiv:2402.12518, 2024.

## 📬 Contact

Amish Anand - amish6202@gmail.com


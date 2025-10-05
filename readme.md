# Auto MPG – GMM & MDN Approaches  

This repository contains implementations of two probabilistic regression methods applied to the **Auto MPG dataset**:  

1. **Gaussian Mixture Model (GMM)**  
2. **Mixture Density Network (MDN)**  

Both methods were developed and analyzed as part of my **MS Thesis work**:  

> **Amish Anand (2025). _Uncertainty Estimation in Regression Using Additive Models and Gaussian Processes._ Master’s Thesis, Department of Data Science and Engineering, IISER Bhopal.**

---

## 📌 Gaussian Mixture Model (GMM) Framework  



![Gaussian Mixture Model] 

After obtaining the initial prediction from out base model (GP-NAM), the residual is computed as:  

\[
r(x) = y - \hat{y}_{GPNAM}(x).
\]

We model the joint distribution of the inputs and residuals with a Gaussian mixture model:  

\[
p(x, r) = \sum_{k=1}^{K} \pi_k \, \mathcal{N} \left( 
\begin{bmatrix}
x \\
r
\end{bmatrix}
\; \bigg| \; \mu_k, \Sigma_k
\right),
\]

where each component \(k\) has:  

- Mixing weight \(\pi_k\), with \(\sum_{k=1}^K \pi_k = 1\).  
- Mean vector \(\mu_k = \begin{bmatrix} \mu^x_k \\ \mu^r_k \end{bmatrix}\).  
- Covariance matrix partitioned as  

\[
\Sigma_k =
\begin{bmatrix}
\Sigma^{xx}_k & \Sigma^{xr}_k \\
\Sigma^{rx}_k & \Sigma^{rr}_k
\end{bmatrix}.
\]

The number of components \(K\) is selected using the **Bayesian Information Criterion (BIC)**.  

For a new input \(x^*\), the conditional distribution of \(r\) is computed using Gaussian conditioning:  

\[
\mu_{r|x}^k = \mu^r_k + \Sigma^{rx}_k (\Sigma^{xx}_k)^{-1}(x^* - \mu^x_k),
\]

\[
\sigma^2_{r|x}^k = \Sigma^{rr}_k - \Sigma^{rx}_k (\Sigma^{xx}_k)^{-1} \Sigma^{xr}_k.
\]

The responsibility for each component is:  

\[
h_k(x^*) = \frac{\pi_k \, \mathcal{N}(x^* | \mu^x_k, \Sigma^{xx}_k)}{\sum_{j=1}^K \pi_j \, \mathcal{N}(x^* | \mu^x_j, \Sigma^{xx}_j)}.
\]

The overall predicted residual and its variance are:  

\[
\hat{r}(x^*) = \sum_{k=1}^K h_k(x^*) \, \mu_{r|x}^k,
\]

\[
\hat{\sigma}^2(x^*) = \sum_{k=1}^K h_k(x^*) \left[ \sigma^2_{r|x}^k + \left( \mu_{r|x}^k - \hat{r}(x^*) \right)^2 \right].
\]

Thus, the final prediction is:  

\[
\hat{y}(x^*) = \hat{y}_{GPNAM}(x^*) + \hat{r}(x^*),
\]

with predictive uncertainty quantified by:  

\[
\hat{\sigma}(x^*) = \sqrt{\hat{\sigma}^2(x^*)}.
\]

---

## 📌 Mixture Density Network (MDN) Framework  

The MDN models the distribution of residuals using a **neural network-based probabilistic formulation**.  

![Mixture Density Network]
Residuals are defined as:  

\[
r = y - \hat{y}_{base}.
\]

### MDN Structure  

1. **Hidden Layer:** Uses nonlinear activations (ReLU) to capture complex patterns.  
2. **Output Layer:** Outputs \(3m\) values for a mixture of \(m\) Gaussians:  
   - Means: \(\mu_1, \dots, \mu_m\).  
   - Log-standard deviations: \(\tilde{\sigma}_1, \dots, \tilde{\sigma}_m\), with \(\sigma_i = \exp(\tilde{\sigma}_i)\).  
   - Mixing coefficients:  

\[
\alpha_i = \frac{\exp(\alpha_i^*)}{\sum_{j=1}^m \exp(\alpha_j^*)}, \quad \sum_i \alpha_i = 1.
\]

The MDN models the conditional distribution:  

\[
p(r|x) = \sum_{i=1}^m \alpha_i \, \mathcal{N}(r ; \mu_i, \sigma_i^2).
\]

### Expected Value & Variance  

\[
\mathbb{E}[r|x] = \sum_{i=1}^m \alpha_i \mu_i,
\]

\[
\text{Var}(r|x) = \sum_{i=1}^m \alpha_i (\sigma_i^2 + \mu_i^2) - \left( \sum_{i=1}^m \alpha_i \mu_i \right)^2.
\]

### Final Prediction  

\[
\hat{y} = \hat{y}_{base} + \mathbb{E}[r|x].
\]

---

## 📖 Citation  

If you use this work, please cite the thesis and foundational references:  

**Thesis**  
- Amish Anand. *Uncertainty Estimation in Regression Using Additive Models and Gaussian Processes.* MS Thesis, IISER Bhopal, 2025.  

**Foundational References**  

-  Douglas A Reynolds et al. Gaussian mixture models. Encyclopedia of biomet
rics, 741(659-663):3, 2009
- Alexander Fabisch. gmr: Gaussian mixture regression. Journal of Open Source
 Software, 6(62):3054, 2021.
-  Zoubin Ghahramani and Michael Jordan. Supervised learning from incom
plete data via an em approach. Advances in neural information processing
 systems, 6, 1993.
- Christopher M Bishop. Mixture density networks. 1994.
- Axel Brando. Mixture density networks (mdn) for distribution and uncer
tainty estimation, 2017. Report of the Master’s Thesis: Mixture Density
 Networks for distribution and uncertainty estimation.
- Axel Brando. Mixture density networks (mdn) for distribution and uncer
tainty estimation, 2017. GitHub repository with a collection of Jupyter note
books intended to solve a lot of problems related to MDN.
- Wei Zhang, Brian Barr, and John Paisley. Gaussian process neural additive
 models. In arXiv preprint arXiv:2402.12518, 2024.



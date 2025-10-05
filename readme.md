# Auto MPG – GMM & MDN Approaches  

This repository contains implementations of two probabilistic regression methods applied to the **Auto MPG dataset**:  

1. **Gaussian Mixture Model (GMM)**  
2. **Mixture Density Network (MDN)**  

Both methods were developed and analyzed as part of my **MS Thesis work**:  

> **Amish Anand (2025). _Uncertainty Estimation in Regression Using Additive Models and Gaussian Processes._ Master’s Thesis, Department of Data Science and Engineering, IISER Bhopal.**

---

## 📌 Gaussian Mixture Model (GMM) Framework  

![Gaussian Mixture Model]
After obtaining the initial prediction from our base model (GP-NAM), the residual is computed as:  

![equation](https://latex.codecogs.com/svg.latex?r(x)%20=%20y%20-%20\hat{y}_{GPNAM}(x))  

We model the joint distribution of the inputs and residuals with a Gaussian mixture model:  

![equation](https://latex.codecogs.com/svg.latex?p(x,r)%20=%20\sum_{k=1}^{K}%20\pi_k%20\mathcal{N}\left(\begin{bmatrix}x%20\\%20r\end{bmatrix}%20\Bigg|%20\mu_k,%20\Sigma_k\right))  

where each component *k* has:  

- Mixing weight ![equation](https://latex.codecogs.com/svg.latex?\pi_k), with ![equation](https://latex.codecogs.com/svg.latex?\sum_{k=1}^K\pi_k=1).  
- Mean vector ![equation](https://latex.codecogs.com/svg.latex?\mu_k=\begin{bmatrix}\mu^x_k\\\mu^r_k\end{bmatrix}).  
- Covariance matrix partitioned as:  

![equation](https://latex.codecogs.com/svg.latex?\Sigma_k=\begin{bmatrix}\Sigma^{xx}_k&\Sigma^{xr}_k\\\Sigma^{rx}_k&\Sigma^{rr}_k\end{bmatrix})  

The number of components *K* is selected using the **Bayesian Information Criterion (BIC)**.  

For a new input ![equation](https://latex.codecogs.com/svg.latex?x^*), the conditional distribution of residual *r* is computed using Gaussian conditioning:  

![equation](https://latex.codecogs.com/svg.latex?\mu_{r|x}^k=\mu^r_k+\Sigma^{rx}_k(\Sigma^{xx}_k)^{-1}(x^*-\mu^x_k))  

![equation](https://latex.codecogs.com/svg.latex?\sigma^2_{r|x}^k=\Sigma^{rr}_k-\Sigma^{rx}_k(\Sigma^{xx}_k)^{-1}\Sigma^{xr}_k)  

The responsibility for each component is:  

![equation](https://latex.codecogs.com/svg.latex?h_k(x^*)=\frac{\pi_k\mathcal{N}(x^*|\mu^x_k,\Sigma^{xx}_k)}{\sum_{j=1}^K\pi_j\mathcal{N}(x^*|\mu^x_j,\Sigma^{xx}_j)})  

The overall predicted residual and its variance are:  

![equation](https://latex.codecogs.com/svg.latex?\hat{r}(x^*)=\sum_{k=1}^Kh_k(x^*)\mu_{r|x}^k)  

![equation](https://latex.codecogs.com/svg.latex?\hat{\sigma}^2(x^*)=\sum_{k=1}^Kh_k(x^*)\Big[\sigma^2_{r|x}^k+(\mu_{r|x}^k-\hat{r}(x^*))^2\Big])  

Thus, the final prediction is:  

![equation](https://latex.codecogs.com/svg.latex?\hat{y}(x^*)=\hat{y}_{GPNAM}(x^*)+\hat{r}(x^*))  

with predictive uncertainty quantified by:  

![equation](https://latex.codecogs.com/svg.latex?\hat{\sigma}(x^*)=\sqrt{\hat{\sigma}^2(x^*)})  

---

## 📌 Mixture Density Network (MDN) Framework  

The MDN models the distribution of residuals using a **neural network-based probabilistic formulation**.  

![Mixture Density Network]

Residuals are defined as:  

![equation](https://latex.codecogs.com/svg.latex?r=y-\hat{y}_{base})  

### MDN Structure  

1. **Hidden Layer:** Uses nonlinear activations (ReLU) to capture complex patterns.  
2. **Output Layer:** Outputs *3m* values for a mixture of *m* Gaussians:  
   - Means: ![equation](https://latex.codecogs.com/svg.latex?\mu_1,...,\mu_m).  
   - Log-standard deviations: ![equation](https://latex.codecogs.com/svg.latex?\tilde{\sigma}_1,...,\tilde{\sigma}_m), with ![equation](https://latex.codecogs.com/svg.latex?\sigma_i=\exp(\tilde{\sigma}_i)).  
   - Mixing coefficients:  

![equation](https://latex.codecogs.com/svg.latex?\alpha_i=\frac{\exp(\alpha_i^*)}{\sum_{j=1}^m\exp(\alpha_j^*)},\quad\sum_i\alpha_i=1)  

The MDN models the conditional distribution:  

![equation](https://latex.codecogs.com/svg.latex?p(r|x)=\sum_{i=1}^m\alpha_i\mathcal{N}(r;\mu_i,\sigma_i^2))  

### Expected Value & Variance  

![equation](https://latex.codecogs.com/svg.latex?\mathbb{E}[r|x]=\sum_{i=1}^m\alpha_i\mu_i)  

![equation](https://latex.codecogs.com/svg.latex?\text{Var}(r|x)=\sum_{i=1}^m\alpha_i(\sigma_i^2+\mu_i^2)-\Big(\sum_{i=1}^m\alpha_i\mu_i\Big)^2)  

### Final Prediction  

![equation](https://latex.codecogs.com/svg.latex?\hat{y}=\hat{y}_{base}+\mathbb{E}[r|x])  

---

## 📖 Citation  

If you use this work, please cite the thesis and foundational references:  

**Thesis**  
- Amish Anand. *Uncertainty Estimation in Regression Using Additive Models and Gaussian Processes.* MS Thesis, IISER Bhopal, 2025.  

**Foundational References**  
- Douglas A Reynolds et al. *Gaussian mixture models.* Encyclopedia of Biometrics, 741(659-663):3, 2009.  
- Alexander Fabisch. *gmr: Gaussian mixture regression.* JOSS, 6(62):3054, 2021.  
- Zoubin Ghahramani and Michael Jordan. *Supervised learning from incomplete data via an EM approach.* NeurIPS, 1993.  
- Christopher M Bishop. *Mixture density networks.* Technical Report, 1994.  
- Axel Brando. *Mixture density networks (MDN) for distribution and uncertainty estimation.* Master’s Thesis, 2017.  
- Axel Brando. *Mixture density networks (MDN) for distribution and uncertainty estimation.* GitHub repository, 2017.  
- Wei Zhang, Brian Barr, and John Paisley. *Gaussian process neural additive models.* arXiv:2402.12518, 2024.  

---

## 📬 Contact  

📧 amish6202@gmail.com

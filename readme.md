This repository contains implementations of two probabilistic regression methods applied to the Auto MPG dataset:

Gaussian Mixture Model (GMM)

Mixture Density Network (MDN)

Both methods were developed and analyzed as part of my MS Thesis work:

Amish Anand (2025). Uncertainty Estimation in Regression Using Additive Models and Gaussian Processes. Master’s Thesis, Department of Data Science and Engineering, IISER Bhopal.

📌 Gaussian Mixture Model (GMM) Framework
After obtaining the initial prediction from our base model (GP-NAM), the residual is computed as:

<p align="center">
<img src="https://math.vercel.app?from=r(x)%20=%20y%20-%20\hat{y}_{GPNAM}(x)">
</p>

We model the joint distribution of the inputs and residuals with a Gaussian mixture model:

<p align="center">
<img src="https://math.vercel.app?from=p(x,r)%20=%20\sum_{k=1}^{K}%20\pi_k%20\mathcal{N}\left(\begin{bmatrix}x%20\%20r\end{bmatrix}%20\Bigg|%20\mu_k,%20\Sigma_k\right)">
</p>

where each component k has:

Mixing weight <img src="https://math.vercel.app?from=\pi_k">, with <img src="https://math.vercel.app?from=\sum_{k=1}^K\pi_k=1">.

Mean vector <img src="https://math.vercel.app?from=\mu_k=\begin{bmatrix}\mu^x_k\\mu^r_k\end{bmatrix}">.

Covariance matrix partitioned as:

<p align="center">
<img src="https://math.vercel.app?from=\Sigma_k=\begin{bmatrix}\Sigma^{xx}_k%26\Sigma^{xr}_k\\Sigma^{rx}_k%26\Sigma^{rr}_k\end{bmatrix}">
</p>

The number of components K is selected using the Bayesian Information Criterion (BIC).

For a new input <img src="https://math.vercel.app?from=x^*">, the conditional distribution of residual r is computed using Gaussian conditioning:

<p align="center">
<img src="https://math.vercel.app?from=\mu_{r|x}^k=\mu^r_k%2B\Sigma^{rx}_k(\Sigma^{xx}k)^{-1}(x^*-\mu^x_k)">
</p>
<p align="center">
<img src="https://math.vercel.app?from=\sigma^2{r|x}^k=\Sigma^{rr}_k-\Sigma^{rx}_k(\Sigma^{xx}_k)^{-1}\Sigma^{xr}_k">
</p>

The responsibility for each component is:

<p align="center">
<img src="https://math.vercel.app?from=h_k(x%3D\frac{\pi_k\mathcal{N}(x^|\mu^x_k,\Sigma^{xx}k)}{\sum{j=1}^K\pi_j\mathcal{N}(x^|\mu^x_j,\Sigma^{xx}_j)}">
</p>

The overall predicted residual and its variance are:

<p align="center">
<img src="https://math.vercel.app?from=\hat{r}(x^)%3D\sum_{k=1}^Kh_k(x^)\mu_{r|x}^k">
</p>
<p align="center">
<img src="https://math.vercel.app?from=\hat{\sigma}^2(x^)%3D\sum_{k=1}^Kh_k(x^)\Big[\sigma^2_{r|x}^k%2B(\mu_{r|x}^k-\hat{r}(x^*))^2\Big]">
</p>

Thus, the final prediction is:

<p align="center">
<img src="https://math.vercel.app?from=\hat{y}(x^)%3D\hat{y}_{GPNAM}(x^)%2B\hat{r}(x^*)">
</p>

with predictive uncertainty quantified by:

<p align="center">
<img src="https://math.vercel.app?from=\hat{\sigma}(x^)%3D\sqrt{\hat{\sigma}^2(x^)}">
</p>

📌 Mixture Density Network (MDN) Framework
The MDN models the distribution of residuals using a neural network-based probabilistic formulation.

Residuals are defined as:

<p align="center">
<img src="https://math.vercel.app?from=r=y-\hat{y}_{base}">
</p>

MDN Structure
Hidden Layer: Uses nonlinear activations (ReLU) to capture complex patterns.

Output Layer: Outputs 3m values for a mixture of m Gaussians:

Means: <img src="https://math.vercel.app?from=\mu_1,...,\mu_m">.

Log-standard deviations: <img src="https://math.vercel.app?from=\tilde{\sigma}_1,...,\tilde{\sigma}_m">, with <img src="https://math.vercel.app?from=\sigma_i=\exp(\tilde{\sigma}_i)">.

Mixing coefficients:

<p align="center">
<img src="https://math.vercel.app?from=\alpha_i%3D\frac{\exp(\alpha_i^)}{\sum_{j=1}^m\exp(\alpha_j^)},\quad\sum_i\alpha_i=1">
</p>

The MDN models the conditional distribution:

<p align="center">
<img src="https://math.vercel.app?from=p(r%3D\sum_{i=1}^m\alpha_i\mathcal{N}(r;\mu_i,\sigma_i^2)">
</p>

Expected Value & Variance
<p align="center">
<img src="https://math.vercel.app?from=\mathbb{E}[r|x]%3D\sum_{i=1}^m\alpha_i\mu_i">
</p>
<p align="center">
<img src="https://math.vercel.app?from=\text{Var}(r|x)%3D\sum_{i=1}^m\alpha_i(\sigma_i^2%2B\mu_i^2)-\Big(\sum_{i=1}^m\alpha_i\mu_i\Big)^2">
</p>

Final Prediction
<p align="center">
<img src="https://math.vercel.app?from=\hat{y}%3D\hat{y}_{base}%2B\mathbb{E}[r|x]">
</p>

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

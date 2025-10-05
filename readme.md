This repository contains implementations of two probabilistic regression methods applied to the Auto MPG dataset:

Gaussian Mixture Model (GMM)

Mixture Density Network (MDN)

Both methods were developed and analyzed as part of my MS Thesis work:

Amish Anand (2025). Uncertainty Estimation in Regression Using Additive Models and Gaussian Processes. Master’s Thesis, Department of Data Science and Engineering, IISER Bhopal.

📌 Gaussian Mixture Model (GMM) Framework
After obtaining the initial prediction from our base model (GP-NAM), the residual is computed as:

r(x)=y− 
y
^
​
  
GPNAM
​
 (x)

We model the joint distribution of the inputs and residuals with a Gaussian mixture model:

p(x,r)= 
k=1
∑
K
​
 π 
k
​
 N([ 
x
r
​
 ] 

​
 μ 
k
​
 ,Σ 
k
​
 )
where each component k has:

Mixing weight π 
k
​
 , with ∑ 
k=1
K
​
 π 
k
​
 =1.

Mean vector μ 
k
​
 =[ 
μ 
k
x
​
 
μ 
k
r
​
 
​
 ].

Covariance matrix partitioned as:

Σ 
k
​
 =[ 
Σ 
k
xx
​
 
Σ 
k
rx
​
 
​
  
Σ 
k
xr
​
 
Σ 
k
rr
​
 
​
 ]
The number of components K is selected using the Bayesian Information Criterion (BIC).

For a new input x 
∗
 , the conditional distribution of residual r is computed using Gaussian conditioning:

μ 
r∣x
k
​
 =μ 
k
r
​
 +Σ 
k
rx
​
 (Σ 
k
xx
​
 ) 
−1
 (x 
∗
 −μ 
k
x
​
 )
$$\sigma^2_{r|x}^k=\Sigma^{rr}_k-\Sigma^{rx}_k(\Sigma^{xx}_k)^{-1}\Sigma^{xr}_k$$
The responsibility for each component is:

h 
k
​
 (x 
∗
 )= 
∑ 
j=1
K
​
 π 
j
​
 N(x 
∗
 ∣μ 
j
x
​
 ,Σ 
j
xx
​
 )
π 
k
​
 N(x 
∗
 ∣μ 
k
x
​
 ,Σ 
k
xx
​
 )
​
 
The overall predicted residual and its variance are:

r
^
 (x 
∗
 )= 
k=1
∑
K
​
 h 
k
​
 (x 
∗
 )μ 
r∣x
k
​
 
$$\hat{\sigma}^2(x^*)=\sum_{k=1}^Kh_k(x^*)\Big[\sigma^2_{r|x}^k+(\mu_{r|x}^k-\hat{r}(x^*))^2\Big]$$
Thus, the final prediction is:

y
^
​
 (x 
∗
 )= 
y
^
​
  
GPNAM
​
 (x 
∗
 )+ 
r
^
 (x 
∗
 )
with predictive uncertainty quantified by:

σ
^
 (x 
∗
 )= 
σ
^
  
2
 (x 
∗
 )

​
 
📌 Mixture Density Network (MDN) Framework
The MDN models the distribution of residuals using a neural network-based probabilistic formulation.

Residuals are defined as:

r=y− 
y
^
​
  
base
​
 

MDN Structure
Hidden Layer: Uses nonlinear activations (ReLU) to capture complex patterns.

Output Layer: Outputs 3m values for a mixture of m Gaussians:

Means: μ 
1
​
 ,...,μ 
m
​
 .

Log-standard deviations:  
σ
~
  
1
​
 ,..., 
σ
~
  
m
​
 , with σ 
i
​
 =exp( 
σ
~
  
i
​
 ).

Mixing coefficients:

α 
i
​
 = 
∑ 
j=1
m
​
 exp(α 
j
∗
​
 )
exp(α 
i
∗
​
 )
​
 , 
i
∑
​
 α 
i
​
 =1
The MDN models the conditional distribution:

p(r∣x)= 
i=1
∑
m
​
 α 
i
​
 N(r;μ 
i
​
 ,σ 
i
2
​
 )
Expected Value & Variance
E[r∣x]= 
i=1
∑
m
​
 α 
i
​
 μ 
i
​
 
Var(r∣x)= 
i=1
∑
m
​
 α 
i
​
 (σ 
i
2
​
 +μ 
i
2
​
 )−( 
i=1
∑
m
​
 α 
i
​
 μ 
i
​
 ) 
2
 
Final Prediction
y
^
​
 = 
y
^
​
  
base
​
 +E[r∣x]
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

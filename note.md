# _belief rule-based system_

rule-based mode: $R=<U,A,D,F>$

antecedent attributes: $U=\{U_i,i=1,\cdots,T\}$  

referential values: $A=\{A_1,A_2,\cdots,A_T\},A_i=\{A_{ij},j=1,\cdots,J_i=|A_i|\}$

consequents: $D=\{D_n,n=1,\cdots,N\}$

logic function: $F$

the k-th rule in BRB system can be written as:

$R_k:if\{(A_1^k,\alpha_1^k) \wedge (A_2^k,\alpha_2^k) \wedge \cdots \wedge (A_{T_k}^k,\alpha_{T_k}^k)\}then\{(D_1,\overline{\beta}_{1k}),(D_2,\overline{\beta}_{2k}),\cdots,(D_N,\overline{\beta}_{Nk})\}$

with a rule weight $\theta_k$
and attribute weights $\delta_{k1},\delta_{k2},\cdots,\delta_{kT_k}$

$\alpha_i^k$ and $\overline{\beta}_{ik}$ are belief degree

aggregation function to calculate the toltal rule belief degree:

$\alpha_k=\varphi((\delta_{k1},\alpha_1^k),(\delta_{k2},\alpha_2^k),\cdots,(\delta_{kT_k},\alpha_{T_k}^k))$
$\{ex:\alpha_k=\prod_{i=1}^{T_k}(\alpha_i^k)^{\overline{\delta}_{ki}},\overline{\delta}_{ki}=\frac{\delta_{ki}}{\max_{j=1,\cdots,T_k}\delta_{kj}}\}$

generating and normalizing the activation weight of $R_k$:

$w_k=\frac{\theta_k\alpha_k}{\sum_{i=1}^L\theta_i\alpha_i}$

distribution on the referential values are $S(A_i^*,\varepsilon_i)=\{(A_{ij},\alpha_{ij});j=1,\cdots,J_i\}$,the input $A_i^*$
i is assessed to the referential value
$A_{ij}$ with the degree of belief of $\alpha_{ij}$

updating $\overline{\beta}_{ik}$ in consquent $D_i$:

$\beta_{ik}=\overline{\beta}_{ik}\frac{\sum_{t=1}^{T_k}(\tau(t,k))\sum_{j=1}^{J_i}\alpha_{tj}}{\sum_{t=1}^{T_k}\tau(t,k)},\tau(t,k)=\begin{cases}1,U_t\in R_k\\0,otherwise\end{cases}$

belief rule expression matrinx for a rule base

$
\begin{array}{c|}
O|I & A^1(w_1) & \cdots & A^L(w_L) \\
\hline
D_1 & \beta_{11} & \cdots & \beta_{1L} \\
\vdots & \vdots & \ddots & \vdots \\
D_N & \beta_{N1} & \cdots & \beta_{NL}
\end{array}
$

evidential reasoning approach

transform belief degrees into probability masses:

$m_{j,k}=w_k\beta_{j,k},j=1,\cdots,N$

$m_{D,k}=1-\sum_{j=1}^Nm_{j,k}=1-w_k\sum_{j=1}^{N}\beta_{j,k}$

$\overline{m}_{D,k}=1-w_k$

$\widetilde{m}_{D,k}=w_k(1-\sum_{j=1}^N\beta_{j,k})$

$m_{D,k}=\overline{m}_{D,k}+\widetilde{m}_{D,k}$

the remaining degree of belief of unassigned to any consequent $m_{D,k}$ is split into two parts:$\overline{m}_{D,k}$ caused by the relative importance $w_k$ and $\widetilde{m}_{D,k}$ caused by the incompleteness

suppose $m_{j,I(k)}$ is the combined degree of belief in $D_j$ by aggregating first $k$ rules,$m_{D,I(k)}$ is the remaining degress, let $m_{j,I(1)}=m_{j,1}$ and $m_{D,I(1)}=m_{D,1}$

the combined degree of first $k+1$ rules can be generated as:

$m_{j,I(k+1)}=K_{I(k+1)}[m_{j,I(k)}m_{j,k+1}+m_{j,I(k)}m_{D,k+1}+m_{D,I(k)}m_{j,k+1}]$

$\overline{m}_{D,I(k+1)}=K_{I(k+1)}[\overline{m}_{D,I(k)}\overline{m}_{D,k+1}]$

$\widetilde{m}_{D,I(k+1)}=K_{I(k+1)}[\widetilde{m}_{D,I(k)}\widetilde{m}_{D,k+1}+\widetilde{m}_{D,I(k)}\overline{m}_{D,k+1}+\overline{m}_{D,I(k)}\widetilde{m}_{D,k+1}]$

$K_{I(k+1)}=[1-\sum_{j=1}^N\sum_{t=1,t \neq j}^Nm_{j,I(k)}m_{t,k+1}]^{-1}$

$\beta_j=\frac{m_{j,I(L)}}{1-\overline{m}_{D,I(L)}},\beta_D=\frac{\widetilde{m}{D,I(L)}}{1-\overline{m}_{D,I(L)}},\sum_{j=1}^N\beta_j+\beta_D=1$

an analytical ER algorithm is developed:

$m_j=k[\prod_{i=1}^L(m_{j,i}+m_{D,i})-\prod_{i=1}^Lm_{D,i}]$

$\overline{m}_D=k[\prod_{i=1}^L\overline{m}_{D,i}]$

$\widetilde{m}_D=k[\prod_{i=1}^Lm_{D,i}-\prod_{i=1}^L\overline{m}_{H,i}]$

$k=[\sum_{j=1}^N\prod_{i=1}^L(m_{j,i}+m_{D,i})-(N-1)\prod_{i=1}^Lm_{D,i}]^{-1}$

$\beta_j=\frac{m_j}{1-\overline{m}_D}=\frac{\prod_{i=1}^L(m_{j,i}+m_{D,i})-\prod_{i=1}^Lm_{D,i}}{\sum_{j=1}^N\prod_{i=1}^L(m_{j,i}+m_{D,i})-(N-1)\prod_{i=1}^Lm_{D,i}-\prod_{i=1}^L\overline{m}_{D,i}}$

$\beta_D=\frac{\widetilde{m}_D}{1-\overline{m}_D}=\frac{\prod_{i=1}^Lm_{D,i}-\prod_{i=1}^L\overline{m}_{D,i}}{\sum_{j=1}^N\prod_{i=1}^L(m_{j,i}+m_{D,i})-(N-1)\prod_{i=1}^Lm_{D,i}-\prod_{i=1}^L\overline{m}_{D,i}}$

$\beta_j=\frac{\prod_{i=1}^{L}(\frac{m_{j,i}}{m_{D,i}}+1)-1}{\sum_{j=1}^{N}\prod_{i=1}^{L}(\frac{m_{j,i}}{m_{D,i}}+1)-(N-1)-\prod_{i=1}^{L}\frac{\overline{m}_{D,i}}{m_{D,i}}}$

$\beta_j=\frac{\prod_{i=1}^{L}(\frac{w_i\beta_{j,i}}{1-w_i}+1)-1}{\sum_{j=1}^{N}\prod_{i=1}^{L}(\frac{w_i\beta_{j,i}}{1-w_i}+1)-N}=\frac{\prod_{i=1}^{L}(\frac{\beta_{j,i}}{\frac{1}{w_i}-1}+1)-1}{\sum_{j=1}^{N}\prod_{i=1}^{L}(\frac{\beta_{j,i}}{\frac{1}{w_i}-1}+1)-N}=\frac{\prod_{i=1}^{L}(\frac{\theta_i\alpha_i\beta_{j,i}}{\sum_{k\neq i}{\theta_k\alpha_k}}+1)-1}{\sum_{j=1}^{N}\prod_{i=1}^{L}(\frac{\theta_i\alpha_i\beta_{j,i}}{\sum_{k\neq i}{\theta_k\alpha_k}}+1)-N}$

$\beta_j=\frac{\overline{\beta}_j}{\sum_{i=1}^N\overline{\beta}_i},\overline{\beta}_j=\prod_{i=1}^L(\frac{\theta_i\alpha_i\beta_{j,i}}{\sum_{k\neq i}{\theta_k\alpha_k}}+1)-1,\frac{d\beta_j}{d\overline{\beta}_j}=\frac{\sum_{i=1,i\neq j}^N\overline{\beta}_i}{(\sum_{i=1}^N\overline{\beta}_i)^2}$

&nbsp;

$minimize[F=\frac{1}{2\times batchsize}\sum_{i=1}^{batchsize}\sum_{j=1}^N(\beta_{j,i}^{predict}-\beta_{j,i})^2]$
___

$\frac{d\overline{\beta}_j}{d\alpha_t}=\frac{\theta_t\beta_{j,t}}{\sum_{k\neq t}\theta_k\alpha_k}\prod_{i=1,i\neq t}^L(\frac{\theta_i\alpha_i\beta_{j,i}}{\sum_{k\neq i}{\theta_k\alpha_k}}+1)+\sum_{l=1,l\neq t}^L(-\frac{\theta_l\alpha_l\beta_{j,l}\theta_t}{(\sum_{k\neq l}\theta_k\alpha_k)^2}\prod_{i=1.i\neq l}^L(\frac{\theta_i\alpha_i\beta_{j,i}}{\sum_{k\neq i}{\theta_k\alpha_k}}+1))$

$\alpha_t=\prod_{i=1}^Te^{-\frac{(x_i-x_i^t)^2}{2\delta_i^2}}=e^{-\sum_{i=1}^T\frac{(x_i-x_i^t)^2}{2\delta_i^2}},\frac{d\alpha_t}{dx_j^t}=\frac{x_j-x_j^t}{\delta_j^2}e^{-\sum_{i=1}^T\frac{(x_i-x_i^t)^2}{2\delta_i^2}}$

$\frac{d\beta_j}{dx_j}=\frac{d\beta_j}{d\overline{\beta}_j}\frac{d\overline{\beta}_j}{d\alpha_t}\frac{d\alpha_t}{dx_j^t}$

$\nabla_x F=\sum_{i=1}^{batchsize}\sum_{j=1}^N[(\beta_{j,i}^{predict}-\beta_{j,i})\frac{d\beta_{j,i}^{predict}}{dx}]/batchsize,x_{t+1}=x_t-\mu\nabla_{x_t} F$

___

$\frac{d\overline{\beta}_j}{d\beta_{j,t}}=\frac{\theta_t\alpha_t}{\sum_{k\neq j}\theta_k\alpha_k}\prod_{i=1,i\neq t}^L(\frac{\theta_i\alpha_i\beta_{j,i}}{\sum_{k\neq j}\theta_k\alpha_k}+1),\frac{d\overline{\beta}_{k\neq j}}{d\beta_{j,t}}=\frac{d\overline{\beta}_j}{d\beta_{k\neq j,t}}=0$

$\beta_{j,t}=\frac{e^{y_{j,t}}}{\sum_{k=1}^Ne^{y_{k,t}}},\frac{d\beta_{j,t}}{dy_{j,t}}=\frac{e^{y_{j,t}}\sum_{k=1,k\neq j}^Ne^{y_{k,t}}}{(\sum_{k=1}^Ne^{y_{k,t}})^2}$

$\frac{d\beta_j}{dy_{j,t}}=\frac{d\beta_j}{d\overline{\beta}_j}\frac{d\overline{\beta}_j}{d\beta_{j,t}}\frac{d\beta_{j,t}}{dy_{j,t}}$

$\nabla_yF=\sum_{i=1}^{batchsize}\sum_{j=1}^N[(\beta_{j,i}^{predict}-\beta_{j,i})\frac{d\beta_{j,i}^{predict}}{dy_t}]/batchsize,y_{t+1}=y_t-\mu\nabla_{y_t}F$

___


$\frac{d\overline{\beta_j}}{d\theta_t}=\frac{\alpha_t\beta_{j,t}}{\sum_{k\neq t}\theta_k\alpha_k}\prod_{i=1,i\neq t}^L(\frac{\theta_i\alpha_i\beta_{j,i}}{\sum_{k\neq i}{\theta_k\alpha_k}}+1)+\sum_{l=1,l\neq t}^L(-\frac{\theta_l\alpha_l\beta_{j,l}\alpha_t}{(\sum_{k\neq l}\theta_k\alpha_k)^2}\prod_{i=1.i\neq l}^L(\frac{\theta_i\alpha_i\beta_{j,i}}{\sum_{k\neq i}{\theta_k\alpha_k}}+1))$

$\theta_t=\frac{1}{1+e^{-z_t}},\frac{d\theta_t}{dz_t}=\frac{e^{-z_t}}{(1+e^{-z_t})^2}=\theta_t\times(1-\theta_t)$

$\frac{d\beta_j}{dz_t}=\frac{d\beta_j}{d\overline{\beta}_j}\frac{d\overline{\beta}_j}{d\theta_t}\frac{d\theta_t}{dz_t}$

$\nabla_zF=\sum_{i=1}^{batchsize}\sum_{j=1}^N[(\beta_{j,i}^{predict}-\beta_{j,i})\frac{d\beta_{j,i}^{predict}}{dz_t}]/batchsize,z_{t+1}=z_t-\mu\nabla_{z_t}F$
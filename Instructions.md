# Hybrid EKF-DL Estimator Instructions

## Overview
We will implement the proposed **Hybrid EKF-DL estimator** in Python, applying the AI rules and using all the `<tags>`. The implementation will use multiple classes and functions to build and evaluate an example on a **nonlinear state-space model**, specifically the **Lorenz attractor**. The instructions guide through the steps to simulate the system, integrate a deep learning model, and evaluate performance thoroughly.

## Metadata

- **Type**: Implementation of Hybrid Estimation Architecture
- **Purpose**: Demonstrate Enhanced Nonlinear State Estimation under Uncertainty
- **Paradigm**: Deep Learning Embedded EKF
- **Constraints**: Model Mismatch, Unknown Noise Distributions
- **Objective**: Provide Python Code with Explanations

---

## Core System Description

We will simulate a nonlinear dynamic system—the **Lorenz attractor**—and implement the hybrid **Extended Kalman Filter** with embedded **Deep Learning (EKF-DL)** to address **model mismatch** and **unknown noise distributions**.

The system is defined as:

\[
\begin{aligned}
x_{k+1} &= f(x_k, u_k) + w_k \\
y_k &= h(x_k) + v_k
\end{aligned}
\]

Where:
- \( x_k \in \mathbb{R}^n \): state vector,
- \( u_k \in \mathbb{R}^m \): control input,
- \( y_k \in \mathbb{R}^p \): measurement,
- \( f, h \): nonlinear functions,
- \( w_k, v_k \): process and measurement noise with unknown distributions (epistemic and aleatoric uncertainties).

**Goal**: Estimate \( x_k \) despite model mismatch and unknown noise characteristics.

---

## Think: Problem Analysis and Proposed Solution

### 1. **Disadvantages of EKF**:
- Linearization errors in highly nonlinear systems.
- Sensitivity to model mismatch.
- Assumes Gaussian noise with known covariance, which is invalid for unknown noise distributions.

### 2. **Proposed Solution**:
- Integrate a **Deep Learning (DL)** model to learn residual errors due to model mismatch and unknown noise.
- Embed the **DL model** within the EKF update equations.
- Utilize the DL model to estimate the **Jacobians** and **noise covariances** adaptively.

---

## Expand: Mathematical Details

### 1. **Incorporate DL for Model Mismatch Compensation**
Let the true system dynamics be:
\[
x_{k+1} = f(x_k, u_k) + \Delta f_k + w_k
\]
Where \( \Delta f_k \) represents the **model mismatch**.

#### **Lemma 1**:
If \( \Delta f_k \) can be approximated by a function \( \phi(x_k, u_k; \theta) \) learned by a deep neural network with parameters \( \theta \), then the state prediction can be improved by incorporating \( \phi \).

#### **Proof**:
By approximating \( \Delta f_k \approx \phi(x_k, u_k; \theta) \), we adjust the prediction:
\[
\tilde{x}_{k+1|k} = f(x_k, u_k) + \phi(x_k, u_k; \theta)
\]
This reduces the prediction error due to model mismatch.

---

### 2. **Adaptive Jacobian Estimation**
EKF requires computation of Jacobians:
\[
F_k = \frac{\partial f}{\partial x} \bigg|_{x=\hat{x}_{k|k}, u=u_k}
\]
With model mismatch, \( F_k \) may be inaccurate.

#### **Lemma 2**:
The Jacobian can be estimated using **automatic differentiation** of the augmented function \( f + \phi \).

#### **Proof**:
Since \( \phi \) is differentiable (deep neural network with differentiable activation functions), the Jacobian becomes:
\[
\tilde{F}_k = \frac{\partial [f(x, u) + \phi(x, u; \theta)]}{\partial x} \bigg|_{x=\hat{x}_{k|k}, u=u_k}
\]
This provides a more accurate linearization.

---

### 3. **Estimation of Unknown Noise Covariances**
Let \( Q_k \) and \( R_k \) be the process and measurement noise covariances, which are unknown.

#### **Lemma 3**:
A deep learning model can estimate \( Q_k \) and \( R_k \) **online** using measurement residuals.

#### **Proof**:
Define residuals:
\[
r_k = y_k - h(\hat{x}_{k|k-1})
\]
Train a DL model to map \( r_k \) and other available data to estimates of \( Q_k \) and \( R_k \):
\[
(Q_k, R_k) = \psi(r_k; \theta')
\]
Where \( \psi \) is a neural network with parameters \( \theta' \).

---

## Hybrid EKF-DL Algorithm

### Prediction Step:
\[
\tilde{x}_{k+1|k} = f(\hat{x}_{k|k}, u_k) + \phi(\hat{x}_{k|k}, u_k; \theta)
\]
\[
\tilde{P}_{k+1|k} = \tilde{F}_k P_{k|k} \tilde{F}_k^\top + Q_k
\]

### Update Step:
\[
K_{k+1} = \tilde{P}_{k+1|k} H_{k+1}^\top \left( H_{k+1} \tilde{P}_{k+1|k} H_{k+1}^\top + R_{k+1} \right)^{-1}
\]
\[
\hat{x}_{k+1|k+1} = \tilde{x}_{k+1|k} + K_{k+1} \left[ y_{k+1} - h(\tilde{x}_{k+1|k}) \right]
\]
\[
P_{k+1|k+1} = (I - K_{k+1} H_{k+1}) \tilde{P}_{k+1|k}
\]

### Deep Learning Updates:
Update \( \theta \) and \( \theta' \) using new data to refine \( \phi \) and \( \psi \).

---

## Loop: Algorithm Steps

1. **Initialize** \( \hat{x}_{0|0} \), \( P_{0|0} \), DL parameters \( \theta \), \( \theta' \).
2. **For each time step \( k \)**:
   - **Predict state** using the augmented function:
     \[
     \tilde{x}_{k+1|k} = f(\hat{x}_{k|k}, u_k) + \phi(\hat{x}_{k|k}, u_k; \theta)
     \]
   - **Compute augmented Jacobian** \( \tilde{F}_k \).
   - **Estimate noise covariances** \( Q_k, R_k \) using the DL model \( \psi \).
   - **Predict error covariance** \( \tilde{P}_{k+1|k} \).
   - **Compute Kalman gain** \( K_{k+1} \).
   - **Update state estimate** \( \hat{x}_{k+1|k+1} \) and covariance \( P_{k+1|k+1} \).
   - **Update DL models** \( \phi \) and \( \psi \) with new data.

---

## Verify: Theoretical Effectiveness

**Theorem**: Under mild conditions, the proposed hybrid EKF-DL estimator converges to the true state in the presence of model mismatch and unknown noise distributions.

**Proof Sketch**:
- By incorporating learned corrections \( \phi \) and adaptive covariance estimates \( Q_k, R_k \), the estimator adjusts for discrepancies between the model and reality.
- Under assumptions of observability, controllability, and sufficient excitation, the estimator's error dynamics can be shown to be stable, leading to convergence.

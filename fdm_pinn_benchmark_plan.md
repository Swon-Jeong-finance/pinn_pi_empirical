# FDM-PINN Benchmark 설계 및 코드 수정 계획

이 문서는 현재 `dynalloc_v2` 코드에 **FDM 기반 benchmark backend**를 추가하기 위한 정리 문서다. 목적은 `PINN`이 푸는 **unconstrained HJB PDE**를 neural network 대신 finite difference method, 즉 FDM으로 최대한 정확하게 풀고, 최종 policy는 기존 `pinn`과 동일하게 **unconstrained FOC policy + long-only / risky-cap clipping**으로 평가하는 것이다.

중요한 점은 이 문서의 FDM benchmark가 **constrained HJB solver**가 아니라는 것이다. 지금 목표는 `pipinn`처럼 constrained QP를 HJB 안에 직접 넣는 것이 아니라, `pinn`과 같은 문제를 grid solver로 푼 뒤 동일한 clipping policy를 적용하는 것이다. 따라서 이 benchmark는 다음과 같이 정의하는 것이 가장 정확하다.

\[
\boxed{
\text{FDM-PINN benchmark}
=
\text{unconstrained HJB solved by FDM}
+
\text{FOC policy}
+
\text{ex-post clipping}
}
\]

현재 업로드된 코드 기준으로는 `dynalloc_v2`가 `pipinn`과 `pinn` backend를 포함하고 있고, `dynalloc_v1`에는 이 backend 구조가 직접 들어 있지 않다. 따라서 수정 범위는 `dynalloc_v2` 중심으로 잡는 것이 맞다.

---

## 1. FDM에서 state 차원은 어떻게 봐야 하는가?

### 1.1 원래 HJB의 변수

원래 dynamic portfolio choice 문제의 value function은 다음과 같이 쓸 수 있다.

\[
V(t,x,z),
\]

여기서

- \(t\): calendar time,
- \(x\): wealth,
- \(z\): predictive state vector,
- \(z\in\mathbb R^L\).

현재 코드와 PDF에서 사용하는 문제는 CRRA utility를 갖는다.

\[
U(x)=\frac{x^{1-\gamma}}{1-\gamma}.
\]

CRRA utility와 portfolio weight control \(\pi\) 구조 때문에 value function은 wealth에 대해 homogeneous하다. 따라서

\[
V(t,x,z)
=
\frac{x^{1-\gamma}}{1-\gamma}g(t,z)
\]

로 분리할 수 있고, numerical implementation에서는 보통

\[
u(t,z)=\log g(t,z)
\]

또는 잔여 horizon \(\tau=T-t\) 기준으로

\[
u(\tau,z)=\log g(T-\tau,z)
\]

를 푼다.

따라서 FDM에서 wealth \(x\) 차원은 풀 필요가 없다. 즉 원래 \(V(t,x,z)\)의 3개 종류 변수 중에서 wealth는 제거되고, 실제 grid는

\[
u(\tau,z)
\]

에 대해서만 잡는다.

---

### 1.2 PLS state가 2차원인 경우

현재 사용 구조에서는 PLS를 통해 나오는 state가 보통 2차원이다.

\[
z=(z_1,z_2).
\]

그러면 reduced log-HJB의 unknown은

\[
u(\tau,z_1,z_2)
\]

이다.

따라서 구현상 FDM solution array는 세 개의 축을 갖는다.

\[
(N_\tau+1)\times (N_1+1)\times (N_2+1).
\]

즉 저장되는 numerical solution은

\[
u^n_{i,j}\approx u(\tau_n,z_{1,i},z_{2,j})
\]

형태가 된다.

이런 의미에서는 사용자가 말한 것처럼 **time axis까지 포함하면 3축 grid**가 맞다.

다만 수치 PDE 용어로는 이것을 “3차원 state FDM”이라고 부르기보다는,

\[
\boxed{
\text{2-dimensional state parabolic PDE를 time 방향으로 marching한다}
}
\]

라고 표현하는 것이 더 정확하다.

왜냐하면 \(\tau\)는 state가 아니라 time-marching direction이기 때문이다. 각 time step마다 풀어야 하는 sparse linear system의 공간 크기는

\[
(N_1+1)(N_2+1)
\]

이다. 반면 진짜 state가 3차원인 경우, 예를 들어 \(z=(z_1,z_2,z_3)\)라면 매 time step의 공간 grid 크기가

\[
(N_1+1)(N_2+1)(N_3+1)
\]

이 된다. 이 차이가 매우 크다.

따라서 현재 PLS rank가 2인 경우의 FDM은 다음과 같이 표현하는 것이 가장 정확하다.

> PLS state가 2차원이므로 FDM benchmark는 \((\tau,z_1,z_2)\) grid 위에서 reduced log-HJB를 푼다. 여기서 \(\tau\)는 parabolic PDE의 time-marching 방향이고, \((z_1,z_2)\)가 실제 state-space 차원이다. 따라서 numerical solution은 세 개의 축을 가진 grid에 저장되지만, 각 time step에서 푸는 공간 문제는 2차원 state grid 문제이다.

---

### 1.3 PINN과 FDM의 차원 표현 차이

PINN에서는 neural network 입력이

\[
(\tau,z_1,z_2)
\]

이므로 input dimension이 3이다.

즉 PINN은

\[
u_\theta(\tau,z_1,z_2):\mathbb R^3\to\mathbb R
\]

를 학습한다.

FDM도 저장 grid는 \((\tau,z_1,z_2)\) 세 축을 갖지만, solver는 다음과 같이 한 time layer씩 전진한다.

\[
u^0(z_1,z_2)=0
\]

에서 시작해

\[
u^1(z_1,z_2),\quad
u^2(z_1,z_2),\quad
\ldots,\quad
u^{N_\tau}(z_1,z_2)
\]

를 순서대로 계산한다.

따라서 PINN과 FDM의 대응은 다음과 같다.

| 구분 | PINN | FDM |
|---|---|---|
| unknown | \(u_\theta(\tau,z_1,z_2)\) | \(u^n_{i,j}\) |
| time 처리 | network input의 한 좌표 | marching direction |
| state 처리 | network input의 state 좌표 | spatial grid |
| derivative | automatic differentiation | finite difference |
| policy gradient | \(\nabla_z u_\theta\) | grid derivative + interpolation |

---

### 1.4 policy에 필요한 gradient

policy extraction에는 time derivative \(u_\tau\)가 직접 들어가지 않는다.

`pinn`과 FDM-PINN benchmark에서 policy는 다음 형태로 구한다.

\[
\pi_{\text{unc}}(\tau,z)
=
\frac{1}{\gamma}\Sigma^{-1}
\left[
\mu(z)+C\nabla_z u(\tau,z)
\right].
\]

따라서 필요한 gradient는

\[
\nabla_z u(\tau,z)
=
\begin{pmatrix}
u_{z_1}(\tau,z_1,z_2)\\
u_{z_2}(\tau,z_1,z_2)
\end{pmatrix}
\]

이다.

FDM에서는 grid 위에서 \(u_{z_1}\), \(u_{z_2}\)를 finite difference로 계산하고, 실제 evaluation state \((\tau_t,z_{1,t},z_{2,t})\)에서 interpolation해서 policy를 만든다.

---

## 2. 현재 `pinn`이 푸는 PDE

현재 `dynalloc_v2/pipinn_backend.py` 안에서는 `pipinn`과 `pinn`이 같은 trainer 함수 `train_pipinn_policy(...)`를 공유하고, `training_mode`에 따라 branch가 갈라진다.

중요한 함수는 다음이다.

- `_pinn_unconstrained_hamiltonian(...)`
- `_policy_evaluation(..., training_mode='pinn')`
- `TrainedPIPINN.policy_weights_with_debug(...)`
- `_solve_unconstrained_foc_np(...)`
- `_clip_foc_policy_np(...)`

현재 코드에서 `pinn`의 Hamiltonian은 다음이다.

\[
H_{\text{unc}}(z,\nabla u)
=
\sup_{\pi\in\mathbb R^N}
\left\{
\pi^\top a
-
\frac{\gamma}{2}\pi^\top\Sigma\pi
\right\}
\]

where

\[
a(z,\nabla u)
=
\mu(z)+C\nabla_z u.
\]

unconstrained FOC는

\[
\gamma\Sigma\pi=a
\]

이므로

\[
\pi_{\text{unc}}
=
\frac{1}{\gamma}\Sigma^{-1}a.
\]

Hamiltonian value는

\[
H_{\text{unc}}(z,\nabla u)
=
\frac{1}{2\gamma}
a^\top\Sigma^{-1}a.
\]

따라서 잔여 horizon \(\tau=T-t\) 기준의 log-HJB는 다음처럼 쓸 수 있다.

\[
\boxed{
 u_\tau
 =
 b(z)^\top\nabla_z u
 +
 \frac12\operatorname{tr}\left(QD^2_{zz}u\right)
 +
 \frac12\nabla_z u^\top Q\nabla_z u
 +
 (1-\gamma)
 \left[
 r+\frac{1}{2\gamma}a^\top\Sigma^{-1}a
 \right]
}
\]

with

\[
a=\mu(z)+C\nabla_z u.
\]

초기 조건은 terminal condition을 \(\tau\) 기준으로 바꾼 것이므로

\[
\boxed{u(0,z)=0.}
\]

현재 FDM benchmark는 바로 이 PDE를 neural network가 아니라 finite difference method로 푸는 것이다.

---

## 3. FDM-PINN benchmark의 정확한 목표

추가하려는 backend는 다음 문제를 푼다.

### 3.1 Training / value solve

FDM으로 다음 unconstrained HJB를 푼다.

\[
 u_\tau
 =
 b(z)^\top\nabla_z u
 +
 \frac12\operatorname{tr}\left(QD^2_{zz}u\right)
 +
 \frac12\nabla_z u^\top Q\nabla_z u
 +
 (1-\gamma)
 \left[
 r+
 \frac{1}{2\gamma}
 \left(\mu(z)+C\nabla_z u\right)^\top
 \Sigma^{-1}
 \left(\mu(z)+C\nabla_z u\right)
 \right].
\]

여기서

- \(\mu(z)=a+Bz\): mean model,
- \(b(z)=c+(A-I)z\): discrete transition을 continuous-style drift로 바꾼 것,
- \(\Sigma\): training covariance,
- \(Q\): state innovation covariance,
- \(C\): return-state covariance,
- \(r\): risk-free 또는 policy config의 `risk_premium_r` / `risk_free_rate`,
- \(\gamma\): CRRA risk aversion.

### 3.2 Evaluation / policy extraction

평가 시점에는 기존 `pinn`과 동일하게

\[
g_t=\nabla_z u(\tau_t,z_t)
\]

를 가져오고,

\[
a_t=\mu(z_t)+C_{\text{eval}}g_t
\]

를 만든 뒤,

\[
\pi_{\text{unc},t}
=
\frac1\gamma
\Sigma_{\text{eval}}^{-1}a_t
\]

를 계산한다.

그 다음 `pinn`과 같은 clipping을 적용한다.

\[
\pi_i^+=\max(\pi_{\text{unc},i},0)
\]

and

\[
\pi_t=
\begin{cases}
\pi^+, & \mathbf 1^\top\pi^+\le \ell,\\[4pt]
\ell\dfrac{\pi^+}{\mathbf 1^\top\pi^+}, & \mathbf 1^\top\pi^+>\ell.
\end{cases}
\]

여기서

\[
\ell=\min(\texttt{policy.risky\_cap},1-\texttt{policy.cash\_floor}).
\]

즉 FDM-PINN benchmark의 최종 weight는 기존 `pinn`처럼 feasible하다.

\[
\pi_i\ge0,
\qquad
\mathbf 1^\top\pi\le\ell.
\]

하지만 이것은 `pipinn`처럼 constrained QP를 푼 policy가 아니다. 이 점은 결과 해석에서 분명히 적어야 한다.

---

## 4. 왜 이 FDM benchmark가 필요한가?

현재 `pinn`은 neural network가 PDE residual을 줄이는 방식으로 unconstrained HJB를 학습한다. 그런데 `pinn` 결과가 `pipinn`이나 다른 방법과 크게 다를 때, 원인을 구분해야 한다.

가능한 원인은 크게 두 가지다.

1. `pinn`이 unconstrained HJB PDE 자체를 잘 못 맞추고 있다.
2. PDE를 잘 맞춘다고 해도, unconstrained value + ex-post clipping이라는 구조 자체가 constrained policy problem과 다르다.

FDM-PINN benchmark는 첫 번째 문제를 확인하는 데 유용하다.

즉 neural PINN 대신 FDM으로 같은 unconstrained HJB를 훨씬 deterministic하게 풀어보고, 그 결과에서 나온 FOC-clipping policy가 어떤 성과를 내는지 확인한다.

만약 FDM-PINN과 neural `pinn`이 비슷하다면, neural training 자체보다는 “unconstrained HJB + clipping” 구조가 문제일 가능성이 커진다.

반대로 FDM-PINN과 neural `pinn`이 크게 다르다면, 현재 neural `pinn`의 PDE fitting, gradient quality, spectral bias, collocation sampling, boundary condition 등이 문제일 수 있다.

---

## 5. FDM solver 설계

### 5.1 기본 grid

PLS state가 2차원일 때 grid는 다음과 같다.

\[
\tau_n=n\Delta\tau,
\qquad
n=0,\ldots,N_\tau.
\]

\[
z_{1,i}=z_{1,\min}+i\Delta z_1,
\qquad
i=0,\ldots,N_1.
\]

\[
z_{2,j}=z_{2,\min}+j\Delta z_2,
\qquad
j=0,\ldots,N_2.
\]

solution은

\[
u^n_{i,j}\approx u(\tau_n,z_{1,i},z_{2,j}).
\]

실제 배열 shape은

```text
(n_tau + 1, n_z1, n_z2)
```

또는 index convention에 따라

```text
(n_tau + 1, n_z1 + 1, n_z2 + 1)
```

가 된다.

초기 구현에서는 코드 config에 `n_z1`, `n_z2`, `n_tau`를 명시하는 것이 좋다.

예:

```yaml
fdm:
  n_z1: 81
  n_z2: 81
  n_tau: 240
  scheme: semi_implicit
  boundary: neumann
  max_state_dim: 2
```

state domain은 PINN과 비교 공정성을 위해 현재 `PIPINNEnvFromPPGDPO`가 쓰는 domain과 맞추는 것이 좋다.

현재 env는 `pipinn.x_domain_quantile_low`, `pipinn.x_domain_quantile_high`, `pipinn.x_domain_buffer`를 이용해 state box를 만든다. FDM도 같은 domain을 쓰면 된다.

---

### 5.2 PDE decomposition

PDE를 다음처럼 나눈다.

\[
u_\tau=L u+N(u).
\]

여기서 linear operator는

\[
L u
=
b(z)^\top\nabla_z u
+
\frac12\operatorname{tr}(QD^2_{zz}u).
\]

nonlinear part는

\[
N(u)
=
\frac12\nabla_z u^\top Q\nabla_z u
+
(1-\gamma)
\left[
r+
\frac{1}{2\gamma}
a^\top\Sigma^{-1}a
\right],
\]

where

\[
a=\mu(z)+C\nabla_z u.
\]

---

### 5.3 추천 time stepping: semi-implicit scheme

처음 구현에서는 fully nonlinear implicit Newton solver보다 semi-implicit scheme이 적절하다.

\[
(I-\Delta\tau L)u^{n+1}
=
u^n+
\Delta\tau N(u^n).
\]

즉

- drift/diffusion linear operator \(L\)은 implicit하게 처리,
- nonlinear gradient-square term과 unconstrained Hamiltonian은 previous step \(u^n\)에서 explicit하게 평가.

이 방식은 explicit scheme보다 안정적이고, fully implicit nonlinear solver보다 구현 부담이 훨씬 작다.

구체적인 반복은 다음과 같다.

1. \(u^0=0\)으로 초기화한다.
2. 현재 \(u^n\)에서 \(\nabla_z u^n\)를 finite difference로 계산한다.
3. \(N(u^n)\)를 계산한다.
4. sparse linear system

\[
(I-\Delta\tau L)u^{n+1}=u^n+\Delta\tau N(u^n)
\]

을 푼다.
5. \(n=0,\ldots,N_\tau-1\) 반복한다.

---

### 5.4 finite difference derivative

state가 2차원이고

\[
Q=
\begin{pmatrix}
Q_{11} & Q_{12}\\
Q_{12} & Q_{22}
\end{pmatrix}
\]

이면 diffusion term은

\[
\frac12\operatorname{tr}(QD^2u)
=
\frac12Q_{11}u_{z_1z_1}
+
Q_{12}u_{z_1z_2}
+
\frac12Q_{22}u_{z_2z_2}.
\]

따라서 다음 derivative가 필요하다.

\[
u_{z_1},\quad
u_{z_2},\quad
u_{z_1z_1},\quad
u_{z_1z_2},\quad
u_{z_2z_2}.
\]

기본 finite difference는 다음과 같다.

#### First derivatives, central difference

\[
u_{z_1}(i,j)
\approx
\frac{u_{i+1,j}-u_{i-1,j}}{2\Delta z_1}.
\]

\[
u_{z_2}(i,j)
\approx
\frac{u_{i,j+1}-u_{i,j-1}}{2\Delta z_2}.
\]

#### Second derivatives

\[
u_{z_1z_1}(i,j)
\approx
\frac{u_{i+1,j}-2u_{i,j}+u_{i-1,j}}{\Delta z_1^2}.
\]

\[
u_{z_2z_2}(i,j)
\approx
\frac{u_{i,j+1}-2u_{i,j}+u_{i,j-1}}{\Delta z_2^2}.
\]

#### Cross derivative

\[
u_{z_1z_2}(i,j)
\approx
\frac{
 u_{i+1,j+1}
 -u_{i+1,j-1}
 -u_{i-1,j+1}
 +u_{i-1,j-1}
}{4\Delta z_1\Delta z_2}.
\]

Drift term \(b(z)^\top\nabla u\)는 advection 성격이 있으므로 central difference만 쓰면 불안정할 수 있다. 따라서 안정성을 위해 upwind difference를 고려해야 한다.

예를 들어 \(b_1(z)>0\)이면

\[
u_{z_1}\approx\frac{u_{i,j}-u_{i-1,j}}{\Delta z_1},
\]

\(b_1(z)<0\)이면

\[
u_{z_1}\approx\frac{u_{i+1,j}-u_{i,j}}{\Delta z_1}.
\]

초기 구현에서는 다음 둘 중 하나를 선택할 수 있다.

1. central drift + grid convergence test,
2. upwind drift + central diffusion.

benchmark의 신뢰성을 생각하면 2번, 즉 upwind drift가 더 낫다.

---

### 5.5 boundary condition

초기 구현에서는 zero-Neumann boundary가 가장 무난하다.

\[
\frac{\partial u}{\partial n}=0.
\]

즉 grid boundary에서 외부 방향 gradient를 0으로 두는 방식이다.

구현 방식은 다음 중 하나다.

1. ghost point를 사용해 boundary derivative를 0으로 반영,
2. boundary에서 one-sided derivative 사용,
3. boundary 값을 인접 interior 값과 같게 두는 reflective rule 사용.

처음 benchmark에서는 reflective zero-Neumann이 단순하다.

예:

\[
u_{-1,j}=u_{1,j}
\]

또는 gradient 계산에서

\[
u_{z_1}(0,j)=0
\]

으로 처리할 수 있다.

boundary choice는 결과에 영향을 줄 수 있으므로 반드시 metadata에 기록해야 한다.

---

### 5.6 sparse linear system

semi-implicit scheme에서는 매 time step마다 다음 sparse linear system을 푼다.

\[
A_{\text{fdm}}u^{n+1}=rhs^n,
\]

where

\[
A_{\text{fdm}}=I-\Delta\tau L.
\]

2D grid를 flatten해서

\[
k=iN_2+j
\]

로 index를 만들면, \(u^n\)은 length \(N_1N_2\) vector가 된다.

`L`은 sparse matrix로 구성한다.

- \(u_{z_1z_1}\): \(z_1\) direction second derivative sparse operator,
- \(u_{z_2z_2}\): \(z_2\) direction second derivative sparse operator,
- \(u_{z_1z_2}\): mixed derivative sparse operator,
- drift term: variable coefficient first derivative operator.

구현은 `scipy.sparse.diags`, `scipy.sparse.kron`, `scipy.sparse.eye`, `scipy.sparse.linalg.spsolve`를 쓰는 방식이 가장 직접적이다.

---

### 5.7 gradient interpolation

FDM은 neural network처럼 임의의 \((\tau,z)\)에서 바로 autograd를 할 수 없다. 따라서 evaluation state가 grid point에 정확히 놓이지 않을 때 interpolation이 필요하다.

추천 방식은 다음이다.

1. solve가 끝난 뒤 각 time slice에서 grid gradient를 계산한다.
2. \(u_{z_1}\), \(u_{z_2}\) arrays를 저장한다.
3. `scipy.interpolate.RegularGridInterpolator`로 gradient 자체를 interpolation한다.

즉 `TrainedFDMPINN.grad_u(...)`는 다음을 반환해야 한다.

\[
\nabla_z u(\tau,z)
=
\begin{pmatrix}
\text{interp}(u_{z_1})(\tau,z_1,z_2)\\
\text{interp}(u_{z_2})(\tau,z_1,z_2)
\end{pmatrix}.
\]

value를 interpolate한 뒤 다시 numerical derivative를 취하는 것보다 gradient grid를 미리 만들고 gradient 자체를 interpolate하는 방식이 더 안정적이다.

---

## 6. 새 backend 이름

추천 backend 이름은

```text
fdm_pinn
```

이다.

이유는 이 backend가 constrained FDM이 아니라 `pinn`과 같은 unconstrained HJB를 FDM으로 푸는 benchmark이기 때문이다.

피해야 할 이름은 단순히

```text
fdm
```

이다. 나중에 constrained FDM, FDM policy iteration, Howard iteration 등을 추가하면 이름이 헷갈릴 수 있다.

따라서 config에서는 다음처럼 쓰는 것이 좋다.

```yaml
optimizer_backend: fdm_pinn
pipinn:
  policy_output_mode: foc_clip
fdm:
  n_z1: 81
  n_z2: 81
  n_tau: 240
  scheme: semi_implicit
  boundary: neumann
```

---

## 7. 코드 수정 범위 전체 요약

수정 범위는 크게 다음이다.

1. 새 파일 추가: `fdm_backend.py`
2. `schema.py`에 backend와 FDM config 추가
3. `experiments.py`에 backend branch 추가
4. `bridge_common.py`에 backend validation과 payload 추가
5. `native_selection.py`에 backend validation, default mode, label map 추가
6. `rank_sweep.py`에 backend 인식 추가
7. 필요하면 `cli.py` help 문구 또는 choices 추가
8. output metadata / benchmark notes / strategy label 정리

현재 코드의 public interface를 최대한 유지하는 방향이 좋다. 특히 FDM trainer가 `TrainedPIPINN`과 같은 method를 제공하면 `experiments.py` 수정량이 줄어든다.

---

## 8. 새 파일: `fdm_backend.py`

### 8.1 역할

새 파일 경로는 다음을 추천한다.

```text
dynalloc_v2/fdm_backend.py
```

이 파일은 neural network 학습 대신 FDM grid solve를 수행한다.

주요 public 함수는 다음이다.

```python
def train_fdm_pinn_policy(
    states_t,
    returns_tp1,
    cfg,
    transaction_cost,
    *,
    mean_model,
    transition,
    cross_est,
    cov_model,
    factor_repr,
    progress_label=None,
    tau_max=None,
    warm_start_from=None,
) -> TrainedFDMPINN:
    ...
```

signature는 기존 `train_pipinn_policy(...)`와 비슷하게 맞추는 것이 좋다. 그래야 `experiments.py`에서 backend 분기만 바꿔도 대부분의 인자를 그대로 넘길 수 있다.

`transaction_cost`, `returns_tp1`, `warm_start_from` 등은 FDM solve 자체에는 직접 필요하지 않을 수 있다. 그래도 caller compatibility를 위해 signature에 유지하고 내부에서 `del` 처리하면 된다.

---

### 8.2 기존 `pipinn_backend.py`에서 재사용할 것

FDM backend는 기존 `pipinn_backend.py`의 다음 요소를 재사용하는 것이 좋다.

```python
from .pipinn_backend import (
    PIPINNEnvFromPPGDPO,
    _select_training_covariance,
    _symmetrize_psd,
    _solve_unconstrained_foc_np,
    _clip_foc_policy_np,
)
```

재사용 이유는 다음이다.

- `PIPINNEnvFromPPGDPO`가 mean map, transition drift, \(\Sigma\), \(Q\), \(C\), state domain, risky cap 등을 이미 일관되게 구성한다.
- `_select_training_covariance(...)`가 `pinn`/`pipinn`에서 쓰는 training covariance 선택 로직을 이미 갖고 있다.
- `_solve_unconstrained_foc_np(...)`와 `_clip_foc_policy_np(...)`를 재사용하면 FDM-PINN의 policy extraction이 기존 `pinn`과 완전히 같아진다.

다만 private 함수 `_...`를 다른 module에서 import하는 것은 장기적으로 깔끔하지 않다. 장기 개선안으로는 다음과 같은 공통 유틸 파일을 만들 수 있다.

```text
control_utils.py
```

그 안에 다음을 옮긴다.

- `_symmetrize_psd`
- `_solve_unconstrained_foc_np`
- `_clip_foc_policy_np`
- `_solve_qp_long_only_budget_full` 또는 numpy variant

하지만 초기 구현에서는 private import를 사용해도 무방하다. benchmark 추가가 목적이므로 수정 범위를 작게 유지하는 편이 낫다.

---

### 8.3 `TrainedFDMPINN` class

FDM trainer는 기존 `TrainedPIPINN`과 동일한 public interface를 갖는 것이 좋다.

```python
class TrainedFDMPINN:
    def __init__(...):
        self.env = env
        self.state_columns = list(env.state_columns)
        self.asset_columns = list(env.asset_columns)
        self.train_objective = ...
        self.train_seed = ...
        self.train_history = ...
        self.best_validation_loss = ...
        self.solution = ...

    def grad_u(self, state_row, *, tau=None) -> np.ndarray:
        ...

    def estimate_costates(self, state_row, *, wealth=1.0, tau0=None) -> CostateEstimate:
        ...

    def policy_weights(self, state_row, *, covariance=None, cross_mat=None, tau=None) -> np.ndarray:
        ...

    def policy_weights_with_debug(self, state_row, *, covariance=None, cross_mat=None, tau=None):
        ...
```

이 interface가 중요한 이유는 `experiments.py`가 현재 `pipinn`/`pinn` trainer에 대해 다음을 호출하기 때문이다.

```python
trainer.policy_weights(state_row, tau=tau_remaining)
trainer.estimate_costates(state_row, tau0=tau_remaining)
trainer.policy_weights_with_debug(
    state_row,
    covariance=cov_eval,
    cross_mat=cross_mat,
    tau=tau_remaining,
)
```

FDM trainer도 이 method들을 제공하면 backtest loop를 거의 그대로 재사용할 수 있다.

---

### 8.4 `CostateEstimate` 반환

기존 `TrainedPIPINN.estimate_costates(...)`는 다음 구조를 쓴다.

```python
CostateEstimate(
    JX=1.0 / wealth,
    JXX=-gamma / (wealth * wealth),
    JXY=grad / wealth,
    closed_form=True,
)
```

FDM에서도 동일하게 반환하면 된다.

이유는 log value \(u\)의 state gradient가 다음 관계를 갖기 때문이다.

\[
J_X=\frac1x,
\qquad
J_{XX}=-\frac{\gamma}{x^2},
\qquad
J_{XY}=\frac{\nabla_z u}{x}.
\]

따라서 FDM의 `estimate_costates(...)`는 기존 `pinn`과 같은 closed-form costate interface를 제공할 수 있다.

---

### 8.5 `policy_weights_with_debug(...)`

FDM-PINN의 policy debug는 `pinn` branch와 거의 동일해야 한다.

구현 논리는 다음이다.

```python
cov = env.Sigma_train if covariance is None else symmetrize_psd(covariance)
cross = env.C_train if cross_mat is None else np.asarray(cross_mat)
mu = env.mean_map.predict_batch(x).reshape(-1)
grad_raw = self.grad_u(state_row, tau=tau)
hedge_signal = cross @ grad_raw
a_vec = mu + hedge_signal
pi_unc = _solve_unconstrained_foc_np(cov, a_vec, gamma=env.gamma)
w = _clip_foc_policy_np(pi_unc, cap=env.risky_cap, long_only=env.long_only)
return w, debug_dict
```

`debug_dict`는 기존 `pinn`과 맞추는 것이 좋다.

포함할 key 예시:

```python
{
    'hedge_signal': hedge_signal,
    'mu_term': mu_term,
    'hedge_term': hedge_signal,
    'a_vec': a_vec,
    'pi_unc': pi_unc,
    'pi_pre_clip': pi_unc,
    'risky_sum_before_clip': float(np.nansum(pi_unc)),
    'risky_sum_after_clip': float(np.nansum(w)),
    'risky_cap': float(env.risky_cap),
    'long_only_clip': bool(env.long_only),
    'neg_jxx': float(env.gamma),
    'neg_jxx_is_gamma': True,
    'control_update_space': 'foc_clip',
    'closed_form_costates': True,
    'grad_training': grad_raw,
    'grad_raw': grad_raw,
    'fdm_grid_shape': ...,
    'fdm_scheme': ...,
    'fdm_boundary': ...,
}
```

이렇게 하면 monthly output에서 hedge signal, costate norm 등 기존 logging 구조와 맞는다.

---

### 8.6 `FDMGridSolution` dataclass

FDM solution을 보관하기 위해 dataclass를 두는 것이 좋다.

```python
@dataclass
class FDMGridSolution:
    tau_grid: np.ndarray
    z1_grid: np.ndarray
    z2_grid: np.ndarray
    u_grid: np.ndarray
    grad_z1_grid: np.ndarray
    grad_z2_grid: np.ndarray
    scheme: str
    boundary: str
    diagnostics: dict[str, Any]
```

state dimension이 1D인 경우까지 지원하고 싶다면 일반화해서

```python
z_grids: list[np.ndarray]
grad_grids: list[np.ndarray]
```

로 둘 수 있다. 하지만 초기 구현은 PLS rank 2 benchmark가 목적이므로 2D 전용으로 단순하게 시작하는 편이 낫다.

---

### 8.7 state dimension validation

FDM은 curse of dimensionality 때문에 state dimension을 제한해야 한다.

초기 구현에서는 다음처럼 하는 것이 안전하다.

```python
if env.n_states != 2:
    raise ValueError("fdm_pinn currently supports exactly 2 state variables.")
```

조금 더 유연하게 하려면 1D와 2D만 허용한다.

```python
if env.n_states > int(cfg.fdm.max_state_dim):
    raise ValueError("FDM benchmark supports only low-dimensional state spaces.")
```

현재 사용 목적이 PLS rank 2라면 `exactly 2`로 시작해도 된다. 다만 rank 1 실험 가능성을 생각하면 `1 or 2`를 지원하는 구조도 좋다.

---

## 9. `schema.py` 수정 계획

현재 `schema.py`에서 backend는 다음으로 제한되어 있다.

```python
optimizer_backend: Literal['ppgdpo', 'pipinn', 'pinn'] = 'ppgdpo'
```

이를 다음처럼 바꿔야 한다.

```python
optimizer_backend: Literal['ppgdpo', 'pipinn', 'pinn', 'fdm_pinn'] = 'ppgdpo'
```

---

### 9.1 `FDMConfig` 추가

새 config class를 추가한다.

```python
class FDMConfig(BaseModel):
    n_z1: int = 81
    n_z2: int = 81
    n_tau: int = 240
    scheme: Literal['explicit', 'semi_implicit'] = 'semi_implicit'
    boundary: Literal['neumann'] = 'neumann'
    max_state_dim: int = 2
    store_full_solution: bool = True
    compute_gradient_grid: bool = True
    clip_state_to_domain: bool = True
    linear_solver: Literal['spsolve'] = 'spsolve'
```

더 자세히 관리하고 싶으면 다음도 추가할 수 있다.

```python
    drift_difference: Literal['upwind', 'central'] = 'upwind'
    diffusion_difference: Literal['central'] = 'central'
    mixed_derivative: Literal['central'] = 'central'
    regular_grid_interpolation: Literal['linear', 'nearest'] = 'linear'
    min_dtau: float = 1.0e-12
    psd_floor: float = 1.0e-10
```

하지만 초기에는 너무 많은 옵션을 열지 않는 것이 좋다. benchmark 재현성을 위해 기본값을 고정하는 편이 낫다.

---

### 9.2 `Config`에 `fdm` 추가

```python
fdm: FDMConfig = Field(default_factory=FDMConfig)
```

---

### 9.3 validator 수정

현재 validator는 `pinn`이면 `pipinn.policy_output_mode='foc_clip'`을 강제한다.

```python
if backend == 'pinn':
    if mode != 'foc_clip':
        raise ValueError(...)
```

FDM-PINN도 동일하게 강제해야 한다.

```python
if backend in {'pinn', 'fdm_pinn'}:
    if mode != 'foc_clip':
        raise ValueError(
            "optimizer_backend='pinn' or 'fdm_pinn' requires pipinn.policy_output_mode='foc_clip'. "
            "These backends solve an unconstrained HJB and use FOC-derived policy plus clipping."
        )
```

이유는 `fdm_pinn`에서 `pure_qp`를 허용하면 benchmark 정의가 바뀌기 때문이다. 그렇게 되면 “unconstrained FDM + constrained QP extraction”이 되어버려 현재 목표와 달라진다.

---

## 10. `experiments.py` 수정 계획

현재 `experiments.py`는 backend가 `pipinn` 또는 `pinn`일 때 `train_pipinn_policy(...)`를 호출한다.

현재 import는 대략 다음이다.

```python
from .pipinn_backend import train_pipinn_policy
```

새 backend를 추가하면 다음 import가 필요하다.

```python
from .fdm_backend import train_fdm_pinn_policy
```

---

### 10.1 helper set 추가

반복되는 조건문을 줄이기 위해 helper를 두는 것이 좋다.

```python
VALUE_GRAD_BACKENDS = {'pipinn', 'pinn', 'fdm_pinn'}
NEURAL_VALUE_BACKENDS = {'pipinn', 'pinn'}
FOC_CLIP_BACKENDS = {'pinn', 'fdm_pinn'}
```

또는 함수형으로:

```python
def _is_value_backend(backend: str) -> bool:
    return str(backend).lower() in {'pipinn', 'pinn', 'fdm_pinn'}


def _is_neural_value_backend(backend: str) -> bool:
    return str(backend).lower() in {'pipinn', 'pinn'}


def _requires_foc_clip(backend: str) -> bool:
    return str(backend).lower() in {'pinn', 'fdm_pinn'}
```

이렇게 해야 `backend in {'pipinn','pinn'}`가 여러 곳에 흩어지는 것을 줄일 수 있다.

---

### 10.2 `_fit_dynamic_policy_backend(...)`

현재 구조는 다음과 같다.

```python
backend = _optimizer_backend(cfg)
if backend in {'pipinn', 'pinn'}:
    trainer = train_pipinn_policy(..., training_mode=backend)
else:
    trainer = train_warmup_policy(...)
```

이를 다음처럼 바꾼다.

```python
backend = _optimizer_backend(cfg)
if backend in {'pipinn', 'pinn'}:
    trainer = train_pipinn_policy(
        ...,
        tau_max=tau_max,
        warm_start_from=prev_trainer,
        training_mode=backend,
    )
elif backend == 'fdm_pinn':
    trainer = train_fdm_pinn_policy(
        state_train,
        ret_train_next,
        cfg,
        transaction_cost=transaction_cost,
        mean_model=mean_model,
        transition=transition,
        cross_est=cross_est,
        cov_model=cov_model,
        factor_repr=factor_repr,
        progress_label=progress_label,
        tau_max=tau_max,
        warm_start_from=prev_trainer,
    )
else:
    trainer = train_warmup_policy(...)
```

FDM은 warm-start를 처음에는 무시해도 된다. 하지만 signature에는 남겨서 future extension을 쉽게 한다.

---

### 10.3 `_resolve_pipinn_output_dir(...)`

현재 함수는 backend가 `pipinn` 또는 `pinn`이 아니면 base output dir를 반환한다.

```python
if _optimizer_backend(cfg) not in {'pipinn', 'pinn'}:
    return base_dir
```

FDM-PINN도 별도 output subdir를 쓸 수 있게 하려면 다음처럼 확장한다.

```python
if _optimizer_backend(cfg) not in {'pipinn', 'pinn', 'fdm_pinn'}:
    return base_dir
```

다만 함수 이름이 `_resolve_pipinn_output_dir`라서 FDM까지 포함하면 이름이 어색해진다. 이름을 바꾸려면 다음이 낫다.

```python
_resolve_value_backend_output_dir
```

하지만 이름 변경은 수정 범위가 늘어나므로 초기에는 기존 이름을 유지해도 된다.

FDM의 output tag는 `pipinn` hyperparameter가 아니라 `fdm` grid hyperparameter를 반영해야 한다. 예:

```text
fdm_nz1-81__nz2-81__ntau-240__scheme-semi_implicit
```

초기에는 auto subdir를 끄고 base output dir로 가도 된다.

---

### 10.4 run-level validation

현재 `_run_ppgdpo_experiment` 안에 다음 검증이 있다.

```python
if backend == 'pinn' and pipinn_policy_output_mode != 'foc_clip':
    raise ValueError(...)
```

이를 다음처럼 바꾼다.

```python
if backend in {'pinn', 'fdm_pinn'} and pipinn_policy_output_mode != 'foc_clip':
    raise ValueError(
        "optimizer_backend='pinn' or 'fdm_pinn' requires pipinn.policy_output_mode='foc_clip'."
    )
```

---

### 10.5 training log flags

현재:

```python
save_training_logs = backend in {'pipinn', 'pinn'} and cfg.pipinn.save_training_logs
show_progress = backend in {'pipinn', 'pinn'} and cfg.pipinn.show_progress
```

FDM은 epoch training이 없으므로 두 가지 선택지가 있다.

#### 선택 A: FDM은 training log 저장 안 함

```python
save_training_logs = backend in {'pipinn', 'pinn'} and ...
show_progress = backend in {'pipinn', 'pinn', 'fdm_pinn'} and ...
```

FDM progress를 따로 보고 싶으면 `cfg.fdm.show_progress` 같은 option을 만들면 된다.

#### 선택 B: FDM diagnostic log 저장

FDM trainer에 `train_history`를 만들고 다음 정보를 저장한다.

- grid size,
- dtau,
- max_abs_u,
- max_abs_grad,
- nan_count,
- scheme,
- boundary,
- solve time,
- linear solver info.

초기 구현에서는 선택 A가 낫다. 하지만 논문 benchmark라면 선택 B가 재현성에 좋다.

---

### 10.6 refit 시 `tau_max` 전달

현재 refit 시 trainer 호출에서

```python
tau_max=tau_remaining if backend in {'pipinn', 'pinn'} else None
```

으로 되어 있다.

FDM도 value function을 \([0,\tau_{\max}]\)에서 풀어야 하므로 포함해야 한다.

```python
tau_max=tau_remaining if backend in {'pipinn', 'pinn', 'fdm_pinn'} else None
```

또는 helper 사용:

```python
tau_max=tau_remaining if _is_value_backend(backend) else None
```

---

### 10.7 covariance evaluation branch

현재 evaluation covariance branch는 다음이다.

```python
if backend in {'pipinn', 'pinn'}:
    trainer_asset_order = list(mean_model.assets)
    asset_perm = [ret_source_order.index(a) for a in trainer_asset_order]
    cov_full = np.asarray(cross_est.current_asset_cov())[np.ix_(asset_perm, asset_perm)]
else:
    asset_perm = [ret_source_order.index(a) for a in returns.columns]
    cov_fc = cov_model.predict(...)
    cov_full = cov_fc.asset_cov
```

FDM-PINN도 `pinn`과 동일하게 joint estimator의 covariance와 cross를 써야 한다.

수정:

```python
if backend in {'pipinn', 'pinn', 'fdm_pinn'}:
    ...
```

이유는 FDM-PINN이 `pinn`의 benchmark이므로 훈련과 평가에서 covariance/cross 처리도 동일해야 하기 때문이다.

---

### 10.8 `policy_weights` 호출

현재 rebalance branch:

```python
if backend in {'pipinn', 'pinn'}:
    pgdpo_w = trainer.policy_weights(state_row, tau=tau_remaining)
else:
    pgdpo_w = trainer.policy_weights(state_row)
```

FDM-PINN도 \(\tau\)가 필요하므로 포함한다.

```python
if backend in {'pipinn', 'pinn', 'fdm_pinn'}:
    pgdpo_w = trainer.policy_weights(state_row, tau=tau_remaining)
else:
    pgdpo_w = trainer.policy_weights(state_row)
```

---

### 10.9 `estimate_costates` 호출

현재:

```python
if backend in {'pipinn', 'pinn'}:
    last_costates = trainer.estimate_costates(state_row, tau0=tau_remaining)
else:
    last_costates = trainer.estimate_costates(state_row)
```

FDM-PINN도 포함한다.

```python
if backend in {'pipinn', 'pinn', 'fdm_pinn'}:
    last_costates = trainer.estimate_costates(state_row, tau0=tau_remaining)
else:
    last_costates = trainer.estimate_costates(state_row)
```

---

### 10.10 cross-mode ablation branch

현재 cross-mode branch는 다음과 같다.

```python
policy_output_mode = str(getattr(cfg.pipinn, 'policy_output_mode', 'pure_qp')).lower()
if backend in {'pipinn', 'pinn'} and policy_output_mode in {'pure_qp', 'foc_clip'}:
    ppgdpo_w, proj_debug = trainer.policy_weights_with_debug(
        state_row,
        covariance=cov_eval,
        cross_mat=cross_mat,
        tau=tau_remaining,
    )
else:
    ppgdpo_w, proj_debug = solve_ppgdpo_projection(...)
```

FDM-PINN도 이 branch를 타야 한다.

```python
if backend in {'pipinn', 'pinn', 'fdm_pinn'} and policy_output_mode in {'pure_qp', 'foc_clip'}:
    ...
```

단, `fdm_pinn`은 `policy_output_mode='foc_clip'`만 허용해야 한다. 따라서 실제로는 FDM-PINN에서는 `foc_clip` branch만 실행된다.

---

### 10.11 factor variance logging

현재 `pipinn`/`pinn` branch에서는 `factor_var_dict = {}`로 둔다.

```python
if backend in {'pipinn', 'pinn'}:
    factor_var_dict = {}
else:
    factor_var_dict = {k: float(v) for k, v in cov_fc.factor_var.items()}
```

FDM-PINN도 joint estimator branch를 쓰므로 `cov_fc`가 없을 수 있다. 따라서 포함해야 한다.

```python
if backend in {'pipinn', 'pinn', 'fdm_pinn'}:
    factor_var_dict = {}
else:
    factor_var_dict = ...
```

---

### 10.12 benchmark notes payload

현재 `_write_benchmark_notes(...)` 호출에서

```python
pipinn_policy_output_mode=... if backend in {'pipinn', 'pinn'} else None
```

로 되어 있다.

FDM-PINN도 `foc_clip` mode를 기록해야 한다.

```python
pipinn_policy_output_mode=... if backend in {'pipinn', 'pinn', 'fdm_pinn'} else None
```

다만 이름이 `pipinn_policy_output_mode`라서 어색하다. 초기에는 그대로 둬도 되지만, 장기적으로는 `value_policy_output_mode` 같은 이름이 더 낫다.

---

## 11. strategy metadata 수정 계획

현재 `_strategy_metadata(...)`는 backend가 `pipinn` 또는 `pinn`일 때 다음처럼 처리한다.

```python
if backend in {'pipinn', 'pinn'}:
    base = 'pinn' if backend == 'pinn' else 'pipinn'
    label = 'PINN' if backend == 'pinn' else 'PI-PINN'
```

FDM-PINN을 추가하면 다음 mapping이 필요하다.

```python
def _value_backend_label_parts(backend: str) -> tuple[str, str, str]:
    if backend == 'pinn':
        return 'pinn', 'PINN', 'FOC-derived unconstrained policy with long-only/risky-cap clipping'
    if backend == 'pipinn':
        return 'pipinn', 'PI-PINN', 'value-gradient pure-QP/projection policy'
    if backend == 'fdm_pinn':
        return 'fdm_pinn', 'FDM-PINN', 'FDM-solved unconstrained HJB with FOC clipping'
```

FDM strategy labels는 다음을 추천한다.

| cross mode | strategy | display | legacy label |
|---|---|---|---|
| estimated | `fdm_pinn` | `FDM-PINN` 또는 `fdm_pinn` | `ppgdpo` |
| zero | `fdm_pinn_zero` | `FDM-PINN (No-hedge)` 또는 `fdm_pinn_zero` | `ppgdpo_zero` |
| regime_gated | `fdm_pinn_regime_gated` | `FDM-PINN (Regime-gated)` 또는 `fdm_pinn_regime_gated` | `ppgdpo_regime_gated` |

현재 코드 스타일이 display에도 lowercase label을 쓰고 있으므로, 최소 수정은 다음이다.

```python
base = 'fdm_pinn'
label = 'FDM-PINN'
```

mapping:

```python
'estimated': ('fdm_pinn', 'fdm_pinn', 'ppgdpo', 'FDM-PINN FDM-solved unconstrained HJB with FOC clipping with estimated cross')
'zero': ('fdm_pinn_zero', 'fdm_pinn_zero', 'ppgdpo_zero', ...)
' regime_gated': ('fdm_pinn_regime_gated', 'fdm_pinn_regime_gated', 'ppgdpo_regime_gated', ...)
```

조금 더 보기 좋은 display를 쓰고 싶다면:

```python
'estimated': ('fdm_pinn', 'FDM-PINN', 'ppgdpo', ...)
'zero': ('fdm_pinn_zero', 'FDM-PINN (No-hedge)', 'ppgdpo_zero', ...)
'regime_gated': ('fdm_pinn_regime_gated', 'FDM-PINN (Regime-gated)', 'ppgdpo_regime_gated', ...)
```

---

## 12. `_write_benchmark_notes(...)` 수정 계획

현재 `_write_benchmark_notes(...)`는 backend가 `pipinn`/`pinn`이면 base를 `pipinn` 또는 `pinn`으로 잡는다.

FDM-PINN도 추가해야 한다.

```python
if backend_norm in {'pipinn', 'pinn', 'fdm_pinn'}:
    if backend_norm == 'pinn':
        base = 'pinn'
    elif backend_norm == 'pipinn':
        base = 'pipinn'
    else:
        base = 'fdm_pinn'
```

strategy label map에 다음을 추가한다.

```python
strategy_label_map.update({
    base: 'ppgdpo',
    f'{base}_zero': 'ppgdpo_zero',
    f'{base}_regime_gated': 'ppgdpo_regime_gated',
    f'{base}_traincov_diag': 'pgdpo',
})
```

FDM-PINN note는 다음처럼 적는 것이 좋다.

```text
fdm_pinn / fdm_pinn_zero / fdm_pinn_regime_gated solve the same unconstrained log-HJB as the traditional PINN using finite differences, then use FOC-derived unconstrained policy followed by long-only/risky-cap clipping.
```

---

## 13. `bridge_common.py` 수정 계획

현재 `_build_v2_config_dict(...)`에서 backend validation은 다음이다.

```python
if backend not in {'ppgdpo', 'pipinn', 'pinn'}:
    raise ValueError(...)
```

이를 다음으로 바꾼다.

```python
if backend not in {'ppgdpo', 'pipinn', 'pinn', 'fdm_pinn'}:
    raise ValueError(...)
```

---

### 13.1 payload에 `pipinn` block 포함

현재:

```python
if backend in {'pipinn', 'pinn'} or pipinn_payload is not None:
    merged_pipinn = _default_pipinn_payload()
    ...
```

FDM-PINN도 tau mode, policy output mode, domain 설정 등을 위해 `pipinn` block을 재사용할 수 있다. 따라서:

```python
if backend in {'pipinn', 'pinn', 'fdm_pinn'} or pipinn_payload is not None:
    ...
```

---

### 13.2 `foc_clip` 강제

현재:

```python
if backend == 'pinn':
    ... policy_output_mode='foc_clip' ...
```

수정:

```python
if backend in {'pinn', 'fdm_pinn'}:
    raw_pipinn_payload = dict(pipinn_payload or {})
    if raw_pipinn_payload.get('policy_output_mode') is None:
        merged_pipinn['policy_output_mode'] = 'foc_clip'
    if str(merged_pipinn.get('policy_output_mode')).lower() != 'foc_clip':
        raise ValueError(
            "optimizer_backend='pinn' or 'fdm_pinn' requires pipinn.policy_output_mode='foc_clip'."
        )
```

---

### 13.3 FDM block 추가

bridge payload에 `fdm` block을 넣을 수 있다.

```python
if backend == 'fdm_pinn':
    payload['fdm'] = {
        'n_z1': 81,
        'n_z2': 81,
        'n_tau': 240,
        'scheme': 'semi_implicit',
        'boundary': 'neumann',
        'max_state_dim': 2,
    }
```

또는 함수 인자로 `fdm_payload`를 추가한다.

```python
def _build_v2_config_dict(..., fdm_payload: dict[str, Any] | None = None):
    ...
```

초기에는 default block만 두어도 충분하다.

---

## 14. `native_selection.py` 수정 계획

`native_selection.py`에는 backend-specific validation과 label map이 여러 곳 있다.

---

### 14.1 `_strategy_label_map_for_backend(...)`

현재:

```python
if backend_norm == 'pinn':
    out.update({'pinn': 'ppgdpo', ...})
elif backend_norm == 'pipinn':
    out.update({'pipinn': 'ppgdpo', ...})
else:
    out.update({'ppgdpo': 'ppgdpo', ...})
```

FDM-PINN 추가:

```python
elif backend_norm == 'fdm_pinn':
    out.update({
        'fdm_pinn': 'ppgdpo',
        'fdm_pinn_zero': 'ppgdpo_zero',
        'fdm_pinn_regime_gated': 'ppgdpo_regime_gated',
    })
```

---

### 14.2 `_comparison_benchmark_notes_for_backend(...)`

현재:

```python
if backend_norm == 'pinn':
    variants = ['pinn', 'pinn_zero', 'pinn_regime_gated']
    method_note = 'traditional PINN value-gradient FOC policy with long-only/risky-cap clipping'
elif backend_norm == 'pipinn':
    variants = ['pipinn', 'pipinn_zero', 'pipinn_regime_gated']
    method_note = 'PI-PINN value-gradient policy output'
else:
    variants = ['ppgdpo', ...]
```

추가:

```python
elif backend_norm == 'fdm_pinn':
    variants = ['fdm_pinn', 'fdm_pinn_zero', 'fdm_pinn_regime_gated']
    method_note = 'finite-difference solution of the unconstrained PINN log-HJB with FOC clipping'
```

---

### 14.3 default eval mode

현재 native selection에서 backend가 `pinn`이면 default mode를 `foc_clip`으로 둔다.

```python
if selection_eval_mode is None:
    selection_eval_mode = 'foc_clip' if backend_norm == 'pinn' else 'pure_qp'
if pipinn_policy_output_mode is None:
    pipinn_policy_output_mode = 'foc_clip' if backend_norm == 'pinn' else 'pure_qp'
```

FDM-PINN도 동일하게 해야 한다.

```python
if selection_eval_mode is None:
    selection_eval_mode = 'foc_clip' if backend_norm in {'pinn', 'fdm_pinn'} else 'pure_qp'
if pipinn_policy_output_mode is None:
    pipinn_policy_output_mode = 'foc_clip' if backend_norm in {'pinn', 'fdm_pinn'} else 'pure_qp'
```

---

### 14.4 incompatible mode validation

현재:

```python
if backend_norm == 'pinn':
    if selection_eval_mode_norm != 'foc_clip': ...
    if pipinn_policy_output_mode_norm != 'foc_clip': ...
```

수정:

```python
if backend_norm in {'pinn', 'fdm_pinn'}:
    if selection_eval_mode_norm != 'foc_clip':
        raise ValueError(
            "selection_optimizer_backend='pinn' or 'fdm_pinn' requires --selection-eval-mode foc_clip."
        )
    if pipinn_policy_output_mode_norm != 'foc_clip':
        raise ValueError(
            "selection_optimizer_backend='pinn' or 'fdm_pinn' requires --pipinn-policy-output-mode foc_clip."
        )
```

---

### 14.5 full config conversion validation

현재 around `_lite_to_full_config` 쪽에서:

```python
if str(lite_cfg.optimizer_backend).lower() == 'pinn':
    if str(lite_cfg.pipinn_policy_output_mode).lower() != 'foc_clip':
        raise ValueError(...)
```

수정:

```python
if str(lite_cfg.optimizer_backend).lower() in {'pinn', 'fdm_pinn'}:
    ...
```

---

### 14.6 config export block

현재 generated config에서 `pipinn` payload를 backend가 `pipinn`/`pinn`일 때만 쓰는 부분이 있다.

예:

```python
'pipinn': _pipinn_payload_from_lite_cfg(lite_cfg) if backend in {'pipinn', 'pinn'} else None
```

FDM-PINN도 포함해야 한다.

```python
'pipinn': _pipinn_payload_from_lite_cfg(lite_cfg) if backend in {'pipinn', 'pinn', 'fdm_pinn'} else None
```

그리고 FDM config block도 넣을 수 있다.

```python
'fdm': _fdm_payload_from_lite_cfg(lite_cfg) if backend == 'fdm_pinn' else None
```

처음에는 schema default에 의존해도 된다.

---

## 15. `rank_sweep.py` 수정 계획

현재 rank sweep에서는 backend를 다음 set으로 인식한다.

```python
if entry_backend in {'ppgdpo', 'pipinn', 'pinn'}:
    payload['optimizer_backend'] = entry_backend
```

수정:

```python
if entry_backend in {'ppgdpo', 'pipinn', 'pinn', 'fdm_pinn'}:
    payload['optimizer_backend'] = entry_backend
```

현재 `pinn`이면 foc_clip을 강제한다.

```python
if entry_backend == 'pinn':
    payload.setdefault('pipinn', {})
    payload['pipinn']['policy_output_mode'] = 'foc_clip'
```

수정:

```python
if entry_backend in {'pinn', 'fdm_pinn'}:
    payload.setdefault('pipinn', {})
    payload['pipinn']['policy_output_mode'] = 'foc_clip'
```

device logging도 현재:

```python
cfg.pipinn.device if backend in {'pipinn', 'pinn'} else cfg.ppgdpo.device
```

FDM-PINN은 neural device를 안 쓸 수 있다. 여기서 선택지는 두 가지다.

1. FDM은 CPU sparse solver이므로 `cfg.ppgdpo.device` 또는 `'cpu'`로 기록.
2. value backend로 분류해 `cfg.pipinn.device`를 기록.

FDM solver는 SciPy sparse 기반이면 CPU이므로 metadata에는 `cpu` 또는 `fdm_cpu`가 더 정확하다. 다만 최소 수정은:

```python
cfg.pipinn.device if backend in {'pipinn', 'pinn'} else cfg.ppgdpo.device
```

를 그대로 두고, FDM은 else로 가게 해도 된다. 그러나 backend가 `fdm_pinn`인 경우 `ppgdpo.device`가 의미적으로 맞지 않을 수 있으므로 별도 처리하는 것이 낫다.

```python
if backend == 'fdm_pinn':
    device_label = 'cpu'
elif backend in {'pipinn', 'pinn'}:
    device_label = cfg.pipinn.device
else:
    device_label = cfg.ppgdpo.device
```

---

## 16. `cli.py` 수정 계획

`cli.py`에서 backend argument가 단순 문자열이면 큰 수정이 없을 수 있다. 다만 help text나 choices를 명시하고 있다면 `fdm_pinn`을 추가해야 한다.

확인할 항목:

- `--selection-optimizer-backend`
- `--pipinn-policy-output-mode`
- `--selection-eval-mode`

만약 choices가 다음처럼 되어 있다면

```python
choices=['ppgdpo', 'pipinn', 'pinn']
```

수정:

```python
choices=['ppgdpo', 'pipinn', 'pinn', 'fdm_pinn']
```

또한 help 문구에 다음을 추가한다.

```text
fdm_pinn: finite-difference benchmark for the unconstrained PINN HJB with FOC clipping
```

---

## 17. config 파일 수정 예시

새 backend를 쓰는 YAML 예시는 다음과 같다.

```yaml
optimizer_backend: fdm_pinn

pipinn:
  policy_output_mode: foc_clip
  eval_tau_mode: maturity_constant
  eval_tau_maturity_years: 1
  x_domain_quantile_low: 0.001
  x_domain_quantile_high: 0.999
  x_domain_buffer: 0.20

fdm:
  n_z1: 81
  n_z2: 81
  n_tau: 240
  scheme: semi_implicit
  boundary: neumann
  max_state_dim: 2
```

여기서 `pipinn` block을 완전히 없애지 않는 이유는 `fdm_pinn`이 `pinn`과 같은 tau policy, domain policy, policy output mode를 공유하기 때문이다. 다만 장기적으로는 `pipinn`이 아니라 `value_backend` 같은 config namespace로 정리하는 것이 더 깔끔하다.

---

## 18. FDM diagnostics 및 metadata

FDM benchmark는 grid와 scheme 선택에 민감할 수 있다. 따라서 결과 재현성을 위해 다음을 반드시 기록하는 것이 좋다.

```text
fdm_scheme
fdm_boundary
fdm_n_z1
fdm_n_z2
fdm_n_tau
fdm_tau_max
fdm_dtau
fdm_z1_min
fdm_z1_max
fdm_z2_min
fdm_z2_max
fdm_dz1
fdm_dz2
fdm_max_abs_u
fdm_max_abs_grad_z1
fdm_max_abs_grad_z2
fdm_nan_count
fdm_inf_count
fdm_linear_solver
fdm_elapsed_seconds
```

`train_history`에 한 줄짜리 diagnostics를 넣을 수도 있다.

예:

```python
train_history = [{
    'solver': 'fdm_pinn',
    'scheme': 'semi_implicit',
    'boundary': 'neumann',
    'n_tau': n_tau,
    'n_z1': n_z1,
    'n_z2': n_z2,
    'dtau': dtau,
    'dz1': dz1,
    'dz2': dz2,
    'max_abs_u': max_abs_u,
    'max_abs_grad': max_abs_grad,
    'nan_count': nan_count,
    'elapsed_seconds': elapsed,
}]
```

기존 `_write_pipinn_training_log(...)`를 그대로 재사용할 수 있게 하려면 column 이름을 맞춰야 하지만, FDM은 epoch가 없으므로 별도 `_write_fdm_training_log(...)`가 더 자연스럽다.

초기 구현에서는 FDM log를 저장하지 않아도 되지만, 논문 benchmark로 쓰려면 diagnostics 파일을 저장하는 것이 좋다.

---

## 19. FDM grid convergence test 계획

FDM을 benchmark로 쓰려면 grid convergence check가 필요하다.

추천 grid set:

```text
41 x 41 x 120
61 x 61 x 180
81 x 81 x 240
101 x 101 x 300
```

여기서 앞의 두 숫자는 state grid, 마지막 숫자는 time grid다.

비교할 항목:

1. \(u(\tau,z)\) 값의 변화,
2. \(\nabla_z u(\tau,z)\)의 변화,
3. policy weight 변화,
4. hedge signal \(C\nabla_z u\) 변화,
5. backtest 성과 변화,
6. turnover 변화.

특히 policy는 gradient에 민감하므로 value convergence보다 gradient convergence가 더 중요하다.

---

## 20. Python library 선택

### 20.1 결론

이 프로젝트에서는 **SciPy sparse 기반 직접 구현**이 가장 적합하다.

이유는 다음이다.

1. PDE가 일반 heat equation이나 reaction-diffusion equation이 아니라, portfolio HJB의 custom Hamiltonian을 갖는다.
2. `pinn`과 정확히 같은 \(\mu\), \(b\), \(\Sigma\), \(Q\), \(C\), clipping policy를 써야 한다.
3. backtest loop와 `policy_weights_with_debug(...)` interface를 맞춰야 한다.
4. FDM의 목적이 범용 PDE simulation이 아니라, 현재 코드의 `pinn` benchmark이기 때문이다.

따라서 범용 PDE package를 쓰기보다 직접 operator를 구성하는 것이 통제 가능성이 높다.

---

### 20.2 SciPy sparse

추천 도구:

```python
scipy.sparse.diags
scipy.sparse.eye
scipy.sparse.kron
scipy.sparse.linalg.spsolve
scipy.interpolate.RegularGridInterpolator
```

공식 문서 기준으로 `scipy.sparse.diags`는 diagonals로 sparse matrix를 구성하는 함수이고, `scipy.sparse.linalg.spsolve`는 sparse linear system \(Ax=b\)를 푸는 함수다.

장점:

- FDM operator를 직접 구성할 수 있다.
- semi-implicit scheme의 sparse linear system을 풀기 좋다.
- 현재 코드와 통합이 쉽다.
- 불필요한 framework dependency가 없다.

단점:

- boundary condition, mixed derivative, upwind drift 등을 직접 구현해야 한다.

하지만 이 직접 구현 부담은 benchmark 정확성을 위해 감수할 만하다.

---

### 20.3 py-pde

`py-pde`는 PDE를 풀기 위한 Python package이고, fixed grid에서 finite difference로 differential operators를 계산하는 method-of-lines 스타일에 적합하다.

장점:

- 빠른 prototype에 좋다.
- regular grid PDE simulation을 쉽게 시작할 수 있다.

단점:

- custom HJB Hamiltonian과 기존 backtest interface를 맞추려면 wrapper가 많아질 수 있다.
- 현재 목표인 `pinn`과 동일한 policy extraction / debug output을 맞추기에는 직접 구현보다 통제성이 낮다.

결론: 연구 prototype으로는 가능하지만, 현재 benchmark backend로는 SciPy 직접 구현이 더 낫다.

---

### 20.4 FiPy

FiPy는 NIST에서 개발한 finite-volume PDE solver다.

장점:

- diffusion/convection PDE를 finite-volume 방식으로 풀기에 robust하다.
- boundary condition과 PDE term 구성이 framework화되어 있다.

단점:

- 현재 문제는 HJB Hamiltonian, log value gradient, portfolio policy extraction이 핵심이다.
- finite-volume framework에 맞추는 비용이 생긴다.
- 기존 `experiments.py` backtest 구조와 직접 연결하려면 별도 wrapper가 커질 수 있다.

결론: 가능은 하지만, 이 프로젝트에서는 SciPy sparse 직접 구현이 더 적합하다.

---

### 20.5 OSQP / CVXPY

현재 `fdm_pinn` 목표에는 QP가 필요 없다. 이유는 policy가 `pinn`과 동일하게 unconstrained FOC 후 clipping이기 때문이다.

하지만 나중에 constrained FDM을 만들 경우에는 각 grid point에서 다음 QP가 필요하다.

\[
\max_{\pi_i\ge0,\mathbf 1^\top\pi\le\ell}
\left\{
\pi^\top
\left[\mu(z)+C\nabla u\right]
-
\frac\gamma2\pi^\top\Sigma\pi
\right\}.
\]

이때 후보는:

- OSQP,
- CVXPY,
- 현재 코드의 projected-gradient QP solver.

대량 grid point에서 반복적으로 QP를 풀려면 CVXPY는 편하지만 무거울 수 있다. OSQP는 convex QP용 solver로 더 적합할 수 있다. 그러나 대규모 반복에서는 현재 코드의 `_solve_qp_long_only_budget_full(...)` 같은 custom projected-gradient solver를 vectorize하는 방식도 실용적이다.

---

## 21. FDM-PINN과 constrained FDM의 구분

이번 목표는 다음이다.

\[
\boxed{
\texttt{fdm\_pinn}: H_{\text{unc}}\text{를 사용한 unconstrained HJB FDM solve + FOC clipping}
}
\]

반면 constrained FDM은 다음 PDE를 푼다.

\[
 u_\tau
 =
 b^\top\nabla u
 +
 \frac12\operatorname{tr}(QD^2u)
 +
 \frac12\nabla u^\top Q\nabla u
 +
 (1-\gamma)
 \left[
r+H_{\mathcal A}(z,\nabla u)
\right],
\]

where

\[
H_{\mathcal A}(z,\nabla u)
=
\max_{\pi_i\ge0,\mathbf 1^\top\pi\le\ell}
\left\{
\pi^\top
\left[\mu(z)+C\nabla u\right]
-
\frac\gamma2\pi^\top\Sigma\pi
\right\}.
\]

이 constrained FDM은 현재 목표가 아니다. 나중에 만들 경우 이름은 다음처럼 분리하는 것이 좋다.

```text
fdm_constrained
```

또는

```text
fdm_pi
```

그때는 FDM policy evaluation + constrained QP policy improvement 구조, 즉 Howard policy iteration에 가까운 알고리즘이 된다.

---

## 22. 최소 구현 순서

### Phase 1: 최소 backend 추가

1. `schema.py`
   - `optimizer_backend`에 `fdm_pinn` 추가.
   - `FDMConfig` 추가.
   - `fdm_pinn`이면 `policy_output_mode='foc_clip'` 강제.

2. `fdm_backend.py`
   - `train_fdm_pinn_policy(...)` 추가.
   - `TrainedFDMPINN` 추가.
   - 2D FDM solver 추가.
   - gradient interpolation 추가.
   - `policy_weights_with_debug(...)`에서 기존 `pinn`과 동일한 FOC clipping 적용.

3. `experiments.py`
   - import 추가.
   - `_fit_dynamic_policy_backend(...)`에 `fdm_pinn` branch 추가.
   - `backend in {'pipinn','pinn'}` 조건 중 value-gradient backend에 해당하는 것들을 `fdm_pinn`까지 확장.
   - metadata label 추가.

4. `bridge_common.py`, `native_selection.py`, `rank_sweep.py`
   - backend validation set에 `fdm_pinn` 추가.
   - `fdm_pinn`이면 mode default와 validation을 `foc_clip`으로 설정.
   - label map 추가.

---

### Phase 2: diagnostics 추가

- FDM grid metadata 저장.
- max abs value / gradient 저장.
- NaN / Inf check.
- solve time 저장.
- optional FDM training log 저장.

---

### Phase 3: grid convergence test

- `41x41x120`, `61x61x180`, `81x81x240`, `101x101x300` 비교.
- value, gradient, policy, performance 비교.
- convergence가 확인된 grid를 benchmark default로 고정.

---

## 23. 예상되는 구현상 주의점

### 23.1 state dimension이 2가 아니면 실패하게 할 것

FDM은 state dimension이 올라가면 급격히 무거워진다. 초기 구현에서는 PLS rank 2 전용으로 두는 것이 안전하다.

```python
if env.n_states != 2:
    raise ValueError("fdm_pinn currently expects exactly two PLS state variables.")
```

---

### 23.2 state가 domain 밖으로 나가면 어떻게 할 것인가

평가 시점 state \(z_t\)가 FDM grid domain 밖으로 나갈 수 있다.

선택지:

1. domain 밖이면 nearest boundary로 clip,
2. extrapolate 허용,
3. error 발생.

benchmark 안정성을 위해 초기 구현은 clip을 추천한다.

```python
z_eval = np.clip(z_eval, env.x_min, env.x_max)
```

그리고 debug에 다음을 기록한다.

```python
'fdm_state_clipped': True or False
```

---

### 23.3 tau가 domain 밖으로 나가면 어떻게 할 것인가

평가 tau가 \([0,\tau_{\max}]\) 밖이면 clip한다.

```python
tau_eval = np.clip(tau_eval, 0.0, env.tau_max)
```

training/refit 때 `tau_max=tau_remaining`을 넘기면 일반적으로 평가 tau는 domain 안에 있어야 한다. 그래도 safety check는 필요하다.

---

### 23.4 covariance/cross alignment

FDM policy extraction에서 asset order와 state order는 기존 `pinn`과 같아야 한다.

- asset order: `mean_model.assets`
- state order: `cfg.state.columns`
- \(C\): asset x state shape
- \(\Sigma\): asset x asset shape

`experiments.py`의 current branch는 `pipinn`/`pinn`일 때 `mean_model.assets` 기준으로 covariance를 정렬한다. FDM도 이 branch를 타야 한다.

---

### 23.5 policy_output_mode는 반드시 `foc_clip`

`fdm_pinn`에서 `pure_qp`를 허용하면 결과 해석이 모호해진다.

따라서 schema, bridge, native_selection, rank_sweep 등 모든 path에서 `fdm_pinn`이면 `foc_clip`을 강제해야 한다.

---

### 23.6 FDM solve와 evaluation covariance의 차이

기존 `pinn`은 training PDE에서 `Sigma_train`, `C_train`을 쓰고, 평가 policy extraction에서 `cov_eval`, `cross_mat`를 override할 수 있다.

FDM-PINN도 동일해야 한다.

즉 FDM으로 구한 value gradient는 train-window PDE 기준이지만, 최종 policy는 evaluation 시점 covariance/cross를 사용한다.

\[
\pi_{\text{unc},t}
=
\frac1\gamma
\Sigma_{\text{eval},t}^{-1}
\left[
\mu(z_t)+C_{\text{eval},t}\nabla u(\tau_t,z_t)
\right].
\]

이렇게 해야 existing `pinn`과 공정하게 비교된다.

---

## 24. pseudo-code: `train_fdm_pinn_policy`

```python
def train_fdm_pinn_policy(
    states_t,
    returns_tp1,
    cfg,
    transaction_cost,
    *,
    mean_model,
    transition,
    cross_est,
    cov_model,
    factor_repr,
    progress_label=None,
    tau_max=None,
    warm_start_from=None,
):
    del returns_tp1, transaction_cost, warm_start_from

    sigma_train = _select_training_covariance(
        cfg=cfg,
        cov_model=cov_model,
        cross_est=cross_est,
        state_train=states_t,
        factor_train=factor_repr.factors if hasattr(factor_repr, 'factors') else pd.DataFrame(index=states_t.index),
        loadings=factor_repr.loadings,
        residual_var=factor_repr.residual_var,
    )

    tau_cap = int(np.ceil(float(tau_max))) if tau_max is not None else int(cfg.ppgdpo.horizon_steps)
    tau_max_eff = float(max(tau_cap, 1))

    env = PIPINNEnvFromPPGDPO(
        mean_model=mean_model,
        transition=transition,
        cross_est=cross_est,
        states_t=states_t,
        sigma_train=sigma_train,
        cfg=cfg,
        tau_max=tau_max_eff,
        device='cpu',
        dtype=torch.float64,
    )

    if env.n_states != 2:
        raise ValueError("fdm_pinn currently supports exactly 2 state variables")

    solution = solve_unconstrained_log_hjb_fdm_2d(env, cfg.fdm)

    return TrainedFDMPINN(
        env=env,
        solution=solution,
        train_objective=-solution.diagnostics.get('residual_proxy', np.nan),
        train_seed=int(cfg.ppgdpo.train_seed),
        train_history=[solution.diagnostics],
    )
```

---

## 25. pseudo-code: FDM solver

```python
def solve_unconstrained_log_hjb_fdm_2d(env, fdm_cfg):
    n1 = int(fdm_cfg.n_z1)
    n2 = int(fdm_cfg.n_z2)
    nt = int(fdm_cfg.n_tau)

    z1 = np.linspace(env.x_min[0], env.x_max[0], n1)
    z2 = np.linspace(env.x_min[1], env.x_max[1], n2)
    tau = np.linspace(0.0, env.tau_max, nt + 1)

    dz1 = z1[1] - z1[0]
    dz2 = z2[1] - z2[0]
    dt = tau[1] - tau[0]

    Z1, Z2 = np.meshgrid(z1, z2, indexing='ij')
    grid_points = np.column_stack([Z1.ravel(), Z2.ravel()])

    L = build_linear_operator_L(env, z1, z2, boundary=fdm_cfg.boundary, drift_difference='upwind')
    A = scipy.sparse.eye(n1 * n2, format='csr') - dt * L

    u = np.zeros((nt + 1, n1, n2), dtype=float)

    for n in range(nt):
        grad1, grad2 = finite_difference_gradient(u[n], dz1, dz2, boundary=fdm_cfg.boundary)
        grad = np.column_stack([grad1.ravel(), grad2.ravel()])

        mu = env.mean_map.predict_batch(grid_points)
        a = mu + grad @ env.C_train.T
        sigma_inv_a = solve_sigma_batch(env.Sigma_train, a)
        ham = 0.5 / env.gamma * np.sum(a * sigma_inv_a, axis=1)

        quad = 0.5 * np.einsum('bi,ij,bj->b', grad, env.Q, grad)
        nonlinear = quad + (1.0 - env.gamma) * (env.r + ham)

        rhs = u[n].ravel() + dt * nonlinear
        u_next = scipy.sparse.linalg.spsolve(A, rhs)
        u[n + 1] = u_next.reshape(n1, n2)

    grad_z1, grad_z2 = compute_gradient_grids(u, dz1, dz2, boundary=fdm_cfg.boundary)

    return FDMGridSolution(
        tau_grid=tau,
        z1_grid=z1,
        z2_grid=z2,
        u_grid=u,
        grad_z1_grid=grad_z1,
        grad_z2_grid=grad_z2,
        scheme=str(fdm_cfg.scheme),
        boundary=str(fdm_cfg.boundary),
        diagnostics=diagnostics,
    )
```

---

## 26. pseudo-code: `TrainedFDMPINN.grad_u`

```python
class TrainedFDMPINN:
    def grad_u(self, state_row, *, tau=None):
        tau_val = float(self.env.tau_max if tau is None else tau)
        z = self._state_array(state_row)

        tau_eval = np.clip(tau_val, self.solution.tau_grid[0], self.solution.tau_grid[-1])
        z1_eval = np.clip(z[0], self.solution.z1_grid[0], self.solution.z1_grid[-1])
        z2_eval = np.clip(z[1], self.solution.z2_grid[0], self.solution.z2_grid[-1])

        point = np.array([[tau_eval, z1_eval, z2_eval]])
        g1 = float(self.grad_z1_interp(point)[0])
        g2 = float(self.grad_z2_interp(point)[0])
        return np.array([g1, g2], dtype=float)
```

---

## 27. pseudo-code: `TrainedFDMPINN.policy_weights_with_debug`

```python
class TrainedFDMPINN:
    def policy_weights_with_debug(self, state_row, *, covariance=None, cross_mat=None, tau=None):
        cov = self.env.Sigma_train if covariance is None else _symmetrize_psd(np.asarray(covariance), floor=1e-10)
        cross = self.env.C_train if cross_mat is None else np.asarray(cross_mat, dtype=float)
        if cross.ndim == 1:
            cross = cross.reshape(-1, 1)

        x = self._state_array(state_row).reshape(1, -1)
        mu = self.env.mean_map.predict_batch(x).reshape(-1)
        grad_raw = self.grad_u(state_row, tau=tau)
        hedge_signal = cross @ grad_raw.reshape(-1)
        a_vec = mu + hedge_signal

        pi_unc = _solve_unconstrained_foc_np(cov, a_vec, gamma=self.env.gamma)
        w = _clip_foc_policy_np(pi_unc, cap=self.env.risky_cap, long_only=self.env.long_only)

        return w, {
            'hedge_signal': hedge_signal,
            'mu_term': mu,
            'hedge_term': hedge_signal,
            'a_vec': a_vec,
            'pi_unc': pi_unc,
            'pi_pre_clip': pi_unc,
            'risky_sum_before_clip': float(np.nansum(pi_unc)),
            'risky_sum_after_clip': float(np.nansum(w)),
            'risky_cap': float(self.env.risky_cap),
            'long_only_clip': bool(self.env.long_only),
            'neg_jxx': float(self.env.gamma),
            'neg_jxx_is_gamma': True,
            'control_update_space': 'foc_clip',
            'closed_form_costates': True,
            'grad_training': grad_raw,
            'grad_raw': grad_raw,
            'fdm_scheme': self.solution.scheme,
            'fdm_boundary': self.solution.boundary,
        }
```

---

## 28. 논문 또는 보고서에서의 설명 문장

FDM-PINN benchmark는 다음처럼 설명하면 정확하다.

> The FDM-PINN benchmark solves the same unconstrained log-HJB equation as the traditional PINN backend, but replaces the neural value approximation with a finite-difference grid solver over \((\tau,z_1,z_2)\). The time-to-maturity variable \(\tau\) is treated as the parabolic time-marching direction, while the PLS state variables \((z_1,z_2)\) form the two-dimensional spatial state grid. After solving the unconstrained HJB, the policy is extracted by the same unconstrained FOC formula used by the PINN baseline and is then mapped into the long-only risky-cap feasible set by clipping and rescaling. Therefore, FDM-PINN is a numerical benchmark for the unconstrained PINN PDE, not a constrained HJB solver.

한국어로는:

> FDM-PINN benchmark는 기존 PINN backend와 동일한 unconstrained log-HJB를 풀되, neural value approximation 대신 \((\tau,z_1,z_2)\) grid 위의 finite-difference solver를 사용한다. 여기서 \(\tau\)는 parabolic PDE의 time-marching 방향이고, PLS state \((z_1,z_2)\)가 2차원 spatial state grid를 이룬다. HJB를 푼 뒤 policy는 PINN baseline과 동일하게 unconstrained FOC로 산출하고, long-only 및 risky-cap constraint는 clipping/rescaling으로 사후 반영한다. 따라서 FDM-PINN은 constrained HJB solver가 아니라, unconstrained PINN PDE에 대한 numerical benchmark이다.

---

## 29. 최종 요약

현재 목표에 맞는 가장 깔끔한 설계는 다음이다.

1. backend 이름은 `fdm_pinn`으로 둔다.
2. FDM은 \(u(\tau,z_1,z_2)\)를 푼다.
3. wealth dimension은 CRRA homogeneity로 제거한다.
4. time \(\tau\)는 state가 아니라 marching axis다.
5. FDM solution은 3축 array이지만, 각 step의 spatial problem은 2D state grid다.
6. PDE는 현재 `pinn`과 같은 unconstrained Hamiltonian을 사용한다.
7. policy는 기존 `pinn`과 동일하게

\[
\pi_{\text{unc}}
=\frac1\gamma\Sigma^{-1}\left[\mu+C\nabla u\right]
\]

을 구한 뒤 clipping한다.

8. `fdm_pinn`에서는 `policy_output_mode='foc_clip'`을 강제한다.
9. `experiments.py`에서는 `pipinn`/`pinn`과 같은 value-gradient backend branch에 FDM-PINN을 포함한다.
10. solver는 SciPy sparse 기반 직접 구현이 가장 적합하다.
11. py-pde나 FiPy도 가능하지만, 현재 문제의 custom HJB와 backtest integration을 고려하면 직접 구현이 더 통제 가능하다.
12. FDM-PINN은 neural PINN의 PDE fitting 품질을 검증하는 benchmark로 쓰고, constrained HJB solver와는 구분해야 한다.

---

## 30. 참고 링크

- SciPy `spsolve`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html
- SciPy `diags`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags.html
- py-pde documentation: https://py-pde.readthedocs.io/
- FiPy documentation: https://pages.nist.gov/fipy/en/latest/index.html
- OSQP documentation: https://osqp.org/docs/
- CVXPY quadratic program example: https://www.cvxpy.org/version/1.2/examples/basic/quadratic_program.html

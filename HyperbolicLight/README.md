Anonymous GitHub for HyperbolicLight

data:
  Hangzhou:
      anon_4_4_hangzhou_real == hangzhou1
      anon_4_4_hangzhou_real_5816 == hangzhou2
  Jinan:
      anon_3_4_jinan_real == jinan1
      anon_3_4_jinan_real_2000 == jinan2
      anon_3_4_jinan_real_2500 == jinan3
  City256:
      anon_16_16_custom_sumo_2 == city256
  NewYork:
      anon_28_7_newyork_real_double == newyork1
      anon_28_7_newyork_real_triple == newyork2

results:
  all_results for paper results

run hyperbolic DQN for TSC code:
  python run_hyperboliclight.py

run summary ATT:
  python summary.py

## 1 Introduction
Official code for the article "A General Hyperbolic Reinforcement Learning Paradigm and Method for Traffic Signal Control".

The code structure is based on  [Advanced_XLight](https://github.com/AdvancedAI-ComplexSystem/SmartCity/tree/main/Advanced_XLight).

## 2 Requirements
`python3.8`, `tensorflow=2.4`, `torch=1.12.1`, `cityflow`, `pandas`, `numpy`

[`cityflow`](https://github.com/cityflow-project/CityFlow.git) needs a linux environment.

## 3 Usage

Our proposed method:
- For `hyperboliclight`, run:
```shell
python run_hyperboliclight.py
```

Baseline methods:
- For `advanced_colight`, run:
```shell
python run_advanced_colight.py
```

Summary:
```shell
python summary.py
```

## 4 Details
- Hyperbolic distance function: the hyperbolic distance function is the square of the distance between two points in hyperbolic space.
```python
def sqdist(self, p1, p2, c):
    sqrt_c = c ** 0.5
    dist_c = artanh(
        sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
    )
    dist = dist_c * 2 / sqrt_c
    return dist ** 2
```

- Projection operation: Project the point x into the interior of the Poincaré ball, ensuring its norm does not exceed the maximum value.
```python
def proj(self, x, c):
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
    maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)
```

- Exponential map: Map the vector u from the tangent space to a point in hyperbolic space.
```python
def expmap(self, u, p, c):
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
    second_term = (
            tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
            * u
            / (sqrt_c * u_norm)
    )
    gamma_1 = self.mobius_add(p, second_term, c)
    return gamma_1
```

- Möbius matrix-vector multiplication: Perform matrix-vector multiplication in hyperbolic space.
```python
def mobius_matvec(self, m, x, c):
    sqrt_c = c ** 0.5
    x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
    
    mx = x @ m.transpose(-1, -2)
    mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
    res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
    cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return res
```

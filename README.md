The related manuscript is currently under peer review. Any citation of the related work requires prior permission from the author. For inquiries, please feel free to contact the author by email.
liuhao232@mails.ucas.ac.cn
The main work is implemented in v16.6.py, while beam_new.py corresponds to the work on two-dimensional arrays.
Although many subfunctions define parameters such as the target direction, interference directions, SNR, and INR, the final parameter settings are determined by `main()`.
## Reproducibility Note for the Few-Snapshot Setting (`N=32`)

**Important:**  
For the **few-snapshot setting** (`N=32`), please make sure to run the experiments using the **default parameters reported in Table 5** of the paper.  
The figures/results in the paper are reproduced under these settings.

In particular, when `N=32`, please use the parameter configurations defined in:

- `main()`
- `main_quadrature_rule_ablation()` （For fig8 and fig 9）

---

## Default Parameters for Table 5 (`N=32`)

### 1. Main experiment: `main()`

Use the following configuration for the main comparison experiment:

```python
def main():
    out_dir = "./outputs_pub_same_nodes_gl_vs_cc"

    N = 32
    K = 64
    theta_s = 0.0
    theta_j = (-42.0, 56.0)
    snr_db = 10.0
    inr_db = 40.0
    main_half_width = 4.0

    M_each_shared = 12
    Q_each_shared = 17


## Default Parameters for Table 5 (`N=32`)

### 1. Main experiment: `main()`
def main_quadrature_rule_ablation():
    out_dir = "./outputs_pub_quad_rules"

    N = 32
    K = 64
    theta_s = 0.0
    theta_j = (-46.0, 32.0)
    snr_db = 10.0
    inr_db = 40.0
    main_half_width = 4.0

    M_each = 12
    Q_each = 17

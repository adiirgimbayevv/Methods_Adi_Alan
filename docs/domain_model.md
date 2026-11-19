# Domain model: AI feedback and programming outcomes

## Scope

The empirical unit is a student’s work in one lab. The public ProgFeed source has one row per graded function/test per submission; those rows are aggregated to submission-level scores and then to a student-lab trajectory. The comparative analysis is limited to labs 05, 06, 07, and 09, which used randomized feedback assignment.

The measured outcome is programming autograder performance. **Critical thinking is not measured in this dataset and is outside the empirical claims.**

## Feedback conditions

- $A_i=0$: no AI feedback
- $A_i=1$: generated failing test-case hint (tc)
- $A_i=2$: natural-language error explanation (nl)

$A_i$ denotes the assigned feedback arm. $D_i$ denotes whether any feedback text was actually delivered. Since hints are only produced following an error, $A_i$ and $D_i$ are not interchangeable.

## Outcome and estimand

For student-lab unit $i$, let $s_{ijt}$ and $m_{ijt}$ be the earned and maximum points on graded test/function row $j$ in submission $t$. The submission score is

$$
Y_{it}=100\frac{\sum_j s_{ijt}}{\sum_j m_{ijt}}.
$$

Let $t=1$ be the first observed submission and $T_i$ the last observed submission. The descriptive score change for initially failing units is

$$
G_i = Y_{iT_i}-Y_{i1}, \qquad Y_{i1}<100.
$$

The report also gives the retry rate and the proportion that reaches full credit on any observed submission. The primary comparisons are arm-specific means of $G_i$, with a 95% cluster bootstrap interval that resamples students, keeping their lab records together.

A simple adjusted mean model for the randomized-lab subset is

$$
G_i = \beta_0 + \beta_{tc} I(A_i=1)
      + \beta_{nl} I(A_i=2) + \sum_{\ell}\gamma_\ell I(L_i=\ell)
      + \varepsilon_i,
$$

where $L_i$ is the lab. This is a descriptive assignment-arm contrast. Inference must account for repeated student observations and conditional feedback delivery; it does not identify an effect on critical thinking or long-term learning.

For a least-squares fit, write the model matrix row as $\mathbf{x}_i$ and its coefficient vector as $\boldsymbol{\theta}$. The fitted parameters minimize

$$
\mathcal{L}(\boldsymbol{\theta})=\frac{1}{n}\sum_{i=1}^{n}\left(G_i-\mathbf{x}_i^{\mathsf T}\boldsymbol{\theta}\right)^2.
$$

The target summaries are the mean score change $G_i$, retry rate $P(A_i\text{ has a later submission}\mid Y_{i1}<100)$, and eventual full-credit rate $P(\max_t Y_{it}=100\mid Y_{i1}<100)$. Uncertainty for score change is estimated with a student-cluster bootstrap.

## Task 3 pipeline metrics

The separate local ETL benchmark measures latency $L$ in milliseconds and throughput $R$ in rows/second while aggregating real ProgFeed rows. It compares a Python row loop with a pandas groupby at fixed input sizes. These measurements describe this local processing task and hardware; they are not API latency or production-system performance.

For latency observations $L_1,\ldots,L_n$, the report uses the median, $p_{95}$, $p_{99}$, IQR, and a seeded bootstrap confidence interval. Outliers are flagged with the $1.5\times\mathrm{IQR}$ rule and retained.

## Notation

| Symbol | Meaning |
|:--|:--|
| $i$ | Student-lab unit |
| $j$ | Graded test/function row |
| $t$ | Submission order within a student-lab unit |
| $A_i$ | Assigned feedback arm: none, test-case hint, or natural-language hint |
| $D_i$ | Indicator that feedback text was actually delivered |
| $L_i$ | Randomized intervention lab |
| $s_{ijt}$ | Earned points on a graded row |
| $m_{ijt}$ | Maximum points on a graded row |
| $Y_{it}$ | Submission score as percent of available points |
| $G_i$ | Change from first to final score for initially failing units |
| $T_i$ | Last observed submission index |
| $L_k$ | Latency of benchmark run $k$, milliseconds |
| $R_k$ | Throughput of benchmark run $k$, rows/second |
| $n$ | Number of observations in a summarized condition |
| $B$ | Number of bootstrap resamples |
| $s$ | Random seed, fixed to 42 |

---
csl: ''
bibliography: ../thesis.bib
---

Background
==========

Risk Minimisation
-----------------

A classifier $$h \in \mathcal{H}$$ is a mapping $$h: \mathbb{X} \mapsto
\mathbb{C}$$ that determines which class $$c \in \mathbb{C}$$ a sample $$x \in
\mathbb{X}$$ belongs to. For the purposes of this work, we only describe binary
classification in the following pages, however, the task is, naturally,
expandable to a general discrete set $$\mathbb{C}$$. In the classical risk
theory, the classifier $$h$$ is a subject to minimisation of expected risk
$$R(h)$$ given a cost function $$\ell: \mathbb{C} \times \mathbb{C} \mapsto
\mathbb{R}$$.

$$
R(h) = \mathbb{E}_{(x,c) \sim p} \left \ell(h(x), c) \right
$$

Formally, the Expected Risk Minimisation (ERM) is given by:

$$
\min_{h \in \mathcal{H}} R(h)
$$

Typically when working with binary classification, $$\ell$$ is consider a *1-0
loss* which assigns an equal cost of magnitude *1* for misclassifying objects.
The expected risk in this case accounts only for the rate of false positives and
false negatives. If we employ *1-0 loss* into the expected risk, we arrive at
the following form:

$$
R(h) = \sum_{c \in \mathbb{C}} p(c) \int_{x: h(x) \neq \mathsf{c}} p(x|c) \,  dx
$$

The integral can be considered a probability of classifying objects $$x$$ to an
incorrect class given a correct class $$c$$, ie. $$h(x) \neq c$$. Let us
consider binary classification in which $$\mathbb{C} = \{ \mathsf{B}, \mathsf{M}
\}$$ where $$\mathsf{M}$$ stands for a positive class (m for a malicious class)
and $$\mathsf{B}$$ for a negative class (b for a benign class). In the context
of this work, the positive class refers to malicious activity, ie. activity that
is desired to be uncovered, and the negative class covers benign, legitimate or
normal behaviours. To conclude, the risk $$R(h)$$ can be rewritten as a mixture
of two types of errors: the probability of false positives and the probability
of false negatives.

$$
R(h) = p(\mathsf{B}) \cdot p(\mathsf{M}|\mathsf{B}) + p(\mathsf{M}) \cdot p(\mathsf{B}|\mathsf{M})
$$

In practice, the probabilities are not known and, moreover, computing the
expected risk often involves intractable integrals. Therefore, the risk is
empirically estimated from observed samples. The empirical risk $$\hat{R}(h)$$
estimated from a set of training samples $$T_m = \{ (x_i, c_i) \}_{i=1}^{m}$$ is
defined as follows:

$$
\hat{R}_{T_m}(h) = \frac{1}{m} \sum_{(x_i, c_i) \in T_m} L(h(x_i), c_i)
$$

@vapnik showed that with increasing $$m$$ the empirical risk
$$\hat{R}_{T_m}(h)$$ approaches $$R(h)$$.

Regularisation
--------------

When examining possible classifiers, we usually have a priori knowledge of
certain classifier instances being more suitable than others. Hence, some
classifiers $$h$$ correspond to models that are more likely to be inadequate,
and some are a priori preferred. The reasons may vary, but mostly one desires to
decrease models complexity to avoid over-fitting. To capture this knowledge, a
regularisation term $$\Omega_D: \mathcal{H} \mapsto \mathbb{R}$$ penalising some
classifiers $$h$$ is often added to the risk.

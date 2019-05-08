Problem Definition
==================

In the present state of Internet, it is common for a site owner to run models
classifying users or their behaviour. The task spans from user’s interests
specification to detecting deviating activity. Since such applications are
becoming more popular, one may expect the users to modify their behaviour once
they know they are being tracked and classified. Moreover, behaviour
modification may very well be of rational nature, especially when a malicious
user exploits loopholes or carries out lawless activity in order to pursuit its
goal.

In other words, if there is a cost for being disclosed or seen as a certain
category, the users will examine their actions to optimise for lower cost. As a
result, machine learning models of any kind aiming to capture behaviours of
those users necessarily need to have the adversary nature incorporated in their
design.

The straight-forward approach of solving this task would be to collect many
examples of both kinds of user activity; that is to asses a dataset containing
well-represented both malicious and benign users. This approach would follow the
standard ERM framework and would give an activity classifier that minimises
expected risk but omits the adversarial nature. However, one might arrive at
difficulties during the construction of a balanced dataset for there is usually
very few records of malicious activity, disproportionally less than the
collection of normal, benign users. Also, and more importantly, the malicious
actors modify their obfuscate vectors once their method is exposed or they
discover details concerning the detector.

Taking that into account, we consider the setting a game of a classifier
competing with a body of malicious users. This approach necessarily modifies the
ERM framework and enhances it with game-theoretic notions.

This section first discusses the use-case which motivates this work. Then, a
suitable threat model is prosed and the game is formally defined, supplemented
with reasoning for given choices.

Motivation
----------

In this work, we consider a computer security company that runs an API service
which returns rating of a queried URL. This type of service is usually deployed
by such companies to provide their security software with access to most
up-to-date database of URL ratings.

The typicall usage scenario is coined as follows. A client running on an
end-user’s device encounters the user is about to enter a website. To evaluate
the danger level of the website, the client queries the API service with the
website URL. Accordingly, the client may show a warning message notifying the
user of expected danger or carry out an appropriate action.

Usually, URL rating systems aim to identify various danger types of a URL. In
this work, we focus on malware producers that asses a set of URLs that serve as
a communication entry-points for deployed malware units. With one of these URLs,
a unit of deployed malware is able to receive commands and adjust its actions.
However, to maintain consistency and availability of its malware units, the
malware producer needs to regularly check whether any of its URLs has been
exposed – by querying the publicly available URL rating system.

The task now is as follows: the computer security company desires to distinguish
malicious users of the URL rating service from benign ones.

In principle, the task is an instance a broader family of problems. Agents query
a service in order to discern information about a set of critical objects and we
are given a specificity function which evaluates those objects. In other words,
the specificity function $$f$$ heuristically assigns a high value number to
salient objects that we are interested in and a near-to-zero number to unrelated
objects. Consequently, we use the $$f$$-values to specify what queries (or the
queried objects, respectively) are relevant in the given instance of the user
activity classification problem.

Therefore, the communication between an agent and a service, consisting of the
agent querying the service with the objects, can be seen as a sequence of
$$f$$-values of those objects. Based on $$f$$, we design a classifier that sorts
agents into two groups; one given by agents carrying out communication which is
somewhat prone to high values of $$f$$, and the other grouping agents with low
$$f$$-values communication.

This work's use-case takes the URL rating as the specificity function $$f$$, and
each URL is considered an object which the service is queried with. Thus, we aim
to classify the agents which tend to query high-danger URLs to one group and the
other agents to a the other–benign–group.

Game Model
----------

Formally, the service is queried with a query $$q \in \mathbb{Q}$$ where
$$\mathbb{Q}$$ is the set of all queries. The service securely assigns each
query to a user. Thus, we define a query profile $$\pi \in \Pi$$ built using the
queries of the user, possibly supplied with additional query properties. For
example, if a user sends a sequence of queries to the service supplemented with
a timestamp, a source IP or possibly other information, this is recorded and
integrally stored in a corresponding user’s query profile $$\pi$$.

$$
(q_1, t_1, \mathsf{IP}_1, \dots), \, (q_2, t_2, \mathsf{IP}_2, \dots), \, \dots, \, (q_k, t_k, \mathsf{IP}_k, \dots) \longrightarrow \pi
$$

For practical reasons, we define a partially inverse mapping $$\sigma(\pi)
\subseteq \mathbb{Q}$$ which denotes the queries comprising the profile $$\pi$$.
Using the example above $$\sigma(\pi) = \{ q_1, q_2, \dots, q_k \}$$.

A single malicious user posses a private set of critical queries $$\cr{Q}
\subset \mathbb{Q}$$ that contains queries which the malicious user will
necessarily employ to achieve its goal. In other words, there is a piece of
information which is sought by the malicious user and which is obtainable only
by querying the service with critical queries $$\cr{Q}$$. Given its critical
queries, a malicious user sends requests to the service with queries $$Q
\subseteq \mathbb{Q}$$ that may next to its critical queries also contain
legitimate queries which it uses to cover-up its activity.

$$
\cr{Q} \subseteq Q 
$$

If there was no classifier and malicious users were not motivated to adjust
their behaviour, they would simply query the service with $$Q$$ resembling
critical queries, presumably containing just a little overhead, ie. $$Q \cong
\cr{Q}$$. A user query activity comprising only queries in $$\cr{Q}$$ creates a
strictly critical query profile $$\cr{\pi}$$ such that $$\sigma(\cr{\pi}) =
\cr{Q}$$. However, taking into account the overhead queries in $$Q$$, the
probability of a query profile of a malicious user $$p(\pi | M)$$ is given by
the underlying distribution of critical queries $$\cr{Q}$$ and overhead queries.
Assuming there is only little overhead, $$p(\pi | M)$$ becomes the probability
of strictly critical query profiles $$p(\cr{\pi})$$.

$$
p(\pi | M) \cong p(\cr{\pi})
$$

Nonetheless, once there actually is a classifier deployed, implying a cost for
disclosure of a malicious user, the malicious users rationally query the service
with additional legitimate queries to obfuscate the classifier. Since we allow
only *adding* (legitimate) queries to a query profile, each strictly critical
query profile induces a bounded set of profiles $$S(\cr{\pi})$$ that contains
profiles derivable from $$\cr{\pi}$$ by adding queries.

$$
S(\cr{\pi}) = \{ \pi \in \Pi \,\mid\, \sigma(\cr{\pi}) \subseteq \sigma(\pi) \}
$$

This is thoroughly captured by an obfuscation function $$g: \cr{\Pi} \mapsto
\Pi$$ which a malicious user employs to transform its original strictly critical
query profile $$\cr{\pi}$$ to an obfuscated query profile $$g( \cr{\pi} ) \in S(
\cr{\pi} )$$. In this case, the presence of the classifier changes the
probability distributions. Concretely, the distribution of malicious query
profiles is now governed by the distribution of obfuscated strictly critical
query profiles.

$$
\dot{p}(\pi | M) = p( g( \cr{\pi} ) )
$$

To conclude, the activity classification problem is modelled as a game of two
players: a detector and an adversary. The goal of the detector is to identify
the best user activity classifier, while the adversary seeks to optimally modify
query profiles of malicious users in such a way they get misclassified by the
classifier. The detector is a $$-1$$ player and the adversary is a $$+1$$
player.

### Detector

Mathematically, the task of the detector is to find a mapping $$h \in
\mathcal{H}$$ which classifies a query profile’s feature vector $$x \in
\mathcal{X}$$ to a class $$\mathbb{C} = \{ \mathsf{B}, \mathsf{M} \}$$, ie. $$h:
\mathcal{X} \mapsto \mathbb{C}$$. Note that $$\mathsf{B}$$ stands for benign
users, while $$\mathsf{M}$$ denotes malicious users.

A feature vector representing a query profile is given by a feature map $$\Phi :
\Pi \mapsto \mathcal{X}$$ which takes a query profile $$\pi$$ as an argument and
maps it to a real vector $$x \in \mathcal{X} \subseteq \mathbb{R}^d$$.
Naturally, $$\Phi$$ is surjective and is given a priori to task solving.

Following the ERM framework, the optimal classifier $$h^*$$ is given by
minimising its expected risk:

$$
R_{-1}(h) = \E_{(\pi,c) \sim \dot{p}} \left \ell_{-1}(h \circ \Phi(\pi), \, c) \right 
$$

where $$\dot{p}(\pi,c)$$ represents the joint probability of a query profile
$$\pi \in \Pi$$ and a class $$c \in \mathbb{C}$$, partly modified by the
adversary as explained above. $$\ell_{-1}$$ stands for a classification loss
function.

Generally, we prefer some classifier instances to others, therefore, we employ a
regularisation term $$\Omega_{-1}(h)$$. In conclusion, the optimal classifier
$$h^*$$ is given by the following equation.

$$
h^* = \min_{h \in \mathcal{H}} R_{-1}(h) + \Omega_{-1}(h)
$$

### Adversary

Since the objectives of all malicious users are equivalent, ie. they aim to
obfuscate their private set of critical queries $$\cr{Q}$$, the final query
profile of each of them is strictly a function of $$\cr{Q}$$. Due to the shared
goal, we represent the malicious users as a single-body aggregate player, the
adversary.

The adversary aims to identify an obfuscation function $$g: \cr{\Pi} \mapsto
\Pi$$. This is done for a fixed $$h$$ and $$\Phi$$ by minimising the risk of the
adversary $$R_{+1}(g)$$. However, following the threat model we restrict the
adversary to produce only malicious samples (in contrast to the general form of
the Stackelberg Prediction Game by @stackelberg_games) and these are inherently
given by the distribution of strictly critical query profiles $$p(\cr{\pi})$$.
This simplifies the adversary’s risk $$R_{+1}(g)$$ to a new form:

$$
R_{+1}(g) = \E_{ \cr{\pi} \sim p_{\cr{\pi}} } \left \ell_{+1}(h \circ \Phi \circ g (\cr{\pi}), \, \mathsf{M} ) \right 
$$

where $$p_{\cr{\pi}}$$ gives the probability distribution from which
$$\cr{\pi}$$ are drawn. $$\left \ell_{+1}: \mathbb{C} \times \mathbb{C} \mapsto
\mathbb{R}$$ is the adversary’s cost function.

Similarly to $$h$$-regularisation, we prefer some obfuscation functions to
others which is articulated by a regulariser $$\Omega_{+1}(g)$$. In conclusion,
the optimal obfuscation function $$g^*$$ is given by the following equation:

$$
g^* = \min_{g \in \mathcal{G}} R_{+1}(g) + \Omega_{+1}(g)
$$

Zero-one Loss Assumption
------------------------

Let us now assume that both $$\ell_{-1}$$ and $$\ell_{+1}$$ are *1-0 losses*
that penalise misclassification, or detection, respectively. Then the
adversary’s risk simplifies to:

$$
R_{+1}(g) = \sum_{ \substack{ 
\cr{\pi} \in \cr{\Pi}
\n 
h \circ \Phi \circ g(\cr{\pi}) = \mathsf{M} 
}} p(\cr{\pi})
$$

Note that this means the optimal $$g^*(\cr{\pi})$$ gives an obfuscated query
profile $$\pi \in S(\cr{\pi})$$ that is classified as $$\mathsf{B}$$ and
minimises a regulariser $$\Theta_{+1}(\pi, \cr{\pi})$$ which is induced by
$$\Omega_{+1}(g)$$. The adversary’s minimisation problem can be rewritten to:

$$
g^*(\cr{\pi}) \in \min_{\pi \in S(\cr{\pi})} \Theta_{+1}(\pi, \cr{\pi}) \qquad \textsf{s.t.} \quad h \circ \Phi (\pi) = \mathsf{B}
$$

In cases in which there is no obfuscated profile that the classifier labels
benign, $$g^*(\cr{\pi})$$ simply returns $$\cr{\pi}$$ or any element of
$$S(\cr{\pi})}$$ depending on what type of Stackelberg equilibrium is played.

Additionally, if $$\Theta_{+1}(\pi, \cr{\pi})$$ is convex in $$\pi$$, there is a
single optimal value $$g^*(\cr{\pi})$$. However, note that this for example does
not hold true for $$\Theta_{+1}(\pi, \cr{\pi}) = |\sigma(\pi)|$$ as there might
exist a tuple $$\pi_1$$ and $$\pi_2$$, s.t. $$|\sigma(\pi_1)| =
|\sigma(\pi_2)|$$ but $$\sigma(\pi_1) \neq \sigma(\pi_2)$$.

The *1-0 loss* also simplifies the detectors risk. It is convenient to decompose
the risk into two components: the first stands for benign users expectation and
the second for malicious users expectation.

$$
R_{-1}(h) = \sum_{ 
    \substack{ 
        \pi \in \Pi 
        \n
        h \circ \Phi (\pi) = \mathsf{M} 
    } 
} p(\pi, \mathsf{B})
+  
\sum_{
    \substack {
        \cr{\pi} \in \cr{\Pi}
        \n
        h \circ \Phi \circ g^* (\cr{\pi}) = \mathsf{B} 
    }
} p( g^*(\cr{\pi}) , \mathsf{M}) 
$$

Additional remarks
------------------

What is to be done in the game specification:

-   finalise the derivation of the optimal $$h^*(x)$$

-   introduce a naive approach based on the assumption the URL ratings, $$f$$,
    are independent, ie. $$p(\mathsf{B} | \pi) = \prod_{q \in \sigma(\pi)}
    f(q)$$

-   show how this translates to empirical risks and actual classifier
    optimisation criterion

 

 

The threat model is extendable by the following notions:

-   a time window $$\tau$$ is used to collect query profiles; the detection
    repeats at the end of each collection window

-   a malicious user buys a new license once it is detected and continues
    querying the service with the remaining queries of $$\cr{Q}$$

-   a malicious user is allowed to create strictly benign query profiles in
    order to poison future retraining datasets (most likely, will not be
    eventually explored)

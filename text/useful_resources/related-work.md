Related Work
============

Examining adversarial aspects of various machine learning problems has currently
been a popular topic. Mainly, this was triggered by @adversarial_examples who
showed that neural networks are susceptible to adversarial examples. Since then
many endeavours have been carried out to enhance neural networks by making them
robust. Some tried to develop a provably robust classifier [@provable_defenses],
while others reframed the classification problem to incorporate aspects of game
theory [@stackelberg_games]. Despite most of the work deals with image
classification, efforts to utilise the same notions in computer security have
been seen too [@adversarial_malware_pe]. Susceptibility to adversarial examples
is, however, not the only weakness adversaries exploit, they also are able to
modify future training datasets in their favour [@antidote].

Adversarial setting
-------------------

@good_word_attacks explore obfuscate strategies yielding spams that circumvent a
spam filter. The authors consider attacks which are based on adding words to a
spammy e-mail, while other modifications are not allowed. Three pools of words
are defined: in the first obfuscate, random words from a dictionary are drawn;
the second obfuscate utilises common legitimate e-mail words; and in the third
obfuscate, words that are likely to appear in legitimate e-mails but are
uncommon in spams are added.

To select the final set of words with the greatest effect from one of the three
word pools, a black box threat model is used. In particular, the attacker
repeatedly calls the detector to identify words which make the detector label
the spam as benign. As expected, the last pool of words mentioned outperforms
the others. Moreover, this shows that additive changes to a malicious object are
sufficient for obfuscating the detector (within this domain). The authors claim
they are able to add words to spams in such a way the tested detection models do
not detect 50% of them.

To reflect the successful obfuscate algorithm, a defense strategy is proposed.
It is shown that a robust detector which uncovers the adjusted spammy e-mails
can be obtained by simply retraining the model on data now containing the
attacks. However, the authors comment, a repeated obfuscate with a new set of
effective words may again defeat the detector.

A similar notion is seen in more advanced classification models. For instance,
deep neural networks are a popular class of classifiers nowadays for their
performance in a great range of fields. They were shown to outperform other
methods in image classification [ImageNet Challenge; @image_net], natural
language processing [@transformer] and in many others fields. However, it was
found that neural networks are susceptible to artificially crafted images. In
particular, @adversarial_examples show an adversarial example may be labeled as
an arbitrary class when accordingly adjusted. Moreover, despite the
transformation of an input image is substantially bounded, for example by
$$l_\infty$$ norm, classifiers based on neural networks are prone to be
circumvented anyway [@adversarial_examples_2]. The susceptibility to adversarial
samples follows the same observation in spam filtering – a good classifier is
not necessarily robust to test time data manipulation.

As soon as it was recognised the neural networks contain built-in
vulnerabilities which are exploitable, endeavours to improve the architecture
were carried out. To address the weakness, some of the following work focus on a
model definition and consider possible attacks already in the model design. This
approach is summarised by @towards_deep_learning_models who study adversarial
examples in image classification. The authors identify that Empirical Risk
Minimisation (ERM) does not necessarily give models robust to adversarially
crafted samples.

Their work extends the training framework based on ERM by a threat model in
which each data point $$x \in \mathbb{R}^d$$ is assigned a set of perturbations
$$S(x) \subseteq \mathbb{R}^d$$ that is available to the adversary. The authors
work with $$S_\epsilon(x)$$ that contains perturbations bounded by $$l_\infty$$,
creating an $$\epsilon$$-hyper-cube around each $$x$$:

$$
S_\epsilon(x) = \{x' \in \mathbb{R}^d \, \mid \, l_\infty(x - x') \, \leq \, \epsilon \}
$$

The norm $$l_\infty$$ is used for simplicity and roughly represents
human-undetectable image perturbations. Other approaches, however, consider more
complex bounds that capture domain-specific constraints
[@adversarial_examples_glasses].

To fully relate to an adversarial setting, @towards_deep_learning_models propose
that the adversary maximises the classifier's loss function $$L$$ by modifying
an image $$x$$ to an adversarial example $$x' \in S_\epsilon(x)$$. This is
further incorporated into the ERM framework, arriving at a saddle point problem:

$$
\min_{h \in \mathcal{H}} \, \mathbb{E}_{(x, c) \sim p} \Big[ \max_{x' \in S_\epsilon (x)} \,  L(h(x'), \, c) \Big] 
$$

In other words, a solution to the problem gives an optimal robust classifier $$h
\in \mathcal{H}$$ that is likely to classify all objects $$x \in \mathbb{R}^d$$
and their neighbourhood $$S_\epsilon (x)$$ correctly.

The saddle point problem given above consists of two sub-problems: training the
neural network and performing the inner maximisation.
[@towards_deep_learning_models] approach the training part with Stochastic
Gradient Descent (SGD) as it is commonly done in neural networks, while solving
the inner maximisation task with Projected Gradient Descent (PGD) [@pgd]. They
conclude the ERM framework extended by this specific threat model gives a
training method that is able to train neural networks in the adversarial setting
and to produce classifiers robust to $$l_\infty$$ bounded image perturbations.
In addition, they find lower error is obtained with higher capacity models,
suggesting that a robust model requires more parameters (eg. layers in neural
networks).

To address the susceptibility to adversaries, several proposals of neural
networks enhancements were submitted at ICLR 2018. However, seven out of nine
were shown to be flawed due to following a similar ineffective scheme of masking
the gradients [@obfuscated_gradients].

In their paper, @obfuscated_gradients suggest there are three groups of gradient
masking: first, a non-differentiable layer is inserted between the network
layers; second, a classifier randomises its outputs; and third, a function
transforms the input in such a way backward gradient explodes or vanishes.
Showing that the submitted defensive methods follow the schemes, the authors
succeeded in circumventing 7 of 9 proposed models. Concretely, they replaced or
removed defensive non-differentiable components accordingly to estimate the
gradient and crafted adversarial samples with PGD.

Provable robustness
-------------------

Until now, all presented efforts to improve the neural networks susceptibility
were approached empirically and usually without providing provable defenses
[@towards_deep_learning_models]. A method that aims to give provable resistance
to adversarial samples was proposed by @provable_defenses who examine a novel
network architecture that provably classifies all objects in a convex
neighbourhood of a given image correctly. To achieve that, @provable_defenses
redefine a ReLU [@relu] in such a way it is not a function anymore but rather a
set of linear constrains yielding a convex polytope; i.e. a ReLU $$y = \max \{0,
x\}$$ becomes:

$$
y \geq x,
$$

$$
y \geq 0,
$$

$$
y(l-u) \leq -ux + ul 
$$

where $$u$$ and $$l$$ are an upper, respectively lower bound of $$x$$. The
bounds are unknown and need to be estimated for each ReLU.

With a convex relaxation of ReLU, image classification can be rewritten as a
linear program with all components of the network now being linear. In the
training process, the weights of the relaxed neural network are optimised so
that the network correctly classifies not only the input image but also its
convex embedding. More specifically, using a $$l_\infty$$ norm a
$$\epsilon$$-neighbourhood of an input sample is embedded by a convex polytope
and the network learns to disallow any adversarial samples in it.

Solving the optimisation problem in its LP form with a standard LP solver is not
tractable due to a great number of variables needed to express state-of-the-art
deep neural networks. However, the LP can be conveniently used to form an upper
bound on robust classification accuracy. Now, this upper bound combined with the
ReLU input bounds estimation becomes fully differentiable. The training process
follows standard SGD and gives a robust classifier that allows provably at most
6% error on MNIST. In contrast, a classical neural architecture is vulnerable up
to 80% error [@provable_defenses].

Optimising malware
------------------

In contrast to image classification, the space of inputs is usually discrete in
computer security. An image can be represented as a vector in $$[0, 1]^n$$,
while executable binaries span a very sparse subset of the binary space $$\{0,
1\}^n$$. Similarly, a set of executable source codes in a given programming
language is a sparse subset of all character strings. Despite the theoretical
difficulties several papers address the issue. @adversarial_malware propose an
obfuscate that optimises a malicious source code by applying some of the
predefined modifications. The obfuscate method utilises the classifier’s
gradient to choose the most appropriate code modification. The set of plausible
modifications is given beforehand and allows only additive changes. Although
this significantly limits the attacker’s action space, the authors claim
reaching misclassification rates of up to 69%. @adversarial_malware_binary focus
on static portable executables which they encode into a binary feature indicator
vectors. Again, additive modifications are allowed only and malware is optimised
with a bit gradient ascent. @adversarial_malware_pe take a different approach to
malware optimisation and propose an agent which is trained with reinforcement
learning. The agent is given a portable executable and its goal is to choose the
most suitable modification of a piece of malware to lower probability of
detection.

Stackelberg Prediction Game
---------------------------

As already shown, the problem of adversarial samples can be modelled as a game
of two actors. However, @stackelberg_games propose a more general game model
compared to those already mentioned. In particular, the authors define the
players as a classifier and a data generator consisting of *all* actors
generating data – that is the second player aggregately covers both benign and
malicious actors.

This setting is explored using a game-theoretical point of view. The authors
propose a Stackelberg Prediction Game in which a classifier, acting as a leader,
and a data generator, acting as a follower, optimise their action to meet their
objectives. They argue the Stackelberg equilibrium is the most appropriate
concept for trainable models, specifically compared to the Nash equilibrium. It
is so, they claim, mainly because once a model is finalised and deployed, it is
not changed anymore and thus the attacker can potentially learn all details of
the model and adjust its actions to it.

In other words, the actions – the choice of model parameters and the test time
data generation – are not carried out simultaneously, but instead the classifier
commits to a specific parameters vector and the attacker utilises the
information about the model and adjusts its attacking strategy accordingly. The
later is modelled by a distribution shift at test time. The data generator
transforms a probability of data $$p$$ to a test time data probability
$$\dot{p}$$ which maximises its objective function. In addition, the authors
show that linear and kernel-based models together with suitable objective
functions allow reformulating the problem to a quadratic program which yields
the optimal model parameters.

Dataset poisoning
-----------------

The Stackelberg Prediction Game assumes the model is fixed after deployment. In
practice, however, engineers retrain the model on newly obtained data that might
better represent their population. As this might be done periodically, the
adversary shall take advantage of it and adjust its obfuscate strategy.
Concretely, @antidote elaborate on poisoning anomaly detectors.

The poisoning obfuscate consists of purposely providing pre-crafted samples to
the detector over a long period of time in belief, that the samples will create
a blind spot in which all samples are considered benign by the detector. The
authors assume that the input space is usually governed by a distribution of
benign samples concentrated only in certain areas, leaving the rest for
anomalous activity. Given a substantial amount of time, the adversary is
gradually able to poison the detector by targeting the large empty parts of the
input space and populating them with benign samples. In future retraining, the
anomaly detector may mistakenly consider those re-populated areas a new
phenomenon and label them benign. The attacker then simply crafts an obfuscate
near to the poisoned areas of the input space.

The authors present that such an obfuscate is possible with an anomaly detector
based on principal component analysis (PCA) which determines directions of the
sample space with greatest variance. Replacing variance in PCA with median
absolute deviation, which in contrary is a robust scale estimator, their model
is robust to data set poisoning and successfully performs anomaly detection in
backbone networks.

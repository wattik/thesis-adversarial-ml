Introduction
============

In recent years, computer science has seen an advent of powerful algorithms that
are able to learn from examples. Despite the notion of learnable algorithms was
recognised and studied in pioneering times of the field already, its wide-range
real-world applications were to be implemented only with the presence of big
available data collections and vast memory and computational resources.
Therefore, nowadays one meets the abundance of machine learning techniques used
to solve various problems. The field spans from theoretical research to
practical applications in areas such as medical diagnosis, financial predictions
and, most importantly in case of this work, computer security.

Most of the applications coin a similar scenario: a problem is formalised
following a standard machine learning paradigm; a vast data set is collected and
a proper algorithm giving the best results is found forming a model of the
problem. However, in some applications once such a model is deployed to a
complex real-world environment, one soon identifies the model performance
deteriorates due to the key aspects of the reality that have been omitted in the
standard machine learning point of view.

An example of such an observation is seen in computer vision. It was found that
deep neural networks that reign competitions in image classification
[@image_net] are prone to so called adversarial images [@adversarial_examples].
In particular, the state-of-the-art image classifiers based on deep neural
networks score very well in terms of prediction accuracy, when given genuine
images. However, such a classifier can be fooled with an image that was
purposely adjusted. To put it simply, what is seen as a unambiguous cat by a
human observer can be confidently labelled as a dog by a classifier. For
instance, this phenomenon challenges traffic sign classification used in
autonomous vehicles because it has been shown that a few well-placed stickers
are able to fool the classifier and make it mis-recognise a yield sign for a
main road sign [@adversarial_examples_2].

To reflect such weakness, problems are reframed to a game-theoretic setting in
which two autonomous rational players compete while following their mutually
conflicting objectives. The aforementioned example with images is, consequently,
extended in the following way. One of the players acts as an image classifier
and aims to maximise classification accuracy, whereas the other player, an
adversary, perturbs the images to lower prediction confidence or, better, to
make the classifier misclassify the image.

Of course, the same is seen in computer security–the field defined by
adversarial nature. Intruders desire to obfuscate a detector by adjusting their
attacks [@adversarial_malware]; malware is developed by optimising an executable
binary [@adversarial_malware_pe], and spams are improved statistically to avoid
detection [@good_word_attacks].

The task central to this work is the problem of user classification in the
adversarial setting. First, we examine the task as an instance of the user
classification problem viewed by standard machine learning paradigm.
Consequently, we show that omitting the adversarial nature in this particular
problem exposes critical weaknesses, thus we extend the model to incorporate
game-theoretic notions. As an instance of the user classification task, we
consider a detector of malicious users that is deployed by a computer security
company to see which users exploit their public API service.

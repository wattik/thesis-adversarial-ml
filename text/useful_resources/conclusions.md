Conclusions
===========

This work is a draft of a diploma thesis. It is an evaluated outcome of a
semestral project that precedes the thesis. In the draft, a motivation to the
problem and its definition is proposed. Concretely, we deal with a user
behaviour classification problem that incorporates adversarial nature of some of
the actors. The proposed threat model introduces a set of critical objects (here
URLs) that a malicious user necessarily employs in communication with a service.
This is a fundamental building block which imposes necessary modifications of
the existing threat models met commonly in literature.

The draft explores related work and gives a formal definition of the problem,
specifying a threat model that is inspired by the Stackelberg Prediction Game by
@stackelberg_games. However, some modifications to SPG are proposed. Last but
not least, the game definition is augmented by each players’ actions analysis,
arriving at the conclusion the ERM framework is an excellent starting point for
solving adversarial machine learning problems and gives, when combined with game
theory, mathematical programs that are related to robust optimisation.

During thesis preparation, the draft will be enriched with the remaining parts
of players’ actions analysis. This will, *hopefully*, give a mathematical
program for which an algorithm will be proposed. It goes without saying that the
algorithm will then be compared to baseline approaches on real-world data and
the experiment results will be evaluated. For now, it seems the HTTP data from
the university's DNS logs will be used. Yet, this is a subject to change.

Finally, let us naïvely believe the proposed models and experimental findings
will serve the common good, rather then the truly malicious actors.

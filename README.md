# Dorylus

This is Dorylus, a *Scalable*, *Resource-efficient* & *Affordable*
computation system for
[Graph Neural Networks](https://tkipf.github.io/graph-convolutional-networks/),
built upon an architecture combining cheap data servers on
[AWS EC2](https://aws.amazon.com/ec2/) with serverless computing on
[AWS Lambda Threads](https://aws.amazon.com/lambda/).

> Dataserver originally is a push-based ASPIRE implementation, a cleaned up version of gift (forked on July 06, 2016). Implemented streaming-like processing as in Tornado (SIGMOD'16) paper.

Now the main logic of the engine has been completely simplified, and we integrate it with AWS Lambda threads. Ultimate goal is to achieve "*Affordable AI*" with the benefit of **cheap scalability** brought by serverless computing.



## User Guide

Check our [Wiki page](https://bitbucket.org/jothor/dorylus/wiki/Home) for managing your EC2 clusters, building & running Dorylus.

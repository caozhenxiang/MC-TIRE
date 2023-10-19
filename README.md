MC-TIRE
===============================

Repository for developing the multi-channel version of the time-invariant representation autoencoder approach (TIRE) for change point detection (CPD) task. More information can be found in the paper **Change Point Detection in Multi-channel Time Series via a Time-invariant Representation**.


The authors of this paper are:

- [Zhenxiang Cao](https://www.esat.kuleuven.be/stadius/person.php?id=2380) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)
- [Nick Seeuws](https://www.esat.kuleuven.be/stadius/person.php?id=2318) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)
- [Maarten De Vos](https://www.esat.kuleuven.be/stadius/person.php?id=203) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven and Dept. Development and Regeneration, KU Leuven)
- [Alexander Bertrand](https://www.esat.kuleuven.be/stadius/person.php?id=331) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)

All authors are affiliated to [LEUVEN.AI - KU Leuven Institute for AI](https://ai.kuleuven.be). 

Abstract
------------
Change Point Detection (CPD) refers to the task of identifying abrupt changes in the characteristics or statistics of time series data. Recent advancements have led to a shift away from traditional model-based CPD approaches, which rely on predefined statistical distributions, toward neural network-based and distribution-free methods using autoencoders. However, many state-of-the-art methods in this category often neglect to explicitly leverage spatial information across multiple channels, making them less effective at detecting changes in cross-channel statistics. In this paper, we introduce an unsupervised, distribution-free CPD method that explicitly incorporates both temporal and spatial (cross-channel) information in multi-channel time series data based on the so-called Time-Invariant Representation (TIRE) autoencoder. Our evaluation, conducted on both simulated and real-life datasets, illustrates the significant advantages of our proposed multi-channel TIRE (MC-TIRE) method, which consistently delivers more accurate CPD results.

Requirements
------------
This code requires:
**tensorflow**,
**tensorflow_probability**,
**tensorflow-addons**,
**numpy**,
**pandas**,
**scipy**,
**matplotlib**,
**seaborn**,
**scikit-learn**.

To install the required packages, run:

```
cd functions
pip install -e .
```

Usage
-----

The main method is implemented in ``MC-main.py`` and can
be run directly from there, or from ``MC-wrapper.py`` which will run the code
for all datasets and parameter settings at once.

Data Sources
-----
HASC-2011: [Link](http://hasc.jp/hc2011/index-en.html)

Honeybee Dance: [Link](http://www.sangminoh.org/Research/Entries/2009/1/21_Honeybee_Dance_Dataset.html)

UCI-test: [Link](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)

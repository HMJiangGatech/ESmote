# esmote
**`esmote`**, an R package including fast SMOTE algorithm.

This is part of my undergraduate final year project. Which provide a really fast implementation of SMOTE algorithm.

If you have any concerns please contact me: jianghm.ustc@gmail.com

Some functions are still underconstruction. 
Ex. I developed a semi-supervised autoencoder to deal with high-dimensional data. However I did not provide a well documented R warpper. You may refer to the source code (`./tests/testPer.R`).

## Installation

First install  `devtools` in `R`:
```R
install.packages("devtools")
```
Install package via `install_github`:

```R
library(devtools)
install_github('HMJiangGatech/ESmote')
```

## Practical Example

This package contains some test data, such as hand written digits data.

```R
newlabel = digitsTrainLabel;
newlabel[newlabel>0] = 1;
newID = sample(60000);
timestart<-Sys.time();

newdata<-esmote::Smote(digitsTrain[newID,],newlabel[newID], algorithm="rp_forest");

timeend<-Sys.time()
runningtime<-timeend-timestart
print(runningtime)
```

Compared to other packasges such as: `DMwR`, `smotefamily`, it is extremely fast.
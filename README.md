# SVDmerge
A python module implementing *SVDmerge*, a batch effects removal method developed primarly for RNAseq data.


### Description
The method is based on singualar value decomposition and uses two-step filtering.
It is descrived in detail in

Francesc Font-Clos, Stefano Zapperi, Caterina A. M. La Porta  
Integrative analysis of pathway deregulation in obesity  
*npj Systems Biology and Applications (3) 18, 2017*  
[link to journal](https://www.nature.com/articles/s41540-017-0018-z)  

where we used it for the first time to merge 4 batches of adipose tissue transcriptomic data
from lean and obese patients. A series of example notebooks for that case can be found [here](https://github.com/ComplexityBiosystems/obesity-score).

### Installation

At the moment, just clone this repository 
and import it via

```
import sys
sys.path.append("/path/to/SVDmerge/")
import SVDmerge
``` 

### Usage
...under construction...


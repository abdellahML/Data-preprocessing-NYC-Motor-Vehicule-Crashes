# Data preprocessing - NYC Motor Vehicule Crashes
nyc_crashes_project

## Context

This project is a data preprocessing prject based [on collected informations about all the traffic accidents that happened in New York City.](https://github.com/becodeorg/LIE-Thomas-1.26/blob/master/content/additional_resources/datasets/NYC%20Motor%20Vehicle%20Crashes/data_100000.csv)

## Description

In this project I have applied all the first required steps to do a data preprocessing to use it in a machine learning process. 

## Usage

You have to run `Main.ipynb` file.

This file contains specific steps from the importation of the data, the creation of profiling report to the exportation of the final result , `output2.csv` and its profiling report, `profile_report_100000_in.html`
Based on the the profiling report, I have implemented the following required steps.

>Resolve anomlies : rename columns, Eliminate blank spaces, Date format, change dtypes,...
>Resolve the High cardinality based on occurence of unique values.
>Missing values:
>> - This step has been the real challenge of this project. 
>> - To avoid to lose informations, i decided to only drop column with more than 99,75 % of missing values. 
>> - I combine all the possible complementary columns. 
>> - I had also to chose the right way to fulfill the Matrix to avoid biased output because of to this manipulation.
>> - One of the challenge was to find the right algorithm iteration to fullfill the zip_code and the borough columns.


## Python Libraries

The needed libraries are in the requirement.txt. To install it, use the command below:  

`python -m pip install -r requirements.txt`  

I used pandas, numpy, pandas_profiling, scipy.spatial.distance and datetime modules : 
```
pip install -U pandas-profiling[notebook]
```

```
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from scipy.spatial.distance import pdist, cdist
from datetime import datetime as dt 
```

### Author
*Abdellah El Ghilbzouri*

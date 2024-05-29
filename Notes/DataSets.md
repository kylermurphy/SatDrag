# Datasets

GOES X-ray and EUV data may be better training datasets as they are avaialbe near-realtime


| Dataset | Time Span |
|---|---|
| GRACE | 2002-2018 |
| GRACE-FO | 2018-2021 | 
| CHAMP | 2001-2011 | 
| GOCE | 2010-2014 | 
| Swarm | 2014-2021 |
| GOES-R |  |
| 18 | 2022-present | 
| 17 | 2018-2022 |
| 16 | 2017-present |
| GOES NOP (XRS) | |
| 15 | 2010-2020 |
| 14 | 2009-2020 |
| 13 | 2015-2017 |
| 12 | 2003-2007 |
| 11 | 2000-2008|
| 10 | 1998-2009|
| 9 | 1996-1998|
| 8 | 1995-2002|
| GOES NOP (EUV) |  |
| 15 | 2010-2020 |
| 14 | 2009-2020 |


## Overlapping Datasets

For operational purpsoses a model using GOES EUVS and XRS data may be the best.

### EUV and XRS Data (R-series)
- Train Data
    - 2018-2021
        - GRACE-FO
        - GOES 16
- Validation Data
    - 2017-2021
        - Swarm
        - GOES 16

## XRS Data
- Train Data
    - 2002-2016
        - GRACE
        - GOES 8,10,11,12,13,14,15
- Validation Data
    - 2002-2012
        - CHAMP
        - GOES 8,10,11,12,13,14,15
    - 2018-2021
        - GRACE-FO
        - GOES 16

## XRS and EUVS (NOP)
- Train
    - 2009-2016
    - GRACE
    - GOES 14, 15 (compare with GOES 16,17,18)
- Validation
    - 2009-2020
    - GRACE FO
    - CHAMP
    - Swarm

## NOP and R-series Overlap
- XRS
    - GOES 14,15,16,17
- EUVS
    - GOES 14,15,16,17



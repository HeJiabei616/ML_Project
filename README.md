# ML_Project #
## Before you run the project ##
- After setup the project, make sure you python is higher than 3.6 to have 'operator' package if not please install python 3.6 and change to python 3 as default by:
```
    alias python=python3
```
## Code structure ##
```
ML_Project
.
├── CN
│   ├── dev.in
│   ├── dev.out
│   ├── dev.p2_out
│   ├── dev.p3_out
│   ├── dev.p4_out
│   ├── train
│   └── trainb
├── EN
│   ├── dev.in
│   ├── dev.out
│   ├── dev.p2_out
│   ├── dev.p3_out
│   ├── dev.p4_out
│   ├── train
│   └── trainb
├── EvalScript
│   ├── Instruction.txt
│   ├── __MACOSX
│   ├── __pycache__
│   │   └── evalResult.cpython-36.pyc
│   ├── dev.in
│   ├── dev.out
│   ├── dev.p2_out
│   ├── dev.p3_out
│   ├── dev.p4_out
│   └── evalResult.py
├── FR
│   ├── dev.in
│   ├── dev.out
│   ├── dev.p2_out
│   ├── dev.p3_out
│   ├── dev.p4_out
│   ├── train
│   └── trainb
├── Part2.py
├── Part3(old).py
├── Part3.py
├── Part4.py
├── SG
│   ├── dev.in
│   ├── dev.out
│   ├── dev.p2_out
│   ├── dev.p3_out
│   ├── dev.p4_out
│   ├── train
│   └── trainb
├── __pycache__
│   ├── Part2.cpython-36.pyc
│   ├── Part3.cpython-36.pyc
│   └── Part4.cpython-36.pyc
└── run.py
```

## How to run the code ##
- a. Direct to ML_Project and execute:

```
    python run.py
```
- b. Input your selected language set to be labeled

```
    Zhous-MacBook-Pro:ML_Project zhouxuexuan$ python run.py
    Choose a language: 
```
- c. Wait till finish (process can be slow due to huge data input)
```
    Zhous-MacBook-Pro:ML_Project zhouxuexuan$ python run.py
    Choose a language: CN
    Part2 output done!
    Part3 output done!
    Part4 output done!
```
- d. Evalute score with evalResult script:
```
    python evalResult.py CN/dev.out CN/dev.p3_out
```

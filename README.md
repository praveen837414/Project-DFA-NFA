# Intelligent DFA/NFA Learning and Visualization Using Training Data

##  Project Overview

This project develops an educational application that learns finite-state automata from labeled training strings and visualizes the inferred automaton.

The system uses the **RPNI (Regular Positive and Negative Inference)** algorithm to learn DFA models from positive and negative examples. It allows users to understand how training examples are converted into automata and how the learned model recognizes unseen strings.

The project is implemented using Python in Google Colab.

##  Objectives

- Generate datasets containing positive and negative strings.
- Implement Prefix Tree Acceptor (PTA).
- Implement the RPNI automata-learning algorithm.
- Construct DFA models from training examples.
- Simulate string acceptance and rejection.
- Visualize automata states and transitions.
- Measure model accuracy and execution time.
- Evaluate generalization using unseen strings.
- Study the effects of training-data quantity and quality.

##  Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab | Development environment |
| NumPy | Numerical operations |
| Pandas | Dataset management |
| Matplotlib | Graph visualization |
| NetworkX | DFA state-transition visualization |
| Scikit-learn | Model evaluation |

##  Algorithm Used

**RPNI – Regular Positive and Negative Inference**

RPNI is a grammatical inference algorithm that learns regular languages from positive and negative examples.

### Working Steps

1. Generate positive and negative training strings.
2. Construct a Prefix Tree Acceptor (PTA).
3. Identify red and blue states.
4. Merge compatible states.
5. Validate state merging using negative examples.
6. Construct the learned DFA.
7. Test the learned model using unseen strings.

## Project Workflow

```text
Training Dataset
       |
       v
Positive and Negative Examples
       |
       v
Prefix Tree Acceptor (PTA)
       |
       v
RPNI Learning Algorithm
       |
       v
State Merging
       |
       v
Learned DFA Construction
       |
       v
DFA Visualization
       |
       v
String Recognition
       |
       v
Performance Evaluation
```

##  Example Application

The project uses the regular language:

**Binary strings ending with 01**

- Alphabet: `{0, 1}`
- Positive examples: `01`, `101`, `1001`, `1101`
- Negative examples: `10`, `11`, `000`, `1010`

### Expected Recognition Results

| Input String | Expected Result |
|---|---|
| 01 | Accepted |
| 101 | Accepted |
| 1001 | Accepted |
| 1101 | Accepted |
| 10 | Rejected |
| 11 | Rejected |
| 1010 | Rejected |

##  Project Modules

1. Dataset Generation
2. Training Data Preparation
3. Prefix Tree Acceptor Construction
4. RPNI Automata Learning
5. DFA Construction
6. Automata Visualization
7. String Acceptance and Rejection
8. Accuracy and Performance Evaluation

##  How to Run the Project

### Step 1: Open Google Colab

Visit: https://colab.research.google.com/

### Step 2: Create a Notebook

Create a new Python notebook.

### Step 3: Install Required Libraries

```python
!pip install numpy pandas matplotlib networkx scikit-learn
```

### Step 4: Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
```

### Step 5: Execute the Project

Run the project Python cells in sequence:

- Dataset generation
- PTA construction
- RPNI implementation
- DFA construction
- DFA visualization
- String testing
- Performance evaluation

## 📈 Performance Evaluation

The project evaluates the learned automaton using:

- Training accuracy
- Testing accuracy
- Generalization accuracy
- Number of learned DFA states
- RPNI execution time

##  TOC Concepts Used

- Deterministic Finite Automata (DFA)
- Nondeterministic Finite Automata (NFA)
- Regular Languages
- Automata Learning
- State Minimization
- Automata Equivalence

##  Future Enhancements

- Implement NFA construction and simulation.
- Add interactive automata visualization.
- Support configurable alphabets and regular languages.
- Introduce noisy and insufficient training datasets.
- Compare DFA models learned using different training-data sizes.
- Add active learning and additional classification techniques.
- Develop an interactive educational dashboard.

##  Expected Outcome

An educational Python application that learns approximate finite-state automata from labeled strings, visualizes the inferred DFA, tests unseen strings, and evaluates the effect of training-data quantity and quality.

##  Project Information

**Project Name:** Intelligent DFA/NFA Learning and Visualization Using Training Data

**Domain:** Theory of Computation and Machine Learning

**Programming Language:** Python

**Platform:** Google Colab

**Learning Algorithm:** RPNI

---

⭐ If you find this project useful, consider giving the repository a star.

## ACRL - Assignment 2 - Tetris
--

### Setup

To run the project, Numpy, Scikit-learn and py4j are required, to install them you can run:

```pip install -r requirement.txt```

The project was tested with python 3.6.

### Run Java Gateway

For both training or testing make sure you have the Java Server running.
To run the server execute the below command from the `src/` folder:

Linux:
```java -cp .:py4j0.10.8.1.jar StateEntryPoint```

Windows:
```java -cp .;py4j0.10.8.1.jar StateEntryPoint```

### Training

To run the training the agent script run:

```python train.py```

### Testing

To test the agent with pretrained weights run:

```python test.py```

To run for a specific number of games, specify N (default is 5)

```python test.py --iter N```

To run with a visualization in python, use the --vis Flag

```python test.py --iter 1 --vis```

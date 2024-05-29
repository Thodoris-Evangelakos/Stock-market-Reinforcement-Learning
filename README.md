**Stock market Reinforcement Learning**

This project is a Python implementation of the Multiplicative Weights Algorithm (MWA) both in an expert and bandit environment for stock trading. The algorithms are used to make decisions on which stocks to buy based on historical data, with the goal of maximizing profit and minimizing regret.

**File Structure**
main.py: This is the main entry point of the application. It contains the main function and several other functions for running experiments and plotting results.
MWA.py: This file contains the implementation of the Multiplicative Weights Algorithm in an expert environment.
MWBA.py: This file contains the implementation of the Multiplicative Weights Algorithm in a Bandit environment.
stocks.csv: This file contains historical stock data used by the algorithms.
Usage
To run the project, execute the main.py script. This script loads the stock data from stocks.csv, creates instances of the MWA and MWBA algorithms, and runs them. The results are then plotted using Matplotlib.

**Algorithms**
Multiplicative Weights Algorithm (MWA)
The MWA is implemented in the MWA class in MWA.py. The class is initialized with stock data and optional parameters for transaction fees and a custom learning rate (ita). The algo_execution method runs the algorithm and returns the cumulative profits and cumulative regret.

Multiplicative Weights Bandit Algorithm (MWBA)
The MWBA is implemented in the MWBA class in MWBA.py. Like the MWA, the class is initialized with stock data and optional parameters for transaction fees and a custom learning rate (ita). The algo_execution method runs the algorithm and returns the cumulative profits and cumulative regret.

Experiments
The main.py script contains several functions for running experiments and plotting results:

main: This function runs the MWA and MWBA algorithms with and without transaction fees and plots the cumulative profits and cumulative regret. More specifically, it tests the MWA both when fees are and aren't present, finally testing the MWBA in an environment with fees.
experimental: This function runs the MWA algorithm with different values of ita and plots the cumulative profits and cumulative regret.
ita_var_averages: This function runs the MWA algorithm multiple times with different values of ita and averages the results.
epsilon_var_averages: This function runs the MWBA algorithm multiple times with different values of epsilon and averages the results.

**Dependencies**
This project requires Python 3 and the following Python libraries installed:

NumPy
Matplotlib

**License**
Pending

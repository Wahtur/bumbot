\# Project Overview: Classical ML Chess Bot (TD-Leaf)

\*\*Goal:\*\* Build a functional chess bot utilizing classical machine learning (Reinforcement Learning via TD-Leaf) to meet a university final project rubric.

\*\*Deadline:\*\* December 10, 2026

\*\*Target Elo:\*\* 100+ (Baseline capability)

\*\*Rubric Constraint:\*\* Must demonstrate the effort of "two mini-projects" (handled by splitting the Search Engine and the ML Math).



\## Tech Stack \& Environment

\*   \*\*Language:\*\* Python 3.11

\*   \*\*Libraries:\*\* `python-chess` (for board state and move generation), `torch` (PyTorch CPU for TD-Leaf gradient calculations)

\*   \*\*Environment Manager:\*\* Conda

\*   \*\*Version Control:\*\* Git / GitHub



\## System Architecture

The bot relies on a classical ML architecture, deliberately avoiding Deep Neural Networks (CNNs) to satisfy the classical algorithmic requirement. 

1\.  \*\*Evaluation Function:\*\* A linear combination of handcrafted board features and their respective weights: `Score = (w1 \* f1) + (w2 \* f2) + ... + (wn \* fn)`.

2\.  \*\*Search Engine:\*\* A Minimax algorithm using Alpha-Beta pruning to look 3-4 moves ahead, evaluating leaf nodes using the Evaluation Function.

3\.  \*\*Learning Algorithm (TD-Leaf):\*\* A Reinforcement Learning loop where the bot plays itself. The Temporal Difference (TD) algorithm backpropagates the final win/loss signal to adjust the weights (`w`) using PyTorch's automatic differentiation.



\## Feature Engineering Specification

The agent should help extract the following features using `python-chess` into a 1D tensor:

\*   \*\*Material Imbalance:\*\* Net difference in pawns, knights, bishops, rooks, and queens.

\*   \*\*Piece-Square Tables:\*\* 64 binary features per piece type per color (640 features total) indicating board presence.

\*   \*\*Mobility:\*\* Difference in total legal moves between White and Black.

\*   \*\*Center Control:\*\* Count of pieces attacking d4, d5, e4, e5.

\*   \*\*Structure:\*\* Booleans for doubled pawns, isolated pawns, and king shield integrity.



\## Division of Labor (For Agent Context)

When writing code, ensure the modules are distinct so the two partners can take ownership of their respective mini-projects:

\*   \*\*Partner A (Search \& Features):\*\* Owns `feature\_extractor.py` and `search\_engine.py` (Minimax/Alpha-Beta).

\*   \*\*Partner B (ML \& Self-Play):\*\* Owns `td\_leaf\_trainer.py` (PyTorch weight definitions, loss calculation, and the self-play simulation loop).



\## Immediate Agent Tasks

1\. Initialize the `feature\_extractor.py` file and write a function that takes a `chess.Board` object and returns a PyTorch tensor of the engineered features.

2\. Initialize `search\_engine.py` with a standard Alpha-Beta pruning skeleton that accepts the feature tensor and a weight tensor to score board states.


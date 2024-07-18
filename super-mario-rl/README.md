# Super Mario Bros using Reinforcement Learning

This project implements a Reinforcement Learning (RL) agent to play the classic Super Mario Bros game. The agent is trained to navigate through the game's environment using deep learning techniques.

## Project Structure

- `main.py`: The main script to run the training and evaluation of the RL agent.
- `agent.py`: Contains the RL agent implementation.
- `wrappers.py`: Utility functions to wrap the game environment for preprocessing and other purposes.
- `agent_nn.py`: Neural network architecture used by the RL agent.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- `gym-super-mario-bros`: Super Mario Bros environment for OpenAI Gym
- `nes-py`: Emulator for NES games
- `numpy`: Library for numerical computations
- `tensorflow` or `pytorch`: Deep learning frameworks
- `opencv-python`: Library for image processing

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/arnabsaha7/Reinforcement-Learning-Projects/super-mario-rl.git
    cd supermariobros-rl
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

### Running the Project

1. To start training the RL agent:

    ```sh
    python main.py
    ```

2. To evaluate a pre-trained agent:

    ```sh
    python main.py --mode evaluate --model_path path_to_your_model
    ```

## Project Details

### RL Agent

The RL agent is implemented in `agent.py` and uses a deep Q-network (DQN) approach with experience replay and target network for stable training.

### Environment Wrappers

The environment wrappers in `wrappers.py` include functions for:
- Frame skipping and pooling
- Grayscale conversion
- Resizing frames
- Stacking frames

### Neural Network Architecture

The neural network architecture used by the RL agent is defined in `agent_nn.py`. It consists of convolutional layers followed by fully connected layers to process the game screen inputs and output Q-values for each action.

## Results

After training, the RL agent is able to navigate through the levels of Super Mario Bros, learning to jump over obstacles, defeat enemies, and reach the goal. The training process and performance of the agent can be visualized using TensorBoard.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
- [nes-py](https://github.com/Kautenja/nes-py)
- [Deep Q-Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)


## Installation

`pip install -e gym-duckietown`

## Running the Simulation

To initialize the gym-duckietown environment, there are five arguments:

- `--map-name`: the name of the map
- `--seed`: random seed of the environment
- `--start-tile`: the starting tile. E.g. 1,13
- `--goal-tile`: the goal tile. E.g. 3,3
- `--max-steps`: the maximum run step. The default value is 1500. Do not change this default value when you generate the
  control files for submission.

E.g. python example.py --map-name map4_0 --seed 2 --start-tile 1,13 --goal-tile 3,3
![Screenshot (392)](https://user-images.githubusercontent.com/65244703/166099624-4eea2e01-3f20-4aeb-93d4-1b6d231fa71f.png)

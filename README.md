# Unsupervised Representation Learning in Atari
__Reproduced by__: Shekar Ramaswamy, Lawrence Huang, Kendrick Tan, and Tyler Jiang

Original Code can be found here:
https://github.com/mila-iqia/atari-representation-learning

## Project Structure
    .
    ├── data_representation
        ├── get_data.py: collects episode data from Atari games
    ├── data_viz: produce charts of experiment results
    ├── encoders: holds encoder architectures and saved encoders trained with ST-DIM
        ├── rand_cnn.py: base CNN architecture used in Random-CNN and ST-DIM
    ├── handlers
        ├── probe_handler.py: trains the probe (supervised and unsupervised)
        ├── stdim_handler.py: trains the encoder and bilinear layers using ST-DIM (InfoNCE ST-DIM, no ablations)
    ├── probe
        ├── probe.py: regular probe and fully supervised probe (linear layer and linear layer + encoder respectively)
    ├── pipeline.py: runs the pipeline end-to-end (collects data, trains, validates, and tests the encoder)
    └── test_stdim_handler.py: train and save the encoder using ST-DIM or run the probe with a trained encoder
    

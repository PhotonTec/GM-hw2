# DCGAN Generative Model for CIFAR-10

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate images from the CIFAR-10 dataset. The repository includes training and testing scripts for two models: one capable of generating images in specific categories (`model_withlabel.py`), and another that generates images without category constraints (`model.py`). The training code for the labeled generator is in `train_wl.py`. Checkpoints for both models are stored in the `ckpt` folder, and the `requirements.txt` file lists the necessary dependencies.

## Project Structure

- `codes/`
  - `train.py`: Script for training the DCGAN model without category constraints.
  - `test.py`: Script for testing the generated images from the DCGAN model without category constraints.
  - `model.py`: Model definition for the generator without category constraints.
  - `model_withlabel.py`: Model definition for the generator with category constraints.
  - `train_wl.py`: Script for training the DCGAN model with a labeled generator.

- `ckpt/`: Folder containing checkpoints for both models.

- `requirements.txt`: File listing the dependencies required to run the project.

## Getting Started

### Prerequisites

- Python 3.8
- Pip (Python package installer)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/PhotonTec/GM-hw2.git
   cd GM-hw2

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Training the DCGAN Model without Category Constraints

Run the training script for the DCGAN model without category constraints:

```bash
python codes/train.py
```

#### Training the Labeled DCGAN Model

Run the training script for the labeled DCGAN model:

```bash
python codes/train_wl.py
```

#### Testing the Generated Images

Run the testing script to generate images using the trained DCGAN model:

```bash
python codes/test.py
```

## Project Results

Include any relevant results or findings from the training and testing processes.

## Contributing

Feel free to open issues or submit pull requests.

## License

This project is licensed under the [MIT License](https://chat.openai.com/c/LICENSE).

## Acknowledgments

- This is homework2 of generative model class
- Author:2100013158 Xu Tianyi# GM-hw2

# GM-hw2
# GM-hw2

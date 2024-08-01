Llama 2 7B Quantization to 4-bit

This project demonstrates the quantization of the Llama 2 7B model to 4-bit precision, optimizing it for reduced memory usage and faster inference while maintaining a balance between performance and accuracy.
Table of Contents

    Introduction
    Prerequisites
    Installation
    Quantization Process
    Usage
    Evaluation
    Results
    Contributing
    License

Introduction

Quantization is a technique that reduces the precision of the weights and activations of a neural network. This project focuses on quantizing the Llama 2 7B model to 4-bit precision, making it more efficient for deployment on hardware with limited resources.
Prerequisites

    Python 3.7 or later
    PyTorch
    Transformers library
    NumPy
    Llama 2 model weights

Installation

To set up the environment and install the necessary dependencies, follow these steps:

  git clone https://github.com/yourusername/llama-2-7b-quantization.git
  cd llama-2-7b-quantization
  pip install -r requirements.txt

Quantization Process
The quantization process involves converting the Llama 2 7B model's weights from 32-bit floating point to 4-bit integers. This is achieved using the following steps:

    Model Loading: Load the pre-trained Llama 2 7B model.
    Weight Conversion: Convert the weights to 4-bit representation.
    Model Saving: Save the quantized model for future use.

The script quantize.py automates this process:
  python quantize.py --model-path /path/to/llama-2-7b --output-path /path/to/output

Usage
To use the quantized model, load it in your PyTorch environment and run inference as you would with the original model. Here's an example:

python

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/path/to/quantized-model")
model = AutoModelForCausalLM.from_pretrained("/path/to/quantized-model")

input_text = "Once upon a time"
input_ids = tokenizer(input_text, return_tensors='pt').input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

Evaluation

To evaluate the performance of the quantized model, use the evaluate.py script. This script compares the performance and accuracy of the quantized model against the original model on a chosen dataset.
  python evaluate.py --quantized-model /path/to/quantized-model --original-model /path/to/original-model --dataset /path/to/dataset

Results

After running the evaluation, you can analyze the results to see the trade-offs between model size, inference speed, and accuracy. Detailed results can be found in the results directory.
Contributing

We welcome contributions to this project! Please see the CONTRIBUTING.md file for more details on how to contribute.
License

This project is licensed under the MIT License. See the LICENSE file for more details.

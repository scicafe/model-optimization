# Model Optimization
An Exploration of model optimization using the TensorFlow Model Optimization Toolkit (TFMOT) and TensorFlow Lite (TFLite)


## Released Blogs

#### Quantization Aware Training using TFMOT
What you will learn:
- Introduction to TFMOT
- Introduction to QAT
- How QAT works
- Code: A quantization aware model
- Introduction to PTQ
- Introduce TFLite
- Code: Quantize a model using PTQ and TFLite
- Code: Compare Performance of different quantization methods
- Save models and compare model size of different quantization methods

What you will be able to do after reading the blog:
- Understand the differences between different quantization techniques
- Write code to quantize models using different quantization techniques
- Compare the performance of different quantization techniques
- Ch0ose the correct quantization technique for your application

[Blog](https://sci.cafe/quantization-1) | [Colab Notebook](https://colab.research.google.com/drive/1-xiwp2s1Oir8sNh-Utnj60yAtpjwDaUr?usp=sharing) | [Trained Models](https://github.com/scicafe/model-optimization/releases/tag/v1.0) | [Code](/qat_using_tfmot/)

## Future Blogs
- INT8 Inference and Model Latency
    - Measure the latency of models trained in previous blog
    - Introduction to EdgeTPU and Accelerator
    - Convert model and run it on EdgeTPU
- Model Pruning using TFMOT
    - Introduce TFMOT's Model Pruning API
    - Prune Models
    - Exploration of performance
- Choose layers to quantize
    - An NAS approach to chosing which layers to quantize

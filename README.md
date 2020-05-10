# model-optimization
An Exploration of model optimization using TFMOT and TFLite


BLogs Planned:
- Introduction to TFMOT and TFLite: [colab notebook](https://colab.research.google.com/drive/1-xiwp2s1Oir8sNh-Utnj60yAtpjwDaUr?usp=sharing)
    - Introduce TFMOT
    - Introduce QAT
    - Show example of QAT
    - Post training quantization
    - Introduce TFLite
    - Quantize model
    - Compare Performance
    - Save models and compare performance
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

## Real-time ASL fingerspelling

The dataset for this project source and the initial model development are based on this [kaggle dataset](https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out) and [kaggle notebook](https://www.kaggle.com/code/brussell757/american-sign-language-classification).

The current neural network used is VGG19 model with slight updates. Pretrain the model using the notebook and use the model to the realtime_prediction.py file to read video from webcam and for fingerspelling.

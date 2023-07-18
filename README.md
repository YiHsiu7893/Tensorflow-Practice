# Tensorflow-Practice
This is a collection of practice code developed throughout my self-learning experience with TensorFlow.

Reference book: "TensorFlow 2.x人工智慧、機器學習超炫範例200+", published by 碁峰資訊, 2021

The codes are not the same as the samples in the book. I have modified a piece of code according to my habits.

---
Contents
- MLP_Start.py - the first MLP model I constructed
- MLP_OneHot.py - same as MLP_start.py, but apply OneHot-Encoding on labels
- MLP_MultiInput.py - a more complex structured model, adjust to see the ability of the model
- SimpleSL.py - simulate the model on "TensorFlow Playground", and visualize weights and biases of the neuron
- MultiSL.py - same as SimpleSL.py, but with multiple neurons in the hidden layer
- IristoExcel.py - turn Iris dateset to an Excel file
- iris.csv - by-product csv file created during the execution of IristoExcel.py
- iris.xlsx - by-product Excel file created during the execution of IristoExcel.py
- MLP_Iris.py - real case

---
Note: steps of training a MLP model
1. collecting required data
2. preprocessing features or labels if needed
3. building a MLP model
4. compiling and training the model
5. testing and evaluating the ability of the model
6. predicting new data with the trained model

---
Note: approaches to improve accuracy of a model
1. adjust "epochs"
2. adjust "batch_size"
3. increase the number of neurons "units"
4. increase the number of hidden layers
5. increase the number of training data

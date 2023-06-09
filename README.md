
# Constructing a Neural Network with Backpropagation in Excel

In this tutorial, we will see how to build a neural network with backpropagation in Excel.

We’ll use simple formulas and functions to create our neural network and will provide clear and easily understandable explanations of the concepts underlying neural networks and backpropagation.

##  Prerequisites
Before delving into the captivating realm of neural networks, let's familiarize ourselves with Logistic Regression. Logistic Regression is a Machine Learning algorithm utilized for classification problems. It operates as a predictive analysis algorithm, relying on the concept of probability.

Unlike linear regression, Logistic Regression employs a more intricate cost function known as the **'Sigmoid function'** or **'logistic function'**. This choice is made to accommodate the specific requirements of the hypothesis in logistic regression. The aim is to confine the cost function within the range of 0 and 1.

Linear functions are inadequate for representing logistic regression because they can yield values greater than 1 or less than 0, which contradicts the hypothesis of logistic regression. The sigmoid function resolves this issue by constraining the output within the desired range.

## What is the Sigmoid Function?
To establish a correspondence between predicted values and probabilities, we employ the Sigmoid function. This function transforms any real value into a value ranging from 0 to 1. In the context of machine learning, we utilize the Sigmoid function to map predictions to probabilities.

<p align="center">    
    <img width="200" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/1.png" />
</p>

### Hypothesis Representation
When using linear regression we used a formula of the hypothesis i.e.
```bash
    hΘ(x) = β₀ + β₁X
```

For logistic regression we are going to modify it a little bit i.e.

```bash
    σ(Z) = σ(β₀ + β₁X)
```

We have expected that our hypothesis will give values between 0 and 1.

```bash
    Z           = β₀ + β₁X

    hΘ(x)       = sigmoid(Z)

    i.e. hΘ(x)  = 1/(1 + e^-(β₀ + β₁X)     
```
<p align="left" padding-left=10>    
   &ensp;&ensp;&ensp;&ensp;&ensp;&ensp; <img width="200" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/2.png" />
</p> 

### Cost Function
The cost function J(θ) represents optimization objective i.e. we create a cost function and minimize it so that we can develop an accurate model with minimum error.
```bash
    J(θ) = ½ (yi – ŷi)2
```
where:
- o	yi is the ith observed value.
- o	ŷi is the corresponding predicted value.
- o	n = the number of observations.
- o	"(yi – ŷi)" calculates the difference between the actual value and the predicted value.
- o	"(yi – ŷi)^2" squares the difference, resulting in a positive value.
- o	"1/2" scales the squared difference by a factor of 0.5, which is equivalent to multiplying it by 0.5.

The purpose of dividing by 2 is to simplify the mathematical calculations and does not significantly affect the optimization process when minimizing the cost function. It is often included for convenience and mathematical consistency.

By summing up the squared differences for each data point and taking the average, the mean squared error (MSE) provides a measure of how well a regression model fits the data. Minimizing the MSE helps to find the optimal parameters of the model that result in the smallest overall prediction error.

## Backpropagation 
[In the Excel implementation of Backpropagation](https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/Back-Propagation.xlsx), we can examine the detailed calculations involved in the process. Here are the steps:

    1. Randomly initialize the weights of the neural network.
    2. Perform a feedforward pass, where the inputs are propagated through the network to obtain the predicted output.
    3. Calculate the error by comparing the predicted output with the actual output.
    4. Backpropagate the error through the network to compute the gradient of the loss function with respect to the weights.
    5. Update the weights using the calculated gradient and a predetermined learning rate.
    6. Repeat steps 2 to 5 for a specified number of epochs until the network converges to an acceptable solution.

By following these steps iteratively, the neural network gradually learns and adjusts its weights to minimize the error and improve its performance.

### 1.1 Initialization:
The values mentioned below are considered constant or fixed in this context. 

<p align="left">    
    <img width="300" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/3.png" />
</p>


The objective here is to construct a neural network in which we can adjust the weights (W1, W2, W3, W4, etc.) such that when given input values i1 = 0.05 and i2 = 0.1, the network produces the desired outputs t1 = 0.5 and t2 = 0.5.

<p align="left">    
    <img width="620" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/4.png" />
</p>

We will begin by computing the values for h1, h2, a_h1, and a_h2. Here are the formulas to compute these values:

<p align="left">    
    <img width="300" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/5.png" />
</p>

We will assign random weights to w1, w2, w3, w4, w5, w6, w7, and w8.

<p align="left">    
    <img  height="50" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/6.png" />
</p>

### 1.2 Forward propagation:

We will perform forward propagation, which involves passing the inputs through the neural network layers to compute the final output.

We will write down the formulas to compute the aforementioned values.

<p align="left">    
    <img  height="50" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/7.png" />
</p>

### 1.3 Partial derivates

Now, we will compute the derivatives of the error with respect to the weights. In backpropagation, one of the main steps is to calculate the derivative of the error (often represented as the loss function) with respect to the weights of the neural network. This derivative, also known as the gradient, indicates the direction and magnitude of the change needed in the weights to minimize the error. By calculating the derivative of the error with respect to the weights, backpropagation allows us to update the weights in a way that gradually improves the network's performance over time.

First, calculate derivates of ∂E/∂w1, ∂E/∂w2, ∂E/∂w3, ∂E/∂w4, ∂E/∂w5, ∂E/∂w6, ∂E/∂w7, and ∂E/∂w8.

Let's derive the formula for calculating the derivative of the error with respect to W5.

<p align="left">    
    <img  width="350" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/8.png" />
</p>

Similarly, calculate derivatives error with respect to W6, W7, W8

<p align="left">    
    <img height="80" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/9.png" />
</p>

<p align="left">    
    <img height="80" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/10.png" />
</p>

<p align="left">    
    <img height="70" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/11.png" />
</p>

<p align="left">    
    <img height="80" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/12.png" />
</p>

We will utilize these formulas to populate the cells and calculate the derivatives of the error with respect to all available weights as shown below.

<p align="left">    
    <img height="50" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/13.png" />
</p>

### 1.4 Updating weights

In backpropagation, the weights of the neural network are updated using an optimization algorithm. 
This process involves multiplying the gradient by a learning rate, which determines the step size for the weight update. Then subtract this value from the current weight to update it. The learning rate controls how quickly or slowly the weights are adjusted.

E.g., new weight for W1

= current W1 – LR (Learning Rate) * derivate of error with respect to W1

= W1 – (η*∂E/∂w1)

Likewise, we will apply this formula to update all the weights, including w1, w2, … w8.

<p align="left">    
    <img height="50" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/14.png" />
</p>

Now, we can repeat this process for a specified number of iterations, such as 60, by simply copying and pasting the Excel formulas as shown below.

<p align="left">    
    <img height="260" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/15.png" />
</p>


### 1.4 Loss graph

A loss graph in backpropagation illustrates the trend of the loss function's value over the course of the training iterations. It provides a visual representation of how the loss changes as the neural network adjusts its weights through backpropagation.

To create a loss graph in backpropagation, you typically plot the iteration number (x-axis) against the corresponding loss value (y-axis). As the training progresses, you can observe the loss decreasing, indicating that the network is converging towards an optimal solution.

Typically, you would expect the loss to gradually decrease over iterations, indicating that the network is improving its performance and minimizing errors. However, it's important to note that the exact shape and trend of the loss graph can vary depending on factors such as the complexity of the problem, network architecture, and hyperparameter settings.

Loss with respect to Learning rate 1

<p align="left">    
    <img width="530" aling="right" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/16.png" />
</p>

Loss with different Learning Rate (η)

<p align="left">    
    <img hight="100" aling="left" src="https://github.com/Paurnima-Chavan/backpropagation-in-excel/blob/main/imgs/18-new.png" />
</p>

Monitoring the loss graph during backpropagation provides valuable insights into the training process and helps assess the effectiveness of the learning algorithm in optimizing the neural network.

## Conclusion
We have successfully completed the implementation of our simple neural network, which incorporates both forward propagation and backpropagation. This implementation allows us to experiment with various factors such as initial values and learning rate. We have learned how to calculate the partial derivatives of the weights with respect to the cost function and how to update the weights using the gradient descent process to minimize the cost function. Additionally, we have emphasized the significance of selecting appropriate initial values and determining the step size for the gradient descent algorithm. By undertaking this exercise, we have deepened our comprehension of neural networks and their practical application in solving real-world problems.

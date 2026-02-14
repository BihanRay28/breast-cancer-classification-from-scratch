🧠 Breast Cancer Classification from Scratch

End-to-end implementation of a two-layer neural network built entirely from scratch using NumPy for binary classification on the Breast Cancer Wisconsin dataset.

No high-level ML libraries were used for model training — only NumPy and manual implementation of neural network mathematics.

📌 Project Objective

The purpose of this project was to deeply understand:

Forward propagation

Backpropagation

Gradient descent

Binary cross-entropy loss

Parameter initialization

Model evaluation

Rather than using frameworks like TensorFlow or PyTorch, the model logic is implemented manually.

🏗 Model Architecture
Input Layer (30 features)
        ↓
Hidden Layer (8 neurons, tanh activation)
        ↓
Output Layer (1 neuron, sigmoid activation)


Loss Function:

Binary Cross Entropy

Optimization:

Gradient Descent

📊 Results

After training for 8000 iterations:

Training Accuracy: ~98.9%

Test Accuracy: ~98.2%

The cost decreases smoothly during training, confirming stable gradient descent behavior.

📁 Project Structure
data.py       → Dataset loading & preprocessing
nn_core.py    → Neural network logic (forward + backward propagation)
train.py      → Training loop
predict.py    → Prediction function
main.py       → Execution + visualization
📚 Dataset

Breast Cancer Wisconsin (Diagnostic) dataset  
- 569 samples  
- 30 numerical features  
- Binary classification (Malignant / Benign)

🧠 Key Learnings

- Importance of feature normalization for stable training
- Sensitivity of convergence to learning rate
- Role of non-linearity (tanh) in improving classification
- How backpropagation propagates gradients layer-by-layer

  🔭 Next Steps

- Implement L2 regularization
- Add dropout
- Compare against logistic regression baseline
- Rebuild using PyTorch for scalability


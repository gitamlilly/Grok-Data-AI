// Data Collection and Analysis AI - JavaScript Logic
// Reasoning: This script handles data collection in an array, initializes ml5.js neural network for regression,
// trains on user data, and predicts outputs. Comments explain each section's purpose.
// We use vanilla JS for event handling to keep it simple and browser-compatible.
// Data is stored in memory (array) for this demo; in production, use localStorage or a backend.

// Global variables
let data = []; // Array to store collected data points: {inputs: [num1, num2], output: num}
let neuralNetwork; // ml5 neural network instance
let isTrained = false; // Flag to check if model is trained

// DOM elements
const dataForm = document.getElementById('data-form');
const dataList = document.getElementById('data-list');
const trainBtn = document.getElementById('train-btn');
const trainingStatus = document.getElementById('training-status');
const predictForm = document.getElementById('predict-form');
const predictionResult = document.getElementById('prediction-result');

// Event listener for adding data
// Reasoning: Prevent default form submit, extract values, store as object, update UI list.
dataForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const input1 = parseFloat(document.getElementById('input1').value);
    const input2 = parseFloat(document.getElementById('input2').value);
    const output = parseFloat(document.getElementById('output').value);
    
    if (isNaN(input1) || isNaN(input2) || isNaN(output)) {
        alert('Please enter valid numbers.');
        return;
    }
    
    data.push({ inputs: [input1, input2], output: [output] }); // Store as array for ml5 compatibility
    updateDataList();
    dataForm.reset();
});

// Function to update displayed data list
// Reasoning: Clear list, repopulate with current data for real-time feedback.
function updateDataList() {
    dataList.innerHTML = '';
    data.forEach((item, index) => {
        const li = document.createElement('li');
        li.textContent = `Data ${index + 1}: Inputs [${item.inputs[0]}, ${item.inputs[1]}], Output ${item.output[0]}`;
        dataList.appendChild(li);
    });
}

// Event listener for training
// Reasoning: Check for at least 4 data points (minimum for demo), init network, add data, normalize, train.
trainBtn.addEventListener('click', () => {
    if (data.length < 4) {
        alert('Add at least 4 data points to train.');
        return;
    }
    
    trainingStatus.textContent = 'Initializing...';
    
    // Neural network options
    // Reasoning: 2 inputs, 1 output for regression task. Debug true for console logs.
    const options = {
        inputs: 2,
        outputs: 1,
        task: 'regression',
        debug: true
    };
    
    neuralNetwork = ml5.neuralNetwork(options);
    
    // Add all collected data to network
    // Reasoning: Loop through data array and add each point.
    data.forEach(item => {
        neuralNetwork.addData(item.inputs, item.output);
    });
    
    // Normalize data for better training
    // Reasoning: Standardization helps convergence in neural networks.
    neuralNetwork.normalizeData();
    
    // Train the model
    // Reasoning: 50 epochs as a balance between training time and accuracy for small datasets.
    // Callback when done.
    neuralNetwork.train({ epochs: 50 }, () => {
        trainingStatus.textContent = 'Training complete!';
        isTrained = true;
    });
});

// Event listener for prediction
// Reasoning: Check if trained, get inputs, predict using network, display result.
predictForm.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!isTrained) {
        alert('Train the model first.');
        return;
    }
    
    const predInput1 = parseFloat(document.getElementById('pred-input1').value);
    const predInput2 = parseFloat(document.getElementById('pred-input2').value);
    
    if (isNaN(predInput1) || isNaN(predInput2)) {
        alert('Please enter valid numbers.');
        return;
    }
    
    const inputs = [predInput1, predInput2];
    
    // Make prediction
    // Reasoning: ml5 predict returns array with value and label; we use value for regression.
    neuralNetwork.predict(inputs, (err, results) => {
        if (err) {
            console.error(err);
            predictionResult.textContent = 'Error in prediction.';
            return;
        }
        predictionResult.textContent = `Predicted Output: ${results[0].value.toFixed(4)}`;
    });
});
// Enhanced Data Collection and Analysis AI - JavaScript Logic
// Reasoning: Use object format for data (named inputs/outputs) for better readability and ml5 compatibility.
// Custom layers: Hidden dense with relu, output dense with no activation for unbounded regression.
// Added config for hyperparameters, CSV import/export, Chart.js visualization, model save/load.
// Global state for data, model, config, chart.
// Error handling and loading states to prevent errors during train/predict.

// Global variables
let data = []; // Array of {input1, input2, output}
let neuralNetwork; // ml5 neural network
let isTrained = false;
let isTraining = false;
let chart; // Chart.js instance

// Default config
let config = {
    epochs: 50,
    learningRate: 0.2,
    hiddenUnits: 16
};

// DOM elements
const configForm = document.getElementById('config-form');
const dataForm = document.getElementById('data-form');
const dataTableBody = document.querySelector('#data-table tbody');
const dataStats = document.getElementById('data-stats');
const exportCsvBtn = document.getElementById('export-csv');
const importCsvInput = document.getElementById('import-csv');
const importBtn = document.getElementById('import-btn');
const clearDataBtn = document.getElementById('clear-data');
const trainBtn = document.getElementById('train-btn');
const trainingStatus = document.getElementById('training-status');
const saveModelBtn = document.getElementById('save-model');
const loadModelInput = document.getElementById('load-model-file');
const loadModelBtn = document.getElementById('load-model');
const predictForm = document.getElementById('predict-form');
const predictionResult = document.getElementById('prediction-result');
const dataChartCanvas = document.getElementById('data-chart').getContext('2d');

// Initialize chart
// Reasoning: Scatter plot for data visualization; input1 vs output, bubble size by input2 for 2D approx.
function initChart() {
    chart = new Chart(dataChartCanvas, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Data Points',
                data: [],
                backgroundColor: 'rgba(75, 192, 192, 0.6)'
            }, {
                label: 'Predicted Line',
                data: [],
                type: 'line',
                borderColor: 'rgba(255, 99, 132, 1)',
                fill: false
            }]
        },
        options: {
            scales: {
                x: { title: { display: true, text: 'Input 1' } },
                y: { title: { display: true, text: 'Output' } }
            }
        }
    });
}
initChart();

// Event listener for config
// Reasoning: Update config from form; used in training options.
configForm.addEventListener('submit', (e) => {
    e.preventDefault();
    config.epochs = parseInt(document.getElementById('epochs').value);
    config.learningRate = parseFloat(document.getElementById('learningRate').value);
    config.hiddenUnits = parseInt(document.getElementById('hiddenUnits').value);
    alert('Configuration updated!');
});

// Event listener for adding data
// Reasoning: Validate numbers, store as object, update UI, stats, chart.
dataForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const input1 = parseFloat(document.getElementById('input1').value);
    const input2 = parseFloat(document.getElementById('input2').value);
    const output = parseFloat(document.getElementById('output').value);

    if (isNaN(input1) || isNaN(input2) || isNaN(output)) {
        alert('Please enter valid numbers.');
        return;
    }

    data.push({ input1, input2, output });
    updateDataUI();
    dataForm.reset();
});

// Function to update data table, stats, chart
// Reasoning: Rebuild table, calculate mean/std, update scatter with bubble for input2.
function updateDataUI() {
    dataTableBody.innerHTML = '';
    data.forEach(item => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${item.input1}</td><td>${item.input2}</td><td>${item.output}</td>`;
        dataTableBody.appendChild(tr);
    });

    if (data.length > 0) {
        const outputs = data.map(d => d.output);
        const mean = outputs.reduce((a, b) => a + b, 0) / outputs.length;
        const std = Math.sqrt(outputs.reduce((a, b) => a + (b - mean) ** 2, 0) / outputs.length);
        dataStats.textContent = `Data Stats: Count: ${data.length}, Mean Output: ${mean.toFixed(2)}, Std Dev: ${std.toFixed(2)}`;
    } else {
        dataStats.textContent = '';
    }

    chart.data.datasets[0].data = data.map(d => ({ x: d.input1, y: d.output, r: Math.abs(d.input2) * 2 + 5 })); // Bubble size by input2
    chart.update();

    if (isTrained) updatePredictionLine();
}

// Function to update predicted line
// Reasoning: Predict across range of input1 (fixed input2=0 for demo), plot line.
function updatePredictionLine() {
    if (!isTrained || data.length === 0) return;

    const minX = Math.min(...data.map(d => d.input1));
    const maxX = Math.max(...data.map(d => d.input1));
    const steps = 20;
    const stepSize = (maxX - minX) / steps;
    const predData = [];

    let promises = [];
    for (let i = 0; i <= steps; i++) {
        const x = minX + i * stepSize;
        promises.push(new Promise((resolve) => {
            neuralNetwork.predict({ input1: x, input2: 0 }, (err, results) => {
                if (!err) resolve({ x, y: results[0].value });
                else resolve(null);
            });
        }));
    }

    Promise.all(promises).then(results => {
        chart.data.datasets[1].data = results.filter(r => r);
        chart.update();
    });
}

// Clear data
clearDataBtn.addEventListener('click', () => {
    data = [];
    updateDataUI();
});

// Export CSV
// Reasoning: Create CSV string from data, download as file.
exportCsvBtn.addEventListener('click', () => {
    if (data.length === 0) return alert('No data to export.');
    const csv = 'input1,input2,output\n' + data.map(d => `${d.input1},${d.input2},${d.output}`).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'data.csv';
    a.click();
});

// Import CSV
// Reasoning: Read file, parse CSV lines, add to data if valid.
importBtn.addEventListener('click', () => {
    const file = importCsvInput.files[0];
    if (!file) return alert('Select a CSV file.');
    const reader = new FileReader();
    reader.onload = (e) => {
        const lines = e.target.result.split('\n').slice(1); // Skip header
        lines.forEach(line => {
            if (line.trim()) {
                const [input1, input2, output] = line.split(',').map(parseFloat);
                if (!isNaN(input1) && !isNaN(input2) && !isNaN(output)) {
                    data.push({ input1, input2, output });
                }
            }
        });
        updateDataUI();
        importCsvInput.value = '';
    };
    reader.readAsText(file);
});

// Event listener for training
// Reasoning: Check data, init with custom options/layers, add data, normalize, train async.
trainBtn.addEventListener('click', () => {
    if (data.length < 4) return alert('Add at least 4 data points.');
    if (isTraining) return alert('Training in progress.');

    trainingStatus.textContent = 'Initializing...';
    isTraining = true;
    isTrained = false;

    // Custom options with named inputs/outputs, custom layers for proper regression
    const options = {
        inputs: ['input1', 'input2'],
        outputs: ['output'],
        task: 'regression',
        learningRate: config.learningRate,
        debug: true, // Set to false if callback issues
        layers: [
            { type: 'dense', units: config.hiddenUnits, activation: 'relu' },
            { type: 'dense', activation: null } // Linear output
        ]
    };

    neuralNetwork = ml5.neuralNetwork(options);

    // Add data
    data.forEach(item => {
        neuralNetwork.addData({ input1: item.input1, input2: item.input2 }, { output: item.output });
    });

    neuralNetwork.normalizeData();

    // Train
    neuralNetwork.train({ epochs: config.epochs }, () => {
        trainingStatus.textContent = 'Training complete!';
        isTrained = true;
        isTraining = false;
        updatePredictionLine();
    });
});

// Save model
// Reasoning: Use ml5 save() to download model files.
saveModelBtn.addEventListener('click', () => {
    if (!isTrained) return alert('Train model first.');
    neuralNetwork.save();
});

// Load model
// Reasoning: Use file input for model files, load with ml5.
loadModelBtn.addEventListener('click', () => {
    const files = loadModelInput.files;
    if (files.length === 0) return alert('Select model files (.json, .model, .bin).');
    neuralNetwork = ml5.neuralNetwork(); // Init empty
    neuralNetwork.load(files, () => {
        trainingStatus.textContent = 'Model loaded!';
        isTrained = true;
        updatePredictionLine();
    });
});

// Event listener for prediction
// Reasoning: Check trained, predict async, display result.
predictForm.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!isTrained) return alert('Train or load model first.');

    const input1 = parseFloat(document.getElementById('pred-input1').value);
    const input2 = parseFloat(document.getElementById('pred-input2').value);

    if (isNaN(input1) || isNaN(input2)) return alert('Valid numbers required.');

    trainingStatus.textContent = 'Predicting...';
    neuralNetwork.predict({ input1, input2 }, (err, results) => {
        if (err) {
            console.error(err);
            predictionResult.textContent = 'Prediction error.';
            return;
        }
        predictionResult.textContent = `Predicted Output: ${results[0].value.toFixed(4)}`;
        trainingStatus.textContent = '';
    });
});
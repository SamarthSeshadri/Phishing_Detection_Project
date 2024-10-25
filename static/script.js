const socket = io();

socket.on('alert', function(data) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert';

    if (data.prediction === 1) {
        alertDiv.classList.add('alert-danger');
        alertDiv.innerHTML = `Real-Time Alert: This email is likely a phishing attempt. Confidence: ${(data.confidence * 100).toFixed(2)}%. <br> ${data.highlighted_text}`;
    } else {
        alertDiv.classList.add('alert-success');
        alertDiv.innerHTML = `Real-Time Alert: This email is not a phishing attempt. Confidence: ${(data.confidence * 100).toFixed(2)}%. <br> ${data.highlighted_text}`;
    }

    document.getElementById('alerts').appendChild(alertDiv);
});

function checkPhishing() {
    const emailText = document.getElementById('emailText').value;
    const resultElement = document.getElementById('result');
    const highlightElement = document.getElementById('highlight');

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: emailText }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.prediction === 1) {
            resultElement.innerHTML = `This email is likely a phishing attempt. Confidence: ${(data.confidence * 100).toFixed(2)}%`;
            resultElement.style.color = 'red';
        } else {
            resultElement.innerHTML = `This email is not a phishing attempt. Confidence: ${(data.confidence * 100).toFixed(2)}%`;
            resultElement.style.color = 'green';
        }
        highlightElement.innerHTML = data.highlighted_text;
    })
    .catch((error) => {
        resultElement.innerHTML = 'Error: ' + error;
        resultElement.style.color = 'red';
    });
}

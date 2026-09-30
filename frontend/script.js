async function checkNews() {
    const text = document.getElementById('news').value.trim();
    const result = document.getElementById('result');
    const analysis = document.getElementById('analysis');

    if (!text) {
        result.textContent = 'Please enter some news text first.';
        result.style.color = '#ffdddd';
        analysis.textContent = '';
        return;
    }

    result.textContent = 'Checking...';
    result.style.color = '#ffffff';
    analysis.textContent = '';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }

        const label = data.prediction;
        const confidence = data.confidence;
        const explanation = data.ai_explanation || '';

        result.textContent = `${label} News (${confidence}% confidence)`;
        result.style.color = label === 'Real' ? '#7ef29a' : '#ffd166';
        analysis.textContent = explanation;
    } catch (error) {
        result.textContent = error.message;
        result.style.color = '#ff8a80';
        analysis.textContent = '';
    }
}

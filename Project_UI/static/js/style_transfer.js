document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const form = document.getElementById('style-form');
    const contentInput = document.getElementById('content-input');
    const styleInput = document.getElementById('style-input');
    const contentPreview = document.getElementById('content-preview');
    const stylePreview = document.getElementById('style-preview');
    const contentText = document.getElementById('content-text');
    const styleText = document.getElementById('style-text');
    const startBtn = document.getElementById('start-btn');
    const resultsContainer = document.getElementById('results-container');
    const statusBox = document.getElementById('status-box');
    const statusMessage = document.getElementById('statusMessage');
    const errorMessage = document.getElementById('errorMessage');
    const resultImage = document.getElementById('result-image');
    const loader = statusBox.querySelector('.loader');

    let pollingInterval;

    // --- Helper function to handle file input and preview ---
    const handleFileSelect = (input, preview, textEl) => {
        const file = input.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = e => {
                preview.src = e.target.result;
                preview.style.display = 'block';
                textEl.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
        checkFormValidity();
    };

    // --- Helper to enable/disable start button ---
    const checkFormValidity = () => {
        startBtn.disabled = !(contentInput.files.length > 0 && styleInput.files.length > 0);
    };

    // --- Polling function to check job status ---
    const pollStatus = async (jobId) => {
        try {
            const response = await fetch(`/api/nst/status/${jobId}`);
            if (!response.ok) {
                throw new Error('Network error while checking status.');
            }
            const data = await response.json();

            if (data.success) {
                statusMessage.textContent = data.status_message || data.status;

                if (data.status === 'completed') {
                    clearInterval(pollingInterval);
                    loader.style.display = 'none';
                    statusMessage.textContent = 'Styling Complete!';
                    resultImage.src = data.result_url;
                    resultImage.style.display = 'block';
                } else if (data.status === 'failed') {
                    clearInterval(pollingInterval);
                    loader.style.display = 'none';
                    statusMessage.textContent = 'Job Failed';
                    errorMessage.textContent = data.status_message;
                    errorMessage.style.display = 'block';
                }
            } else {
                throw new Error(data.error || 'Failed to get job status.');
            }
        } catch (error) {
            clearInterval(pollingInterval);
            loader.style.display = 'none';
            statusMessage.textContent = 'Error';
            errorMessage.textContent = error.message;
            errorMessage.style.display = 'block';
        }
    };

    // --- Event Listeners ---
    contentInput.addEventListener('change', () => handleFileSelect(contentInput, contentPreview, contentText));
    styleInput.addEventListener('change', () => handleFileSelect(styleInput, stylePreview, styleText));

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Reset UI
        resultsContainer.style.display = 'block';
        statusBox.style.display = 'block';
        loader.style.display = 'block';
        resultImage.style.display = 'none';
        errorMessage.style.display = 'none';
        statusMessage.textContent = 'Uploading images...';
        startBtn.disabled = true;

        const formData = new FormData();
        formData.append('content_image', contentInput.files[0]);
        formData.append('style_image', styleInput.files[0]);

        try {
            const response = await fetch('/api/nst/start', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                 const errorData = await response.json().catch(()=>({error: "Server returned an invalid response"}));
                 throw new Error(errorData.error);
            }

            const data = await response.json();

            if (data.success) {
                statusMessage.textContent = 'Processing started... Please wait.';
                // Start polling every 3 seconds
                pollingInterval = setInterval(() => pollStatus(data.job_id), 3000);
            } else {
                 throw new Error(data.error);
            }
        } catch (error) {
            statusMessage.textContent = 'Submission Failed';
            errorMessage.textContent = error.message;
            errorMessage.style.display = 'block';
            loader.style.display = 'none';
            startBtn.disabled = false;
        }
    });
});
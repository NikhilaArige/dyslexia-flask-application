document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('handwritingUpload');
    const previewContainer = document.getElementById('previewContainer');
    const uploadButton = document.getElementById('uploadButton');
    const nextButton = document.getElementById('nextButton');
    const statusText = document.getElementById('statusText');
    const processingSection = document.getElementById('processing');
    const uploadSection = document.getElementById('handwriting-upload-section');
    
    let selectedFile = null;
    
    // Event listeners for drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('highlight');
    }
    
    function unhighlight() {
        dropArea.classList.remove('highlight');
    }
    
    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            selectedFile = files[0];
            handleFiles(files);
        }
    }
    
    // Handle file input change
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            selectedFile = this.files[0];
            handleFiles(this.files);
        }
    });
    
    // Click on drop area to trigger file input
    dropArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Handle file selection
    function handleFiles(files) {
        const file = files[0];
        
        // Validate file type
        if (!file.type.match('image.*')) {
            alert('Please upload an image file (JPG, PNG, JPEG)');
            return;
        }
        
        // Preview image
        const reader = new FileReader();
        reader.onload = function(e) {
            previewContainer.innerHTML = `
                <h3>Preview:</h3>
                <img src="${e.target.result}" class="handwriting-sample img-fluid" alt="Handwriting sample preview">
                <p>${file.name}</p>
                <div class="mt-3">
                    <button type="button" id="change-image" class="btn btn-outline-secondary btn-sm">Change Image</button>
                </div>
            `;
            previewContainer.style.display = 'block';
            uploadButton.disabled = false;
            
            // Add event listener to change image button
            document.getElementById('change-image').addEventListener('click', function() {
                fileInput.value = '';
                previewContainer.style.display = 'none';
                uploadButton.disabled = true;
                fileInput.click();
            });
        };
        reader.readAsDataURL(file);
    }
    
    // Handle upload
    uploadButton.addEventListener('click', async function() {
        if (!selectedFile) {
            statusText.textContent = 'Please select a file first';
            return;
        }
        
        // Show processing status
        statusText.textContent = 'Uploading and analyzing handwriting sample...';
        uploadButton.disabled = true;
        
        // Create form data for file upload
        const formData = new FormData();
        formData.append('handwriting_image', selectedFile);
        
        try {
            // Send to server
            const response = await fetch('/api/upload-handwriting', {
                method: 'POST',
                body: formData
            });
            
            
            const data = await response.json();
            
            if (response.ok) {
                // Show processing animation
                uploadSection.style.display = 'none';
                processingSection.style.display = 'block';
                
                // Simulate processing with progress bar
                let progress = 0;
                const progressBar = document.getElementById('analysis-progress');
                const progressInterval = setInterval(() => {
                    progress += 5;
                    progressBar.style.width = `${progress}%`;
                    progressBar.setAttribute('aria-valuenow', progress);
                    progressBar.textContent = `${progress}%`;
                    
                    if (progress >= 100) {
                        clearInterval(progressInterval);
                        
                        // Display analysis results and show upload section again
                        processingSection.style.display = 'none';
                        uploadSection.style.display = 'block';
                        
                        // Handle successful upload and analysis
                        statusText.textContent = 'Handwriting analysis complete!';
                        
                        // Display analysis results preview
                        previewContainer.innerHTML += `
                            <div class="analysis-preview">
                                <h3>Analysis Preview:</h3>
                                <p>Detected features:</p>
                                <ul>
                                    <li>Letter spacing: ${data.spacing_score}</li>
                                    <li>Letter reversals detected: ${data.reversals}</li>
                                    <li>Line consistency: ${data.line_consistency}</li>
                                </ul>
                            </div>
                        `;
                        
                        // Store results in session storage for the results page
                        sessionStorage.setItem('handwritingResults', JSON.stringify(data));
                        
                        // Enable next button
                        nextButton.disabled = false;
                    }
                }, 100);
            } else {
                // Show error
                statusText.textContent = data.error || 'Error analyzing handwriting. Please try again.';
                uploadButton.disabled = false;
            }
        } catch (error) {
            console.error('Error:', error);
            statusText.textContent = 'Error analyzing handwriting. Please try again.';
            uploadButton.disabled = false;
        }
    });
    
    // Handle next button
    nextButton.addEventListener('click', function() {
        window.location.href = "/audio";
    });
});
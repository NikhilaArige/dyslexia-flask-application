document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const audioPlayback = document.getElementById('audioPlayback');
    const submitButton = document.getElementById('submitButton');
    const statusMessage = document.getElementById('statusMessage');
    const timerDisplay = document.getElementById('timer');
    
    // Variables
    let mediaRecorder;
    let audioChunks = [];
    let timerInterval;
    let recordingTime = 0;
    let audioBlob;
    
    // Initialize audio recording
    async function initializeRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.addEventListener('dataavailable', event => {
                audioChunks.push(event.data);
            });
            
            mediaRecorder.addEventListener('stop', () => {
                audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                audioPlayback.src = audioUrl;
                audioPlayback.style.display = 'block';
                submitButton.disabled = false;
                statusMessage.textContent = 'Recording complete. You can play it back or submit.';
            });
            
            return true;
        } catch (error) {
            console.error('Error accessing microphone:', error);
            statusMessage.textContent = 'Error accessing your microphone. Please check permissions.';
            return false;
        }
    }
    
    // Format time for display (MM:SS)
    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        
        return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
    
    // Start recording
    recordButton.addEventListener('click', async () => {
        if (await initializeRecording()) {
            audioChunks = [];
            recordingTime = 0;
            
            // Update UI
            recordButton.disabled = true;
            stopButton.disabled = false;
            submitButton.disabled = true;
            audioPlayback.style.display = 'none';
            statusMessage.textContent = 'Recording in progress...';
            
            // Start recording
            mediaRecorder.start();
            
            // Start timer
            timerInterval = setInterval(() => {
                recordingTime++;
                timerDisplay.textContent = formatTime(recordingTime);
                
                // Auto-stop after 2 minutes
                if (recordingTime >= 120) {
                    stopRecording();
                }
            }, 1000);
        }
    });
    
    // Stop recording
    stopButton.addEventListener('click', () => {
        stopRecording();
    });
    
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            clearInterval(timerInterval);
            
            // Update UI
            recordButton.disabled = false;
            stopButton.disabled = true;
        }
    }
    
    // Submit recording
    submitButton.addEventListener('click', () => {
        if (!audioBlob) {
            statusMessage.textContent = 'No recording available to submit.';
            return;
        }
        
        // Create form data for file upload
        const formData = new FormData();
        formData.append('audio_recording', audioBlob, 'recording.wav');
        
        // Update UI
        statusMessage.textContent = 'Uploading and analyzing audio...';
        submitButton.disabled = true;
        
        // Send to server
        fetch('/api/upload-audio', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Handle successful upload and analysis
            statusMessage.textContent = 'Audio analysis complete!';
            
            // Store results in session storage for the results page
            sessionStorage.setItem('audioResults', JSON.stringify(data));
            
            // Redirect to results page
            window.location.href = '/results';
        })
        .catch(error => {
            console.error('Error:', error);
            statusMessage.textContent = 'Error analyzing audio. Please try again.';
            submitButton.disabled = false;
        });
    });
});
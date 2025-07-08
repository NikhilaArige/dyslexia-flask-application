document.addEventListener('DOMContentLoaded', () => {
    const eyeTrackerContainer = document.getElementById('eyeTrackerContainer');
    const startButton = document.getElementById('startTracking');
    const statusText = document.getElementById('statusText');
    const progressBar = document.getElementById('progressBar');
    const nextButton = document.getElementById('nextButton');
    
    let calibrationPoints = [];
    let currentPointIndex = 0;
    let trackingInProgress = false;
    let fixations = [];
    let saccades = [];
    
    // Create calibration points (9-point calibration)
    function createCalibrationPoints() {
        const positions = [
            { x: 10, y: 10 }, { x: 50, y: 10 }, { x: 90, y: 10 },
            { x: 10, y: 50 }, { x: 50, y: 50 }, { x: 90, y: 50 },
            { x: 10, y: 90 }, { x: 50, y: 90 }, { x: 90, y: 90 }
        ];
        
        calibrationPoints = positions.map(pos => {
            return {
                x: pos.x,
                y: pos.y,
                fixationTime: 0
            };
        });
    }
    
    // Display a calibration point
    function showCalibrationPoint(index) {
        if (index >= calibrationPoints.length) {
            finishCalibration();
            return;
        }
        
        const point = document.createElement('div');
        point.className = 'calibration-point';
        point.style.left = `${calibrationPoints[index].x}%`;
        point.style.top = `${calibrationPoints[index].y}%`;
        
        // Clear previous points
        eyeTrackerContainer.innerHTML = '';
        eyeTrackerContainer.appendChild(point);
        
        statusText.textContent = `Look at the red dot (${index + 1}/${calibrationPoints.length})`;
        progressBar.style.width = `${((index + 1) / calibrationPoints.length) * 100}%`;
        
        // Move to next point after 2 seconds
        setTimeout(() => {
            // In a real application, you would collect eye tracking data here
            // For simulation, we're just moving to the next point
            showCalibrationPoint(index + 1);
        }, 2000);
    }
    
    // Finish calibration and show reading test
    function finishCalibration() {
        statusText.textContent = 'Calibration complete. Reading test starting...';
        
        setTimeout(() => {
            // Show reading text
            eyeTrackerContainer.innerHTML = `
                <div class="reading-text">
                    <p>The quick brown fox jumps over the lazy dog. A mysterious voice 
                    whispered in the silent room. The ancient oak tree stood tall against 
                    the stormy sky. Children laughed as they played in the park.</p>
                    
                    <p>Scientists discovered a new species in the depths of the ocean. 
                    The old clock on the wall ticked rhythmically. Bright stars twinkled 
                    in the clear night sky. Music flowed through the open windows.</p>
                </div>
            `;
            
            statusText.textContent = 'Please read the text naturally. Your eye movements are being tracked.';
            
            // Simulate eye tracking data collection
            simulateEyeTracking();
        }, 1500);
    }
    
    // Simulate eye tracking data collection
    function simulateEyeTracking() {
        // In a real application, you would be continuously collecting eye movement data
        // For this simulation, we'll just set a timer and then finish
        
        let timeRemaining = 30; // 30 seconds reading time
        
        const timer = setInterval(() => {
            timeRemaining--;
            statusText.textContent = `Reading in progress... ${timeRemaining} seconds remaining`;
            
            if (timeRemaining <= 0) {
                clearInterval(timer);
                finishReading();
            }
        }, 1000);
    }
    
    // Finish reading test and prepare for next step
    function finishReading() {
        // In a real application, you would analyze the collected eye movement data here
        // For simulation, we'll just generate some random data
        
        // Simulate data analysis (normally this would be done by your Python backend)
        const simulatedData = {
            fixations: Math.floor(Math.random() * 50) + 100, // 100-150 fixations
            saccades: Math.floor(Math.random() * 40) + 80,   // 80-120 saccades
            regressions: Math.floor(Math.random() * 20) + 10 // 10-30 regressions
        };
        
        // Store data in session storage for results page
        sessionStorage.setItem('eyeTrackingData', JSON.stringify(simulatedData));
        
        // Show completion message
        eyeTrackerContainer.innerHTML = `
            <div class="completion-message">
                <h2>Eye Tracking Complete!</h2>
                <p>Thank you for completing the eye tracking assessment.</p>
                <p>We've recorded:</p>
                <ul>
                    <li>${simulatedData.fixations} fixations</li>
                    <li>${simulatedData.saccades} saccades</li>
                    <li>${simulatedData.regressions} regressions</li>
                </ul>
            </div>
        `;
        
        statusText.textContent = 'Eye tracking test completed successfully.';
        nextButton.disabled = false;
        
        // Send data to backend
        sendEyeTrackingData(simulatedData);
    }
    
    // Send eye tracking data to backend
    function sendEyeTrackingData(data) {
        fetch('/process_eye_tracking', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            console.log('Success:', result);
            sessionStorage.setItem('eyeTrackingResults', JSON.stringify(result));
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
    
    // Start eye tracking test
    startButton.addEventListener('click', () => {
        if (trackingInProgress) return;
        
        trackingInProgress = true;
        startButton.disabled = true;
        
        // Initialize calibration
        createCalibrationPoints();
        statusText.textContent = 'Calibrating eye tracker...';
        
        // Start calibration after a short delay
        setTimeout(() => {
            showCalibrationPoint(0);
        }, 1000);
    });
});
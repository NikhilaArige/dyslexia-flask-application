document.addEventListener('DOMContentLoaded', () => {
    const downloadButton = document.getElementById('downloadReport');
    
    // Animate result bars
    const resultBars = document.querySelectorAll('.result-fill');
    
    // Animate the bars after a short delay
    setTimeout(() => {
        resultBars.forEach(bar => {
            const width = bar.style.width;
            bar.style.width = '0%';
            
            setTimeout(() => {
                bar.style.width = width;
            }, 100);
        });
    }, 300);
    
    // Download PDF report
    downloadButton.addEventListener('click', (e) => {
        e.preventDefault();
        
        // In a real application, you'd make an AJAX request to generate a PDF
        // For this example, we'll just show an alert
        alert('This feature would generate and download a PDF report of the assessment results.');
        
        // Example of how you'd trigger a download:
        // window.location.href = '/download_report';
    });
});
document.addEventListener("DOMContentLoaded", () => {
    console.log("Crop Water Stress Prediction page loaded.");
    // Add interactive elements if necessary
    
    const viewBtn = document.querySelector('.btn');
    if (viewBtn) {
        viewBtn.addEventListener('click', (e) => {
            console.log("Navigating to GitHub repository...");
        });
    }
});

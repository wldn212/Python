document.addEventListener('DOMContentLoaded', function() {
    // You can add any client-side interactivity here if needed
    console.log("EduSketch AI Virtual Painter is ready!");
    
    // Example: Highlight selected color
    const colorOptions = document.querySelectorAll('.color-option');
    colorOptions.forEach(option => {
        option.addEventListener('click', function() {
            colorOptions.forEach(opt => opt.style.border = '3px solid #fff');
            this.style.border = '3px solid #3498db';
        });
    });
});
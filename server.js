const express = require('express');
const fs = require('fs');
const path = require('path');
const app = express();
const port = 3000;

app.use(cors({ origin: '*' }));

// Handle JSON body parsing
app.use(express.json());

// Serve static files (if needed)
app.use(express.static('public'));

// Endpoint to save canvas image
app.post('/saveImage', (req, res) => {
    const base64Image = req.body.image;
    const imageBuffer = Buffer.from(base64Image.split(',')[1], 'base64');
    const fileName = `canvas_image_${Date.now()}.png`;
    const filePath = path.join(__dirname, 'uploads', fileName); // Adjust as per your server setup

    fs.writeFile(filePath, imageBuffer, (err) => {
        if (err) {
            console.error('Error saving image:', err);
            res.status(500).json({ error: 'Failed to save image' });
        } else {
            console.log('Image saved successfully:', filePath);
            res.json({ filePath: filePath });
        }
    });
});

// Start server
app.listen(port, () => {
    console.log(`Server started at http://localhost:${port}`);
});

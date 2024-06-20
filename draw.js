let annotations = []
const urlParams = new URLSearchParams(window.location.search);
const imagesParam = urlParams.get('images');
const index = parseInt(urlParams.get('index'));
const canvas = document.getElementById('edit-container');
const ctx = canvas.getContext('2d');
let shape_drawn = false;

let text_label = 'Sample Text';
let text_color = 'black';
let shape_color = 'red';
let shape_width = 2;    
let shape_shadow = '';
let shadow_blur = 20;
let shape_type = 'rectangle';
let points = [];
let lines = [];
let fill = false;

document.getElementById('text-label').addEventListener('input', function() {
    text_label = this.value;
});

document.getElementById('text-color').addEventListener('input', function() {
    text_color = this.value;
});

document.getElementById('color').addEventListener('input', function() {
    shape_color = this.value;
});

document.getElementById('width').addEventListener('input', function() {
    shape_width = this.value;
});

document.getElementById('shadow').addEventListener('input', function() {
    shape_shadow = this.value;
});

document.getElementById('shadow-blur').addEventListener('input', function() {
    shadow_blur = this.value;
});

document.getElementById('shape-form').addEventListener('change', function(event) {
    shape_type = document.querySelector('input[name="shape"]:checked').value;
    console.log('Selected shape:', shape_type);
});
document.getElementById('fill').addEventListener('change', function(event) {
    console.log("Checked"); 
    fill = event.target.checked;
});

// Parse the JSON string back into an array
const images = JSON.parse(decodeURIComponent(imagesParam));
console.log(images);

// Display the image in larger format
const editedImage = new Image();
editedImage.src = images[index];

editedImage.onload = function() {
    // Set canvas dimensions
    canvas.width = editedImage.width;
    canvas.height = editedImage.height;

    // Draw the image on the canvas
    ctx.drawImage(editedImage, 0, 0, canvas.width, canvas.height);
};

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mousemove', draw);

let stx, sty;
let drawing = false;

function getMousePos(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;  // Scaling factors
    const scaleY = canvas.height / rect.height;
    return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY
    };
}

function startDrawing(event){
    drawing = true;
    const pos = getMousePos(event);
    stx = pos.x;
    sty = pos.y;

    if (shape_type === 'connected') {
        points.push({ x: stx, y: sty });
    }
    drawPoints();

}

function stopDrawing(event){
    if(drawing){
        const pos = getMousePos(event)
        const endx = pos.x;
        const endy = pos.y;
        currentAnnotation = {
            type: shape_type,
            stx,
            sty,
            endx,
            endy,
            color: shape_color,
            line_width: shape_width,
            shadow_color: shape_shadow,
            shadow_blur: shadow_blur,
            text: text_label,
            text_color: text_color
        };
        annotation_text = currentAnnotation.text;
        drawing = false;
        shape_drawn = true;
    }
}

function draw(event) {
    if (!drawing) return;
    const pos = getMousePos(event)
    const endx = pos.x;
    const endy = pos.y;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(editedImage, 0, 0);
    ctx.strokeStyle = shape_color;
    ctx.lineWidth = shape_width;
    ctx.font = '16px Arial'; 
    ctx.fillStyle = text_color;
    ctx.shadowColor = shape_shadow;
    ctx.shadowBlur = parseFloat(shadow_blur);
    if (shape_type === 'rectangle') {
        ctx.strokeRect(stx, sty, endx - stx, endy - sty);
        ctx.fillText(text_label, stx, sty - 5);
    } else if (shape_type === 'circle') {
        const radius = Math.sqrt(Math.pow(endx - stx, 2) + Math.pow(endy - sty, 2));
        const angle = Math.atan2(endy - sty, endx - stx);

        // Calculate text position on the circle's border
        const textX = stx + radius * Math.cos(angle);
        const textY = sty + radius * Math.sin(angle);

        ctx.beginPath();
        ctx.arc(stx, sty, radius, 0, 2 * Math.PI);
        ctx.stroke();

        if (fill) {
            ctx.fillStyle = shape_color;
            ctx.fill();
        }

        // Draw the text on the circle's border
        ctx.fillText(text_label, textX, textY);
    }
    else if (shape_type === 'connected') {
        points.push({ x: endx, y: endy });
        ctx.beginPath();
        for (let i = 0; i < points.length - 1; i++) {
            ctx.moveTo(points[i].x, points[i].y);
            ctx.lineTo(points[i + 1].x, points[i + 1].y);
        }
        ctx.stroke();
        points.pop(); // Remove the last point to avoid duplication
        drawPoints();
        ctx.fillText(text_label, points[0].x - 20, points[0].y - 10);
    }
    else if(shape_type === 'line') {
        drawLine(stx, sty, endx, endy);
        ctx.fillText(text_label, endx, endy);
    }
    if(fill === 'true')
    {
        ctx.fillStyle = "green";
        ctx.fill();
    }
}

function drawPoints(){
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI); // Adjust the radius (5) as needed
        ctx.fillStyle = 'blue'; // Adjust the color as needed
        ctx.fill();
    });
}

function clearDrawing(){
    if(shape_drawn){
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(editedImage, 0, 0);
        shape_drawn = false;
        currentAnnotation = false;
        let last_idx = annotations.length - 1;
        console.log(last_idx)
        annotations.splice(last_idx, 1)
        console.log("Annotations after clear")
        console.log(annotations);
        points = [];
        lines = [];
    }
}

function saveAnnotations() {
    if (currentAnnotation) {
        annotations.push(currentAnnotation);
        currentAnnotation = null;
        console.log("Annotations after save")
        console.log(annotations);
    }
}

function downloadImage(data, filename = 'untitled.jpeg') {
    var a = document.createElement('a');
    a.href = data;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}


function drawLine(x1, y1, x2, y2) {
    lines.push({x1: x1, y1: y1, x2: x2, y2:y2});
    ctx.beginPath(); // Start a new path
    ctx.moveTo(x1, y1); // Move to the starting point (x1, y1)
    ctx.lineTo(x2, y2); // Draw a line to the ending point (x2, y2)
    ctx.stroke(); // Draw the line
}



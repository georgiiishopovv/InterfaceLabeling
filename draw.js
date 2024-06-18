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
let shape_width = '2';
let shape_shadow = '';
let shadow_blur = '20';
let shape_type = 'rectangle';

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

// document.getElementById('rectangle-btn').addEventListener('click', function() {
//     shape_type = 'rectangle';
// });

// document.getElementById('circle-btn').addEventListener('click', function() {
//     shape_type = 'circle';
// });

// document.getElementById('connect-btn').addEventListener('click', function() {
//     shape_type = 'connect';
// });

// document.getElementById('line-btn').addEventListener('click', function() {
//     shape_type = 'line';
// });

document.getElementById('w').addEventListener('input', function() {
    shape_type = this.value;
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
            density: shape_width,
            shadow: shape_shadow,
            text: text_label
        };
        annotation_text = currentAnnotation.text;
        drawing = false;
        shape_drawn = true;
        console.log("Annotations")
        console.log(annotations)
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
    ctx.fillStyle = 'red';
    ctx.shadowColor = shadow;
    ctx.shadowBlur = shadow_blur;
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

        // Draw the text on the circle's border
        ctx.fillText(text_label, textX, textY);
    }
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


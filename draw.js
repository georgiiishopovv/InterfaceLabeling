let annotations = []
const urlParams = new URLSearchParams(window.location.search);
const imagesParam = urlParams.get('images');
let index = parseInt(urlParams.get('index'));
const canvas = document.getElementById('edit-container');
const ctx = canvas.getContext('2d');
let shape_drawn = false;
let currentAnnotation = '';
let saved = false;
let xmlAnnotation = '';
let all_shapes = [];

let text_label = 'Sample Text';
let text_color = 'black';
let shape_color = 'red';
let text_size = 16;
let shape_width = 2;    
let shape_type = 'rectangle';
let points = [];
let lines = [];
let fill = false;
let save_name = '';

document.getElementById('text-label').addEventListener('input', function() {
    text_label = this.value;
});

document.getElementById('text-color').addEventListener('input', function() {
    text_color = this.value;
});

document.getElementById('text-size').addEventListener('input', function() {
    text_size = this.value;
});

document.getElementById('color').addEventListener('input', function() {
    shape_color = this.value;
});

document.getElementById('width').addEventListener('input', function() {
    shape_width = this.value;
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

upper_switch = document.getElementById('switch-image-text');
index += 1;
upper_switch.innerText = `${index} / ${images.length}`;

let editedImage = new Image();
// Display the image in larger format
function canvasImage(index){
    editedImage = new Image();
    editedImage.src = images[index-1];

    editedImage.onload = function() {
        // Set canvas dimensions
        canvas.width = editedImage.width;
        canvas.height = editedImage.height;

        // Draw the image on the canvas
        ctx.drawImage(editedImage, 0, 0, canvas.width, canvas.height);
    };
}

canvasImage(index);

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
            starting_x: stx,
            starting_y: sty,
            end_x: endx,
            end_y: endy,
            color: shape_color,
            line_width: shape_width,
            text: text_label,
            text_color: text_color,
            text_size: text_size,
            image_src: images[index]
        };
        annotation_text = currentAnnotation.text;
        annotations.push(currentAnnotation);
        currentAnnotation = null;

        // Redraw canvas with updated shapes
        draw();

        drawing = false;
    }
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(editedImage, 0, 0);

    // Draw finalized shapes (annotations)
    annotations.forEach(ann => {
        ctx.strokeStyle = ann.color;
        ctx.lineWidth = ann.line_width;
        ctx.font = `${ann.text_size}px Arial`;
        ctx.fillStyle = ann.text_color;

        if (ann.type === 'rectangle') {
            ctx.strokeRect(ann.starting_x, ann.starting_y, ann.end_x - ann.starting_x, ann.end_y - ann.starting_y);
            ctx.fillText(ann.text, ann.starting_x, ann.starting_y - 5);
        } else if (ann.type === 'circle') {
            const radius = Math.sqrt(Math.pow(ann.end_x - ann.starting_x, 2) + Math.pow(ann.end_y - ann.starting_y, 2));
            const angle = Math.atan2(ann.end_y - ann.starting_y, ann.end_x - ann.starting_x);
            const textX = ann.starting_x + radius * Math.cos(angle);
            const textY = ann.starting_y + radius * Math.sin(angle);

            ctx.beginPath();
            ctx.arc(ann.starting_x, ann.starting_y, radius, 0, 2 * Math.PI);
            ctx.stroke();

            if (fill) {
                ctx.fillStyle = ann.color;
                ctx.fill();
            }

            ctx.fillText(ann.text, textX, textY);
        } else if (ann.type === 'connected') {
            ctx.beginPath();
            for (let i = 0; i < ann.points.length - 1; i++) {
                ctx.moveTo(ann.points[i].x, ann.points[i].y);
                ctx.lineTo(ann.points[i + 1].x, ann.points[i + 1].y);
            }
            ctx.stroke();
            ctx.fillText(ann.text, ann.points[0].x - 20, ann.points[0].y - 10);
        } else if (ann.type === 'line') {
            drawLine(ann.starting_x, ann.starting_y, ann.end_x, ann.end_y);
            ctx.fillText(ann.text, ann.end_x, ann.end_y);
        }

        if (fill) {
            ctx.fillStyle = "green";
            ctx.fill();
        }
        saved = false;
        shape_drawn = true;
    });

    // Draw dynamically drawing shape (currentAnnotation)
    if (drawing && currentAnnotation !== null) {
        ctx.strokeStyle = currentAnnotation.color;
        ctx.lineWidth = currentAnnotation.line_width;
        ctx.font = `${currentAnnotation.text_size}px Arial`;
        ctx.fillStyle = currentAnnotation.text_color;

        if (currentAnnotation.type === 'rectangle') {
            ctx.strokeRect(currentAnnotation.starting_x, currentAnnotation.starting_y, currentAnnotation.end_x - currentAnnotation.starting_x, currentAnnotation.end_y - currentAnnotation.starting_y);
            ctx.fillText(currentAnnotation.text, currentAnnotation.starting_x, currentAnnotation.starting_y - 5);
        } else if (currentAnnotation.type === 'circle') {
            const radius = Math.sqrt(Math.pow(currentAnnotation.end_x - currentAnnotation.starting_x, 2) + Math.pow(currentAnnotation.end_y - currentAnnotation.starting_y, 2));
            const angle = Math.atan2(currentAnnotation.end_y - currentAnnotation.starting_y, currentAnnotation.end_x - currentAnnotation.starting_x);
            const textX = currentAnnotation.starting_x + radius * Math.cos(angle);
            const textY = currentAnnotation.starting_y + radius * Math.sin(angle);

            ctx.beginPath();
            ctx.arc(currentAnnotation.starting_x, currentAnnotation.starting_y, radius, 0, 2 * Math.PI);
            ctx.stroke();

            if (fill) {
                ctx.fillStyle = currentAnnotation.color;
                ctx.fill();
            }

            ctx.fillText(currentAnnotation.text, textX, textY);
        } else if (currentAnnotation.type === 'connected') {
            ctx.beginPath();
            for (let i = 0; i < points.length - 1; i++) {
                ctx.moveTo(points[i].x, points[i].y);
                ctx.lineTo(points[i + 1].x, points[i + 1].y);
            }
            ctx.stroke();
            ctx.fillText(currentAnnotation.text, points[0].x - 20, points[0].y - 10);
        } else if (currentAnnotation.type === 'line') {
            drawLine(currentAnnotation.starting_x, currentAnnotation.starting_y, currentAnnotation.end_x, currentAnnotation.end_y);
            ctx.fillText(currentAnnotation.text, currentAnnotation.end_x, currentAnnotation.end_y);
        }

        if (fill) {
            ctx.fillStyle = "green";
            ctx.fill();
        }
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
    if(shape_drawn && saved){
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

function convertToXML(obj, indent = '   ') {
    let xml = `<annotation>\n`;
    const childIndent = indent; // Increase indentation for child elements
    for (let key in obj) {
        if (obj.hasOwnProperty(key)) {
            xml += `${childIndent}<${key}> ${obj[key]} </${key}>\n`;
        }
    }
    xml += `</annotation>`;
    return xml;
}

function saveAnnotations() {
    if (currentAnnotation != '') {
        alert('Annotations saved successfully.');
        xmlAnnotation = convertToXML(currentAnnotation);
        console.log(xmlAnnotation);
        currentAnnotation = null;
        annotations.push(xmlAnnotation);
        console.log("Annotations after save");
        console.log("Hello");
        console.log(annotations);
        saved = true;

        save_name = `labeled_image${index}`;
        console.log("Saved name", save_name);
        downloadImage(save_name);
        all_shapes.push(annotations);
        annotations = [];
        console.log("Successful");
        console.log(all_shapes)
    }
}

function downloadImage(filename) {
    const dataURL = canvas.toDataURL('image/jpeg');
    const a = document.createElement('a');
    a.href = dataURL;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    //runCommand();
}

function drawLine(x1, y1, x2, y2) {
    lines.push({x1: x1, y1: y1, x2: x2, y2:y2});
    ctx.beginPath(); // Start a new path
    ctx.moveTo(x1, y1); // Move to the starting point (x1, y1)
    ctx.lineTo(x2, y2); // Draw a line to the ending point (x2, y2)
    ctx.stroke(); // Draw the line
}

document.querySelector('.arrow-left-edit').addEventListener('click', function() {
    if (images.length > 0) {
        if(index > 1){
            console.log("Current index", index);
            index -= 1;
            upper_switch.innerText = `${index} / ${images.length}`;
            canvasImage(index);
            //currentImageIndex = (currentImageIndex - 1 + images.length) % images.length;
        }
    }
});

document.querySelector('.arrow-right-edit').addEventListener('click', function() {
    if (images.length > 0) {
        if(index < images.length){
            console.log("Current index", index);
            index += 1;
            upper_switch.innerText = `${index} / ${images.length}`;
            canvasImage(index);
            //currentImageIndex = (currentImageIndex + 1) % images.length;
        }
    }cd
});  

// async function runCommand() {
//     const command = '-ls -l';
//     const response = await fetch(`http://localhost:3000//runcmd?command=${encodeURIComponent(command)}`);
//     if (response.ok) {
//         const result = await response.text();
//         console.log('Command output:', result);
//         alert('Command output:\n' + result);
//     } else {
//         console.error('Failed to execute command:', response.statusText);
//         alert('Failed to execute command:\n' + response.statusText);
//     }
// }




const urlParams = new URLSearchParams(window.location.search);
const imagesParam = urlParams.get('images');
let index = parseInt(urlParams.get('index'));
const canvas = document.getElementById('edit-container');
const ctx = canvas.getContext('2d');
let shape_drawn = false;
let saved = false;
let xmlAnnotation = '';

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

//current annotation is used to store the metadata for each label as multiple can be used per picture
let currentAnnotation = '';
//annotations stores all labels per specific image that is edited - the data is later transformed into xml
let annotations = []
//all shapes contains all labels of the current session of editing images
let all_shapes = [];

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

document.getElementById('go-back-btn').addEventListener('click', function(event) {
    const returnParam = encodeURIComponent(JSON.stringify(images));
    window.location.href = `interface.html?images=${returnParam}`;
    currentAnnotation = '';
    annotations = [];
    all_shapes = [];
});

//Parse the JSON string back into an array
const images = JSON.parse(decodeURIComponent(imagesParam));
console.log(images);

upper_switch = document.getElementById('switch-image-text');
index += 1;
upper_switch.innerText = `${index} / ${images.length}`;

let editedImage = new Image();

function canvasImage(index){
    editedImage = new Image();
    editedImage.src = images[index-1];

    editedImage.onload = function() {
        //Set canvas dimensions
        canvas.width = editedImage.width;
        canvas.height = editedImage.height;

        //Draw the image on the canvas
        ctx.drawImage(editedImage, 0, 0, canvas.width, canvas.height);
    };
}

canvasImage(index);

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mousemove', draw);

let stx, sty;
let drawing = false;

/*

 +Obtain the mouse positions for precise shape drawings

*/
function getMousePos(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width; 
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
        const pos = getMousePos(event);
        const endx = pos.x;
        const endy = pos.y;
        //The data related to the shape is saved in an array when the shape is created
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
            image_src: images[index],
            points: points
        };

        if (shape_type === "connected") {
            currentAnnotation.warning = "points";
        }

        annotations.push(currentAnnotation);

        //Redraw canvas with updated shapes
        draw();

        drawing = false;
    }
}

/*

 +Draw the specific shapes that were selected as labels
 +Maintains all current labels on screen - allows multiples]
 +Different shapes can be combined and drawn
 +The drawing allows to specify different features of the labels such as color, width, text and more, which are all saved in the xml data

*/
function draw(event) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(editedImage, 0, 0);

    // Draw finalized shapes (annotations) if annotations array is not empty
    if (annotations != 0) {
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
                ctx.fillText(ann.text, points[0].x - 20, points[0].y - 10);
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
    }

    //Dynamically drawing shape (currentAnnotation)
    if (drawing && event) {
        const pos = getMousePos(event);
        ctx.strokeStyle = shape_color;
        ctx.lineWidth = shape_width;
        ctx.font = `${text_size}px Arial`;
        ctx.fillStyle = text_color;

        if (shape_type === 'rectangle') {
            ctx.strokeRect(stx, sty, pos.x - stx, pos.y - sty);
            ctx.fillText(text_label, stx, sty - 5);
        } else if (shape_type === 'circle') {
            const radius = Math.sqrt(Math.pow(pos.x - stx, 2) + Math.pow(pos.y - sty, 2));
            const angle = Math.atan2(pos.y - sty, pos.x - stx);
            const textX = stx + radius * Math.cos(angle);
            const textY = sty + radius * Math.sin(angle);

            ctx.beginPath();
            ctx.arc(stx, sty, radius, 0, 2 * Math.PI);
            ctx.stroke();

            if (fill) {
                ctx.fillStyle = shape_color;
                ctx.fill();
            }

            ctx.fillText(text_label, textX, textY);
        } else if (shape_type === 'connected') {
            ctx.beginPath();
            for (let i = 0; i < points.length - 1; i++) {
                ctx.moveTo(points[i].x, points[i].y);
                ctx.lineTo(points[i + 1].x, points[i + 1].y);
            }
            ctx.stroke();
            ctx.fillText(text_label, points[0].x - 20, points[0].y - 10);
        } else if (shape_type === 'line') {
            drawLine(stx, sty, pos.x, pos.y);
            ctx.fillText(text_label, pos.x, pos.y);
        }

        if (fill) {
            ctx.fillStyle = "green";
            ctx.fill();
        }
    }
}

/*

 +Displaying the points for the connected path

*/
function drawPoints(){
    points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI); // Adjust the radius (5) as needed
        ctx.fillStyle = 'blue'; // Adjust the color as needed
        ctx.fill();
    });
}

/*

 +Clear the last label from the image
 +Remove it from the annotations array

*/
function clearLast(){
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(editedImage, 0, 0);
    shape_drawn = false;
    currentAnnotation = null;
    let last_idx = annotations.length - 1;
    console.log(last_idx)
    annotations.splice(last_idx, 1)
    console.log("Annotations after clear")
    console.log(annotations);
    if(shape_type == "connected")
    {
        points.splice(points.length - 1, 1);
    }
    else if(shape_type == "line")
    {
        lines.splice(lines.length - 1, 1);
    }
}

/*

 +Clear the labels on the image all at once
 +Refresh the annotations array

*/
function clearAll(){
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(editedImage, 0, 0);
    shape_drawn = false;
    currentAnnotation = null;
    points = [];
    annotations = [];
    console.log("Annotations after clear", annotations);
}

/*

 +Converting the data of the labels into xml

*/
function convertToXML(obj, indent = '   ') {
    let xml = `<annotation>\n`;
    const childIndent = indent; 
    for (let key in obj) {
        if (obj.hasOwnProperty(key)) {
            xml += `${childIndent}<${key}> ${obj[key]} </${key}>\n`;
        }
    }
    xml += `</annotation>`;
    return xml;
}

/*

 +When the button is pressed, the image gets downloaded locally
 +All current labels and annotations are saved locally in an array
 +This data from the array can be imported into the image's metadata using t_exif.php
 +The save_name and its path need to be consistent as this is used to access the image and edit it's data

*/
function saveAnnotations() {
    if (annotations != '') {
        for(let a in annotations)
        {
            xmlAnnotation = convertToXML(annotations[a]);
            console.log("=================");
            console.log(xmlAnnotation);
            annotations[a] = xmlAnnotation;
            saved = true;
        }
        save_name = "labeled_image003";
        console.log("Saved name", save_name);
        downloadImage(save_name);
        alert("Downloading?");
        console.log('Saved name', save_name);
        all_shapes.push(annotations);

        //Request to php app
        var xhr = new XMLHttpRequest();
        var url = 't_exif.php';
        xhr.open('POST', url, true);
        xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
        xhr.onreadystatechange = function() {
            if (xhr.readyState === 4 && xhr.status === 200) {
                console.log(xhr.responseText);
            }
        };

        //Send the data to the php in an appropriate format
        //Specify the name of the image for which the metadata has to be updated (the name that was used for saving the image)
        save_name += '.jpg'
        var data = 'input_image=' + save_name +
               '&output_image=' + save_name +
               '&description=' + annotations;

        console.log("Iinput", data);
        xhr.send(data);

        annotations = [];
        currentAnnotation = null;
        console.log("Successful");
        console.log(all_shapes);
        alert('Annotations saved successfully.');
    }
}

/*

 +Download the edited canvas image
 +The new image gets saved into the downloads folder locally

*/
function downloadImage(filename) {
    const dataURL = canvas.toDataURL('image/jpeg');
    const a = document.createElement('a');
    a.href = dataURL;
    localStorage.setItem("recent-image", a);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}


/*

 +Drawing a straight line and adding text to it

*/
function drawLine(x1, y1, x2, y2) {
    lines.push({x1: x1, y1: y1, x2: x2, y2:y2});
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}

/*

 +Left and Right arrows for moving between the images

*/
document.querySelector('.arrow-left-edit').addEventListener('click', function() {
    if (images.length > 0) {
        if(index > 1){
            console.log("Current index", index);
            index -= 1;
            upper_switch.innerText = `${index} / ${images.length}`;
            annotations = []; // Clear annotations
            //draw(); // Redraw canvas without annotations
            
            canvasImage(index);
        }
    }
});

document.querySelector('.arrow-right-edit').addEventListener('click', function() {
    if (images.length > 0) {
        if(index < images.length){
            console.log("Current index", index);
            index += 1;
            upper_switch.innerText = `${index} / ${images.length}`;
            annotations = []; // Clear annotations
            //draw(); // Redraw canvas without annotations
            
            canvasImage(index);
        }
    }
});  

/*

 +Send the canvas image to a node.js server so that it can be saved there
 +The path from the server can be used while editing the image's metadata

*/
const base64Image = canvas.toDataURL('image/png');

function saveCanvasImage() {
    const url = 'http://localhost:3000/saveImage';
    const data = { image: base64Image };

    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        console.log('Image saved successfully:', data.filePath);
    })
    .catch(error => {
        console.error('Error saving image:', error);
    });
}



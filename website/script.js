const fileInput = document.getElementById('imageLoader');
fileInput.addEventListener('change', handleImage, false);

const canvas = document.getElementById('image-preview');
const ctx = canvas.getContext('2d');
const form = document.querySelector('form');

const canvasOutput = document.getElementById('image-output');
const ctxOutput = canvasOutput.getContext('2d');

const rectifyButton = document.getElementById('submit'); 
const imagePreview = document.getElementById('image-preview'); 
const imageRectified = document.getElementById('image-output'); 
// const restartButton = document.getElementsByClassName('restart'); 

function handleImage() {
    
    console.log(fileInput.files[0])
    if (fileInput.files.length !== 0) {
        document.querySelector(".drop-image").style.display = 'none';
        document.getElementById("restart").style.display = 'none'; 

        // rectifyButton.style.visibility = 'visible';
        rectifyButton.style.display = 'block'; 
        imagePreview.style.display='revert'; 
    } else {
        console.log('nothing') 
    }


    var reader = new FileReader();
    reader.onload = function (event) {
        var img = new Image();
        img.onload = function () {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        }
        img.src = event.target.result;

        let count = 1;
        canvas.onclick = function (event) {
            //clientX and clientY for mapping on canvas
            const rect = canvas.getBoundingClientRect();
            let scaled = canvas.width / rect.width;
            const x = Math.floor((event.clientX - rect.left) * scaled);
            const y = Math.floor((event.clientY - rect.top) * scaled);
            // console.log(x,y);
            if (count > 4) {
                return
            }
            document.querySelector('[name=x' + count + ']').value = x;
            document.querySelector('[name=y' + count + ']').value = y;
            count++;

            ctx.fillStyle = '#ADFF00';

            ctx.beginPath();
            ctx.arc(x, y, 4 * scaled, 0, 2 * Math.PI);
            ctx.fill();
        };
        // document.getElementById('load').style.display = 'revert'; 
        

    }
    reader.readAsDataURL(fileInput.files[0]);
}

form.onsubmit = function (event) {
    event.preventDefault();
    // document.getElementById('load').style.display = 'revert'; 

    const formData = new FormData(form);

    fetch('/rectify', {
        method: 'POST',
        body: formData,
    }).then(response => {
        return response.blob();
    }).then(imageBlob => {
        var img = new Image();
        img.onload = function () {
            canvasOutput.width = img.width;
            canvasOutput.height = img.height;
            ctxOutput.drawImage(img, 0, 0);
        }
        img.src = URL.createObjectURL(imageBlob);
        // rectifyButton.style.display='none';
        imageRectified.style.display='revert';
        document.getElementById("restart").style.display = 'revert';
        stopSpinner(); 
    });
}

function startSpinner() {
    document.getElementById("load").style.display = "block";
    rectifyButton.style.display = 'none';
}

function stopSpinner() {
    document.getElementById("load").style.display = "none";
} 





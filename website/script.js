const fileInput = document.getElementById('imageLoader');
fileInput.addEventListener('change', handleImage, false);

const canvas = document.getElementById('image-preview');
const ctx = canvas.getContext('2d');
const form = document.querySelector('form');

const canvasOutput = document.getElementById('image-output');
const ctxOutput = canvasOutput.getContext('2d');

function handleImage() {
    console.log(fileInput.files[0])
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

            ctx.fillStyle = 'red';

            ctx.beginPath();
            ctx.arc(x, y, 4 * scaled, 0, 2 * Math.PI);
            ctx.fill();
        };
    }
    reader.readAsDataURL(fileInput.files[0]);
}

form.onsubmit = function (event) {
    event.preventDefault();

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
    });
}





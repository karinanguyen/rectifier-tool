const inputBox = document.querySelector('input');
document.querySelector('button').onclick = () => {
    fetch('/rectify', {
        method: 'POST',
        body: `{ "name": "${inputBox.value}" }`,
    })
        .then(resp => resp.text())
        .then(response_text => {
            document.querySelector('.results').textContent = response_text;
        })
}
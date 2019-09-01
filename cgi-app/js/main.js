// alert("Hello");

// ファイルAPIに対応しているか確認
if (window.File && window.FileReader && window.FileList && window.Blob) {
    // Great success! All the File APIs are supported.
  } else {
    alert('The File APIs are not fully supported in this browser.');
}

function onFileSelected(input) {
    var file = input.files[0];
    var reader = new FileReader();
    reader.onload = onFileLoaded;
    reader.readAsDataURL(file);
    console.log('selected');
}

function onFileLoaded(e) {
    var src_data = e.target.result;
    var img = new Image();
    img.onload = onImageSetted;
    img.src = src_data;
    console.log('loaded');
}

function onImageSetted(e) {
    var data = createImageData(e.target);
    document.getElementById('test_canvas').getContext('2d').putImageData(data, 0, 0);
    console.log('setted');
}

function createImageData(img) {
    var cv = document.createElement('canvas');
    cv.width = img.naturalWidth;
    cv.height = img.naturalHeight;
    var ct = cv.getContext('2d');
    ct.drawImage(img, 0, 0);
    var data = ct.getImageData(0, 0, cv.width, cv.height);
    console.log('created');
    return data;
}
    

function resetCanvas(){
    var cv = document.createElement('canvas');
    var ct = cv.getContext('2d');
    //natural で指定したい
    ct.clearRect(0, 0, 500, 500);
    console.log('reset');
}

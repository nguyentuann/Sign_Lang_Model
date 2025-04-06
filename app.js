let model;
let webcamElement = document.getElementById('webcam');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let labelElement = document.getElementById('label');
const gestureNames = ['E', 'L', 'F', 'V', 'B'];

async function setupWebcam() {
  return new Promise((resolve, reject) => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        webcamElement.srcObject = stream;
        webcamElement.addEventListener('loadeddata', () => resolve(), false);
      }).catch(err => reject(err));
  });
}

async function predictLoop() {
  canvas.width = 224;
  canvas.height = 224;

  while (true) {
    ctx.drawImage(webcamElement, 0, 0, 224, 224);
    let imageData = ctx.getImageData(0, 0, 224, 224);

    let imgTensor = tf.browser.fromPixels(imageData)
      .expandDims(0)
      .toFloat()
      .div(255.0); // Normalize

    let prediction = await model.predict(imgTensor).data();
    let maxProb = Math.max(...prediction);
    let classIndex = prediction.indexOf(maxProb);

    if (maxProb * 100 >= 95) {
      labelElement.textContent = `${gestureNames[classIndex]} (${(maxProb * 100).toFixed(2)}%)`;
    } else {
      labelElement.textContent = `None`;
    }

    await tf.nextFrame();
  }
}

async function main() {
  model = await tf.loadLayersModel('tfjs_model/model.json');
  await setupWebcam();
  predictLoop();
}

main();

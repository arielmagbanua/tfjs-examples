import "./styles.css";
import * as tf from "@tensorflow/tfjs";

const blankScreenCanvas = document.getElementById("bsod");
const blankScreen = tf.fill([250, 350, 1], 0.2);
tf.browser.toPixels(blankScreen, blankScreenCanvas);

// CHALLENGE
// make a function getImage that takes the number of channels as an argument
// and returns a tensor image with that many channels of random pixel values
// that is 250 pixels tall and 350 pixels wide
// EXAMPLES:
// getImage(1) should return a black and white static image
// getImage(3) should return a random color static image

// For the first exercise
// function getImage(channels) {
//   return tf.randomUniform([250, 350, channels], 0, 1);
// }

function withBackImage(imgData, channels) {
  const backImageTensor = tf.browser.fromPixels(imgData, channels);
  return tf
    .randomUniform([250, 350, channels], 0, 255, "int32")
    .add(backImageTensor)
    .div(512);
}

// END OF YOUR CODE
// Feel free to look at the rest to see how this works,
// but the tensor image generation is done by you!

// Here's a bunch of variables and references to DOM elements to make things work!
let imgIndex = 0;
let canvasArr;
let tensorArr;
let tvPower = true;
const screen = document.getElementById("screen");
const rgbButton = document.getElementById("rgb");
const bwButton = document.getElementById("bw");
const powerButton = document.getElementById("power");
let onLoaded = false;

// const imgML = document.getElementById("ml");

// This function creates 20 canvas elements based on your tensor function and
// loads them to the page with display = 'none'
function loadImages(channels) {
  screen.innerHTML = "";
  canvasArr = [];
  tensorArr = [];

  // create image in memory
  const img = new Image();
  // allow image to cross domains
  img.crossOrigin = "anonymous";

  // set the dimenstions
  img.height = 250;
  img.width = 350;

  // point it at the image we want
  img.src =
    "https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTGF8nuu42y8FiI65iWYqBkR-YrFemz9TKD_w&usqp=CAU";

  img.onload = (event) => {
    for (let i = 0; i < 20; i++) {
      const canvas = document.createElement("canvas");
      screen.appendChild(canvas);
      canvasArr.push(canvas);
      canvas.style.display = "none";

      // Your getImage function at work!
      // const staticImg = getImage(channels);
      const staticImg = withBackImage(event.target, channels);

      tensorArr.push(staticImg);
      tf.browser.toPixels(staticImg, canvas);
    }

    if (!onLoaded) {
      animatetvStatic();
      onLoaded = !onLoaded;
    }
  };
}

// this function animates the static by looping over the array of canvas elements
// and toggling their display to be visible one at a time if the TV is "on"
function animatetvStatic() {
  requestAnimationFrame(() => {
    if (canvasArr[imgIndex]) {
      canvasArr[imgIndex].style.display = "none";
      imgIndex++;
      if (imgIndex >= canvasArr.length) imgIndex = 0;
      if (tvPower) {
        canvasArr[imgIndex].style.display = "block";
      }
    }
    animatetvStatic();
  });
}

// this changes the array of canvas elements to be all black and white static
bwButton.onclick = function () {
  // prevent memory leaks!
  tensorArr.forEach((t) => t.dispose());
  // load images with 1 channel, aka black and white
  loadImages(1);
};

// this changes the array of canvas elements to be all color static
rgbButton.onclick = function () {
  // prevent memory leaks!
  tensorArr.forEach((t) => t.dispose());
  // load images with 3 channel, aka full color static!
  loadImages(3);
};

// toggles a boolean that our animation function uses to decide
// whether or not to display canvas elements
powerButton.onclick = function () {
  tvPower = !tvPower;
  powerButton.textContent = tvPower ? "Off" : "On";
};

// kicks things off with color static!
loadImages(3);

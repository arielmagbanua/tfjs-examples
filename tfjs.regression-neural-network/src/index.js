import P5 from "p5";
import * as tf from "@tensorflow/tfjs";

let x_vals = [];
let y_vals = [];

let dragging = false;

// create the model
const model = tf.sequential();

// create the first hidden layer
// dense layer is a "fully connected layer"
const hidden1 = tf.layers.dense({
  inputShape: [1], // input shape
  units: 4, // number of nodes
  activation: "relu" // activation function
});
// add the layer
model.add(hidden1);

// create the second hidden layer
// dense layer is a "fully connected layer"
const hidden2 = tf.layers.dense({
  units: 3, // number of nodes
  activation: "relu" // activation function
});
// add the layer
model.add(hidden2);

// create another layer
// the input shape is inferred from the previous layer
const output = tf.layers.dense({
  units: 1, // number of nodes
  activation: "linear" // activation function
});
// add the layer
model.add(output);

// compile the model
model.compile({
  optimizer: tf.train.sgd(0.1), // optimizer is a gradient descent
  loss: "meanSquaredError"
});

const tensorCount = document.getElementById("tensor-count");
const lossContainer = document.getElementById("loss");

new P5((p) => {
  p.setup = () => {
    p.createCanvas(600, 600);

    document.getElementById("beginTraining").addEventListener("click", () => {
      setTimeout(train, 10);
    });
  };

  p.mousePressed = () => {
    dragging = true;
  };

  p.mouseReleased = () => {
    dragging = false;
  };

  // Draw for every click at the canvass which.
  // Each data point will be added to training set and then
  // TensorFlowJs will minimize the loss / cost function.
  p.draw = () => {
    if (dragging) {
      p.frameRate(30);
      let x = p.map(p.mouseX, 0, p.width, -1, 1);
      let y = p.map(p.mouseY, 0, p.height, 1, -1);

      // add the points to the training data set
      x_vals.push(x);
      y_vals.push(y);
    }

    p.frameRate(30);
    p.strokeWeight(6);
    p.background(0);

    p.stroke(255);

    for (let i = 0; i < x_vals.length; i++) {
      let px = p.map(x_vals[i], -1, 1, 0, p.width);
      let py = p.map(y_vals[i], -1, 1, p.height, 0);
      p.point(px, py);
    }

    const curveX = [];
    for (let x = -1; x <= 1; x += 0.05) {
      curveX.push(x);
    }
    const curveXs = tf.tensor1d(curveX);

    // equivalent predicted values of x
    const curveYs = model.predict(curveXs);
    const curveY = curveYs.dataSync();
    // curveYs.print();
    curveYs.dispose();
    curveXs.dispose();

    p.beginShape();
    p.noFill();
    p.stroke(255, 0, 0);
    p.strokeWeight(2);
    for (let i = 0; i < curveX.length; i++) {
      let x = p.map(curveX[i], -1, 1, 0, p.width);
      let y = p.map(curveY[i], -1, 1, p.height, 0);
      p.vertex(x, y);
    }
    p.endShape();

    tensorCount.innerHTML = `Tensors: ${tf.memory().numTensors}`;
  };

  function train() {
    const xs = tf.tensor1d(x_vals);
    const ys = tf.tensor1d(y_vals);

    trainModel(xs, ys).then((result) => {
      // console.log(result.history.loss[0]);
      // recursively train the model by itself
      const loss = result.history.loss;
      if (loss) {
        lossContainer.innerHTML = `Loss: ${loss[0]}`;
      }

      setTimeout(train, 10);
    });

    xs.dispose();
    ys.dispose();
  }

  function trainModel(xs, ys) {
    return model.fit(xs, ys, {
      shuffle: true,
      epochs: 120
    });
  }
});

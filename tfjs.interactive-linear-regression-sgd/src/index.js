import P5 from "p5";
import * as tf from "@tensorflow/tfjs-core";

let x_vals = [];
let y_vals = [];

// The m and b of y = mx + b
let m, b;

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

const tensorCount = document.getElementById("tensor-count");

new P5((p) => {
  p.setup = () => {
    p.createCanvas(400, 400);

    m = tf.variable(tf.scalar(p.random(1)));
    b = tf.variable(tf.scalar(p.random(1)));
  };

  function predict(x) {
    const xs = tf.tensor1d(x);

    // y = mx + b;
    const ys = xs.mul(m).add(b);

    return ys;
  }

  function loss(predictions, labels) {
    return predictions.sub(labels).square().mean();
  }

  p.mousePressed = () => {
    let x = p.map(p.mouseX, 0, p.width, 0, 1);
    let y = p.map(p.mouseY, 0, p.height, 1, 0);

    x_vals.push(x);
    y_vals.push(y);
  };

  // Draw for every click at the canvass which.
  // Each data point will be added to training set and then
  // TensorFlowJs will minimize the loss / cost function.
  p.draw = () => {
    tf.tidy(() => {
      if (x_vals.length > 0) {
        const ys = tf.tensor1d(y_vals);
        optimizer.minimize(() => loss(predict(x_vals), ys));
      }
    });

    p.background(0);

    p.stroke(255);
    p.strokeWeight(8);

    for (let i = 0; i < x_vals.length; i++) {
      let px = p.map(x_vals[i], 0, 1, 0, p.width);
      let py = p.map(y_vals[i], 0, 1, p.height, 0);
      p.point(px, py);
    }

    tf.tidy(() => {});

    // x values
    let lineX = [0, 1];
    // equivalent predicted values of x
    const ys = tf.tidy(() => predict(lineX));

    // x values for canvas drawing
    let x1 = p.map(lineX[0], 0, 1, 0, p.width);
    let x2 = p.map(lineX[1], 0, 1, 0, p.width);

    // predicted y values for canvass drawing.
    let lineY = ys.dataSync();
    let y1 = p.map(lineY[0], 0, 1, p.height, 0);
    let y2 = p.map(lineY[1], 0, 1, p.height, 0);

    p.stroke(255, 0, 0);
    p.strokeWeight(1);

    // plot the line / cost or loss function
    p.line(x1, y1, x2, y2);
    ys.dispose();

    tensorCount.innerHTML = `Tensors: ${tf.memory().numTensors}`;
  };
});

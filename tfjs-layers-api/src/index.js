import P5 from "p5";
import * as tf from "@tensorflow/tfjs-core";

let x_vals = [];
let y_vals = [];

// The m and b of y = mx + b
let a, b, c, d;
let dragging = false;

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

const tensorCount = document.getElementById("tensor-count");

new P5((p) => {
  p.setup = () => {
    p.createCanvas(600, 600);

    a = tf.variable(tf.scalar(p.random(-1, 1)));
    b = tf.variable(tf.scalar(p.random(-1, 1)));
    c = tf.variable(tf.scalar(p.random(-1, 1)));
    d = tf.variable(tf.scalar(p.random(-1, 1)));
  };

  function predict(x) {
    const xs = tf.tensor1d(x);

    // y = ax^3 + bx2 + cx + d
    // const ys = xs.square().mul(a).add(xs.mul(b)).add(c);
    const ys = xs
      .pow(tf.scalar(3))
      .mul(a)
      .add(xs.square().mul(b))
      .add(xs.mul(c))
      .add(d);
    return ys;
  }

  function loss(predictions, labels) {
    return predictions.sub(labels).square().mean();
  }

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
      p.frameRate(10);

      let x = p.map(p.mouseX, 0, p.width, -1, 1);
      let y = p.map(p.mouseY, 0, p.height, 1, -1);

      // add the points to the training data set
      x_vals.push(x);
      y_vals.push(y);
    } else {
      // train only if data points are already added
      tf.tidy(() => {
        if (x_vals.length > 0) {
          const ys = tf.tensor1d(y_vals);
          optimizer.minimize(() => loss(predict(x_vals), ys));
        }
      });
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

    // x values
    const curveX = [];

    for (let x = -1; x <= 1.01; x += 0.05) {
      curveX.push(x);
    }

    // equivalent predicted values of x
    const ys = tf.tidy(() => predict(curveX));
    let curveY = ys.dataSync();
    ys.dispose();

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
});

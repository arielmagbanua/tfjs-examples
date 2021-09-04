import "./styles.css";
import * as tf from "@tensorflow/tfjs/";

const xs = tf.tensor2d([
  [0, 0, 0],
  [0, 0, 1],
  [0, 1, 0],
  [0, 1, 1],
  [1, 0, 0],
  [1, 0, 1],
  [1, 1, 0],
  [1, 1, 1]
]);

const ys = tf.tensor2d([
  [0, 0],
  [1, 0],
  [0, 0],
  [1, 0],
  [1, 0],
  [0, 1],
  [1, 0],
  [0, 1]
]);

// create the model
const model = tf.sequential();

// create the first hidden layer
// dense layer is a "fully connected layer"
const hidden1 = tf.layers.dense({
  inputShape: [3], // input shape
  units: 5, // number of nodes
  activation: "sigmoid" // activation function
});
// add the layer
model.add(hidden1);

// create the second hidden layer
// dense layer is a "fully connected layer"
const hidden2 = tf.layers.dense({
  units: 4, // number of nodes
  activation: "sigmoid" // activation function
});
model.add(hidden2);

const output = tf.layers.dense({
  units: 2,
  activation: "sigmoid"
});
// add the layer
model.add(output);

// compile the model
model.compile({
  optimizer: tf.train.sgd(1), // optimizer is a gradient descent
  loss: "meanSquaredError"
});

async function train(xs, ys, callbacks) {
  console.log("training started");
  const result = await model.fit(xs, ys, {
    shuffle: true,
    epochs: 150,
    callbacks: callbacks
  });
  console.log(result);

  model.save("downloads://test-model");
}

(() => {
  const loss = document.getElementById("loss");

  const trainingCallbacks = {
    // called when training starts
    // onTrainBegin: (log) => console.log(log),
    // called when training ends
    // onTrainEnd: (log) => console.log(log),
    // called on start of each pass of the training data
    // onEpochBegin: (epoch, log) => console.log(epoch, log),
    // called on each successful pass on training data
    // * This is one of my favorite to actually use!
    onEpochEnd: (epoch, log) => {
      loss.innerHTML = `Loss: ${log.loss}`;
    }
    // Runs before each batch - perfect for large batch training
    // onBatchBegin: (batch, log) => console.log(batch, log)
    // runs after each batch - 32 in this case
    // onBatchEnd: (batch, log) => console.log(batch, log)
  };

  document.getElementById("train-model").addEventListener("click", () => {
    train(xs, ys, trainingCallbacks).then(() => {
      console.log("training finished");
      // dispose the tensors
      tf.dispose(xs, ys);
    });
  });

  document.getElementById("classify-form").addEventListener("submit", (e) => {
    e.preventDefault();

    // grab the inputs
    const looksLikeADuck = document.getElementById("looks").checked ? 1 : 0;
    const swimsLikeADuck = document.getElementById("swims").checked ? 1 : 0;
    const quacksLikeADuck = document.getElementById("quacks").checked ? 1 : 0;
    console.log(looksLikeADuck, swimsLikeADuck, quacksLikeADuck);

    const inputXs = tf.tensor2d([
      [looksLikeADuck, swimsLikeADuck, quacksLikeADuck]
    ]);

    // predict
    let output = model.predict(inputXs);
    output = output.dataSync();
    const probably = Math.round(output[0]);
    const definitely = Math.round(output[1]);

    console.log(probably, definitely);

    alert(`
      Probably: ${probably === 1 ? "Yes" : "No"}
      Definitely: ${definitely === 1 ? "Yes" : "No"}
    `);

    inputXs.dispose();
  });
})();

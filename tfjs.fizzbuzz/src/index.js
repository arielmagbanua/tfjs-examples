import "./styles.css";
import * as tf from "@tensorflow/tfjs/";

const doFizzBuzz = async (binNumArray) => {
  const modelPath =
    "https://s3.amazonaws.com/ir_public/ai/fizzbuzz/fizzbuzz-model.json";

  // load the mmodel
  const fizzBuzzModel = await tf.loadLayersModel(modelPath);

  // create tensor
  const fizzBuzzNumber = tf.tensor([binNumArray]);

  // prediction result
  const result = await fizzBuzzModel.predict(fizzBuzzNumber);

  // get the data of the result
  const resultData = await result.data();

  const outputPrediction = resultData.indexOf(Math.max(...resultData));

  const resultContainer = document.getElementById("prediction");

  switch (outputPrediction) {
    case 1:
      //fizz
      resultContainer.innerHTML = "Fizz";
      break;
    case 2:
      //buzz
      resultContainer.innerHTML = "Buzz";
      break;
    case 3:
      // fizzbuzz
      resultContainer.innerHTML = "FizzBuzz";
      break;
    default:
      // just a number
      resultContainer.innerHTML = parseInt(binNumArray.join(""), 2);
      break;
  }

  fizzBuzzModel.dispose();
  fizzBuzzNumber.dispose();
  result.dispose();
};

function dec2bin(dec, inputShape = 10) {
  return (dec >>> 0)
    .toString(2)
    .padStart(inputShape, "0")
    .split("")
    .map(Number);
}

(() => {
  const header = document.getElementById("tensors");
  header.innerHTML = `Tensors: ${tf.memory().numTensors}`;

  const checkButton = document.getElementById("checkButton");
  checkButton.addEventListener("click", () => {
    // grab the input
    const inputNumber = document.getElementById("num").value;

    if (!inputNumber) {
      return alert("Please input a number!");
    }

    const numBinTensor = dec2bin(inputNumber);
    console.log(numBinTensor);
    // execute fizz buzz
    doFizzBuzz(numBinTensor);
  });
})();

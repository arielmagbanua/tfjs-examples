import "./styles.css";
import * as tf from "@tensorflow/tfjs/";
import * as nsfwjs from "nsfwjs";

(() => {
  document.querySelector(".tsfjs-version").innerHTML = `${tf.version.tfjs}`;
  const imageInput = document.getElementById("image");
  const imageFilePreview = document.getElementById("preview");

  imageInput.onchange = function (event) {
    const imageFile = event.target.files[0];

    // read the file
    const fileReader = new FileReader();
    fileReader.readAsDataURL(imageFile);

    fileReader.onload = async function (readerEvent) {
      imageFilePreview.src = readerEvent.target.result;

      const model = await nsfwjs.load();
      let predictions = null;

      const resultsContainer = document.getElementById("results");
      resultsContainer.innerHTML = "";

      if (imageFile.type === "image/gif") {
        // returns top 1 prediction of each GIF frame, and logs the status to console
        const classifyGifConfig = {
          topk: 1,
          setGifControl: (gifControl) => console.log(gifControl),
          onFrame: ({ index, totalFrames, predictions }) =>
            console.log(index, totalFrames, predictions)
        };

        predictions = await model.classifyGif(
          imageFilePreview,
          classifyGifConfig
        );
      } else {
        predictions = await model.classify(imageFilePreview, 3);
      }
      console.log(predictions);

      predictions.forEach((prediction) => {
        const paragraph = document.createElement("p");

        const textNode = document.createTextNode(
          `${prediction.className}: ${(prediction.probability * 100).toFixed(
            2
          )} %`
        );

        paragraph.appendChild(textNode);
        resultsContainer.appendChild(paragraph);
      });
    };
  };
})();

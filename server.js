const express = require("express");
const app = express();
const cors = require("cors");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const sharp = require("sharp");

let model;
const modelPath = "model.js/model.json";

async function loadModel() {
  try {
    model = await tf.loadLayersModel(`file://${modelPath}`);
    console.log("Model loaded");
  } catch (error) {
    console.error("Error loading the model:", error);
  }
}
loadModel();
app.use(cors());
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads");
  },
  filename: function (req, file, cb) {
    cb(null, file.originalname);
  },  
});
const upload = multer({ storage: storage });

app.get("/", function (req, res) {
  console.log("got a GET request for the home page");
  res.send("Welcome to Home page");
});
// app.post('/uploads',function(req,res){
//     console.log()
//     res.send("recived an image")
// })
app.post("/uploads", upload.single("image"), async (req, res) => {
  // console.log(req)
  console.log("post req recived");
  console.log(req.file);
//   res.send("Received an image");
  try {
  console.log(req.file);
  console.log(req.file.path);
    const processedImage = await sharp(req.file.path)
      .resize({ width: 128, height: 128 })
      .toBuffer();
      console.log('Image processed successfully.');
    //   res.json({ message: 'Image processed successfully.' });
    const inputTensor = tf.node.decodeImage(processedImage);
    const expandedTensor = inputTensor.expandDims();
    const normalizedTensor = expandedTensor.div(255.0);
    const reshapedTensor = normalizedTensor.reshape([1, 128, 128, 3]);
    const predictions = model.predict(reshapedTensor);
    const label = predictions.dataSync()[0] > 0.5 ? 'normal' : 'cracked';
    // console.log({ label, confidence: predictions.dataSync()[0] * 100 });
    res.send({ label, confidence: predictions.dataSync()[0] * 100 });


  } catch (error) { 
    console.error("Error processing image:", error);
    res.status(500).json({ error: "Error processing image" });
  }
});

async function startServer() {
  await loadModel();
  const server = app.listen(8000, () => {
    console.log("Server is listening on port 8000");
  });
}

startServer();

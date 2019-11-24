const toUint8Array = require('base64-to-uint8array');
exports.handler = async (event, context) => {
  const tf = require('@tensorflow/tfjs-node');
  const mobilenet = require('@tensorflow-models/mobilenet');
  try {
    // https://towardsdatascience.com/image-object-detection-with-tensorflow-js-b8861119ed46
    let img = event.body;
    console.log(img.slice(0, 20));
    img = img
      .replace('data:image/jpeg;base64', '')
      .replace('data:image/png;base64', '');
    img = toUint8Array(img);
    const tensor3d = tf.node.decodeJpeg(img, 3);

    const model = await mobilenet.load({
      version: 1,
      alpha: 0.25 | 0.5 | 0.75 | 1.0
    });
    const predictions = await model.classify(tensor3d);
    return {
      statusCode: 200,
      body: JSON.stringify({ predictions })
    };
  } catch (err) {
    console.error(err);
    return { statusCode: 500, body: JSON.stringify({ err: err.toString() }) };
  }
};

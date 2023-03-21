const MODEL_PATH = 
'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json';

let model = undefined;
console.log('Check!!!')

async function loadModel() {
    model = await tf.loadLayersModel(MODEL_PATH);
    model.summary();
    await model.save('localstorage://demo/newModelName');
    console.log(JSON.stringify(await tf.io.listModels()));
  
    // Create a batch of 1.
    const input = tf.tensor2d([[870]]);
    
    // Create a batch of 3
    const inputBatch = tf.tensor2d([[500], [1100], [970]]);
    
    // Actually make the prediction for each batch
    const result = model.predict(input);
    const resultBatch = model.predict(inputBatch);
    
    // Print results to console.
    result.print(); // or use .array() to get result back as array
    resultBatch.print(); // or use .array() to get result back as array
    
    input.dispose();
    inputBatch.dispose();
    result.dispose();
    resultBatch.dispose();
    model.dispose();
}

loadModel();
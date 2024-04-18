async function entrenarModelo() {
  const model = tf.sequential(); //para crear un modelo secuencial
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));//se añade una capa densa al modelo

  // Compilación del modelo antes de entrenarlo
  model.compile({             //"meanSquaredError": error cuatra
    loss: "meanSquaredError",//"loos" funcion de perdida:
    optimizer: "sgd",
  });

  // Datos de entrenamiento
  const xs = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
  const ys = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10], [9, 1]);

  // Entrenamiento del modelo
  await model.fit(xs, ys, { epochs: 500 });

  // Actualización del mensaje de información
  document.getElementById("mensaje").innerText =
    "Modelo entrenado. Listo para predecir.";

  // Guardar el modelo entrenado para que esté disponible para la predicción
  window.trainedModel = model;
}

async function predecir() {
  // Obtener el valor de entrada del usuario
  const inputValue = parseInt(document.getElementById("valor").value);

  // Verificar si el modelo está entrenado
  if (!window.trainedModel) {
    alert("Por favor, entrena el modelo antes de hacer una predicción.");
    return;
  }

  // Predecir el valor de Y para el valor de X ingresado por el usuario
  const model = window.trainedModel;
  const prediction = model.predict(tf.tensor2d([inputValue], [1, 1]));

  // Mostrar la predicción en la página
  const outputField = document.getElementById("campo_salida");
  outputField.innerText = `Predicción: ${prediction.dataSync()[0]}`;
}
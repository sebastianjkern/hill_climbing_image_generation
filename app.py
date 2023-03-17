import keras.applications
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import skia
from PIL import Image

import hcga

WIDTH, HEIGHT = 224, 224

model = keras.applications.MobileNetV3Large(weights="imagenet")

chromosome_bit_length = 8

n_objects = 100
objects_size = 3


def map0_1(value):
    return float(value) / ((2 ** chromosome_bit_length) - 1)


def map0_255(value):
    return int((float(value) / ((2 ** chromosome_bit_length) - 1)) * 255)


generated_image = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
surface = skia.Surface(generated_image)
canvas = surface.getCanvas()

paint = skia.Paint()

paint.setAntiAlias(True)
paint.setColor(skia.ColorBLACK)
paint.setStyle(skia.Paint.kFill_Style)
paint.setStrokeWidth(4)

canvas.clear(skia.ColorWHITE)


def render_image(individual):
    canvas.clear(skia.ColorWHITE)

    for i in range(int(len(individual) / objects_size)):
        # paint.setColor(skia.Color(map0_255(individual[objects_size * i + 3]),
        #                           map0_255(individual[objects_size * i + 4]),
        #                           map0_255(individual[objects_size * i + 5]),
        #                           map0_255(individual[objects_size * i + 6])))

        paint.setColor(skia.Color(0, 0, 0, 150))

        canvas.drawCircle(map0_1(individual[objects_size * i + 0]) * 224,
                          map0_1(individual[objects_size * i + 1]) * 224,
                          map0_1(individual[objects_size * i + 2]) * 20, paint)


# Training related stuff
LABEL = "liner"


def fitness(individual):
    render_image(individual)
    image = np.expand_dims(generated_image[:, :, :3], axis=0)
    image = keras.applications.vgg16.preprocess_input(image)

    predictions = model.predict(image, verbose=0)
    decoded_predictions = keras.applications.mobilenet_v3.decode_predictions(predictions, top=1000)

    return list(filter(lambda x: x[1] == LABEL, decoded_predictions[0]))[0][2]


algo = hcga.HillClimbingAlgorithm(fitness, 0.025, 0.75, objects_size, generations=10000,
                                  chromosome_size=chromosome_bit_length, maximize=True)

algo.run()

for ix, sample in enumerate(algo.samples):
    render_image(sample)

    im = Image.fromarray(generated_image)
    im.save(f"output/{str(ix).zfill(5)}.png")

plt.plot(list(range(len(algo.fitness))), algo.fitness)
plt.show()

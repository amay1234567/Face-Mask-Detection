import numpy as np
import cv2
import os

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------ TRAIN ------------------
TRAIN = False   # 👉 set True only when training

if TRAIN:
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(150,150,3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 🔥 strong augmentation (fix angle issue)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        zoom_range=0.3,
        shear_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.7,1.3]
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(150,150),
        batch_size=16,
        class_mode='binary'
    )

    test_set = test_datagen.flow_from_directory(
        'test',
        target_size=(150,150),
        batch_size=16,
        class_mode='binary'
    )

    print("Class mapping:", training_set.class_indices)

    model.fit(training_set, epochs=15, validation_data=test_set)

    model.save("mymodel.h5")


# ------------------ LOAD MODEL ------------------
if not os.path.exists("mymodel.h5"):
    print("Model not found! Set TRAIN = True first.")
    exit()

mymodel = load_model("mymodel.h5")


# ------------------ DNN FACE DETECTOR ------------------
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# ------------------ WEBCAM ------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    (h, w) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # 🔥 padding (captures mask area better)
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            face_img = img[y1:y2, x1:x2]

            if face_img.size == 0:
                continue

            face_img = cv2.resize(face_img, (150,150))
            face_img = face_img / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            pred = float(mymodel(face_img, training=False)[0][0])

            # 🔥 balanced threshold (stable)
            if pred > 0.5:
                label = "NO MASK"
                color = (0,0,255)
            else:
                label = "MASK"
                color = (0,255,0)

            cv2.rectangle(img, (x1,y1), (x2,y2), color, 3)
            cv2.putText(img, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Mask Detector (MobileNet)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
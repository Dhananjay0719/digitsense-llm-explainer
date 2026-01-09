import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from llm_explainer import explain_prediction

st.set_page_config(page_title="Handwritten Digit Recognizer", layout="centered")

# st.title("✍️ DigitSense - Handwritten Digit Recognizer")
st.markdown(
    """
    <h1 style="
        white-space: nowrap;
        font-size: 3rem;
        margin-bottom: 0.2em;
    ">
        ✍️ DigitSense – Handwritten Digit Recognizer
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("Upload an image or draw a digit (0–9)")

# Load trained model
# model = tf.keras.models.load_model("digits.keras")
def load_digit_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.load_weights("digits.weights.h5")
    return model


# Load trained model (SAFE WAY)
model = load_digit_model()

# Mode selection
mode = st.radio("Choose input method:", ["Draw Digit","Upload Image"],index=0)


# Preprocess
def preprocess(img):
    img = np.array(img)

    # Binary threshold
    img = img > 50

    # Get bounding box
    coords = np.column_stack(np.where(img))
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(0)
    y_max, x_max = coords.max(0)

    img = img[y_min:y_max + 1, x_min:x_max + 1]

    # Resize while keeping aspect ratio (MNIST style)
    h, w = img.shape
    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))

    img = Image.fromarray(img.astype("uint8") * 255)
    img = img.resize((new_w, new_h))

    # Place on 28x28 canvas
    canvas = Image.new("L", (28, 28), 0)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas.paste(img, (x_offset, y_offset))

    img_array = np.array(canvas) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    return img_array


# Upload Image Mode
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload handwritten digit", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, width=150)

        img_array = preprocess(image)
        if img_array is not None:
            prediction = model.predict(img_array)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)

            st.success(f"Predicted Digit: **{digit}**")
            st.write(f"Confidence: `{confidence:.2f}`")

            with st.spinner("AI is explaining the prediction..."):
                explanation = explain_prediction(digit, confidence)

            st.markdown("### 🤖 AI Explanation")
            st.write(explanation)


# Draw Mode
else:
    st.write("Draw a digit below 👇")

    canvas = st_canvas(
        fill_color="black",
        stroke_width=6,           # thinner strokes = better MNIST match
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas.image_data is not None:
        img = Image.fromarray(canvas.image_data.astype("uint8")).convert("L")

        if st.button("Predict"):
            img_array = preprocess(img)

            if img_array is not None:
                prediction = model.predict(img_array)
                digit = np.argmax(prediction)
                confidence = np.max(prediction)

                st.success(f"Predicted Digit: **{digit}**")
                st.write(f"Confidence: `{confidence:.2f}`")

                with st.spinner("AI is explaining the prediction..."):
                    explanation = explain_prediction(digit, confidence)

                st.markdown("### 🤖 AI Explanation")
                st.write(explanation)
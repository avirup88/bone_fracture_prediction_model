import streamlit as st
import tensorflow as tf
from PIL import Image

# Cache the model loading to optimize performance
@st.cache_resource
def load_model_file():
    """
    Load the pre-trained bone fracture detection model.

    Returns:
        model (tf.keras.Model): Loaded TensorFlow model.
    """
    try:
        model = tf.keras.models.load_model('./model_files/bone_fracture_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def validate_and_prepare_image(image_file):
    """
    Validate and preprocess the uploaded image file for model prediction.

    Args:
        image_file: Uploaded image file.

    Returns:
        tuple: Preprocessed image array and the original image.
    """
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    file_name = image_file.name

    if not file_name.lower().endswith(supported_formats):
        raise ValueError("Unsupported file format. Please upload an image in JPEG, PNG, BMP, or GIF format.")

    try:
        # Load the image and preprocess it
        image = Image.open(image_file)
        image = image.convert("RGB")  # Convert to RGB format
        image = image.resize((224, 224))  # Resize to the required input size
        image_array = tf.keras.preprocessing.image.img_to_array(image)  # Convert to array
        image_array = tf.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
        return image_array, image
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

def predict_fracture():
    """
    Streamlit application for bone fracture detection.
    """
    # Title and introduction
    st.title("Bone Fracture Detection")
    st.write("This app predicts whether an X-ray image shows a bone fracture. Upload a single image to get started!")

    # Sidebar for instructions
    st.sidebar.header("Instructions")
    st.sidebar.write("1. Upload a single image (JPEG, PNG, BMP, or GIF).")
    st.sidebar.write("2. Select the actual label from the dropdown.")
    st.sidebar.write("3. Click **Submit** to start the process.")
    st.sidebar.write("4. Review the comparison of the actual and predicted labels.")

    # File uploader for single image
    uploaded_file = st.file_uploader("Upload an X-ray image:", type=["jpg", "jpeg", "png", "bmp", "gif"])

    # Dropdown for selecting the actual label
    class_names = ['fractured', 'not fractured']  # Class names must match the model's output order
    actual_label = st.selectbox("Choose your diagnosis", class_names)

    # Submit and Reset buttons with side-by-side layout
    col1, col2 = st.columns(2)
    with col1:
        submit_button = st.button("Submit", use_container_width=True)
    with col2:
        reset_button = st.button("Reset", use_container_width=True)

    # Reset the app if the Reset button is clicked
    if reset_button:
        st.rerun()

    # Handle the Submit button click
    if submit_button:
        if uploaded_file is not None:
            try:
                # Step 1: Validate and preprocess the uploaded image
                st.write("🔍 **Validating and preprocessing the image...**")
                image_array, uploaded_image = validate_and_prepare_image(uploaded_file)

                # Step 2: Load the pre-trained model
                st.write("🚀 **Loading the pre-trained model...**")
                model = load_model_file()
                if not model:
                    return

                # Step 3: Generate prediction using the model
                st.write("🧠 **Generating prediction...**")
                prediction = model.predict(image_array).flatten()  # Flatten the prediction for compatibility
                
                # Determine the predicted label (0 for fractured, 1 for not fractured)
                predicted_label = int(tf.where(prediction < 0.5, 0, 1)[0])

                # Step 4: Style results based on correctness
                predicted_color = "#d4edda" if actual_label == class_names[predicted_label] else "#f8d7da"
                actual_color = "#d4edda" if actual_label == class_names[predicted_label] else "#f8d7da"

                # Step 5: Display results in a styled card
                st.markdown(
                    f"""
                    <div style="background-color:#ffffff; color:#000000; padding:20px; border-radius:10px; max-width: 600px; margin: auto; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
                        <p style="font-size:18px; line-height:2; background-color:{predicted_color}; border-radius:5px; padding:10px;">
                            <b>🤖 Machine Prediction:</b> {class_names[predicted_label]}
                        </p>
                        <p style="font-size:18px; line-height:2; background-color:{actual_color}; border-radius:5px; padding:10px;">
                            <b>👨‍⚕️ Human Diagnosis:</b> {actual_label}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # Step 6: Display the uploaded image
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)  # Add spacing
                st.image(uploaded_image, caption="Uploaded X-ray Image", use_column_width=True)

                # Step 7: Success message
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)  # Add spacing
                st.success("🎉 Prediction and comparison completed successfully!")

            except ValueError as e:
                st.error(e)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please upload an image before clicking Submit.")

# Run the app
if __name__ == "__main__":
    predict_fracture()

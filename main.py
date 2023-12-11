import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow model prediction
def model_prediction(test_image):
    model=tf.keras.models.load_model("trained_model6.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr2=np.array([input_arr])
    predictions = model.predict(input_arr2)
    return np.argmax(predictions)
#Sidebar
st.sidebar.title("DashBoard")
app_mode=st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

#MainPage
if(app_mode=="Home"):
    
    st.snow()
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path="home_img.jpg"
    st.image(image_path)

#AboutPage
elif(app_mode=="About Project"):
    st.header("ABOUT PROJECT")
    st.subheader("ABOUT DATASET")
    st.text("This dataset contains images of the following food items:")
    st.subheader("Fruits")
    st.code("banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.subheader("Vegetables")
    st.code("cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("CONTEXT")
    st.text("This dataset contains three folders:")
    st.code("1.train (100 images each)")
    st.code("2.test (10 images each)")
    st.code("3.validation (10 images each)")
    st.text("each of the above folders contains subfolders for different fruits and vegetables")
    st.text("wherein the images for respective food items are present.")

 #Predictions
elif(app_mode=="Prediction"):
    st.header("Model Predictions")
    test_image=st.file_uploader("Choose an image:")
    if(st.button("Show Image")):
        st.image(test_image,width=2,use_column_width=True)  
  # Set background image
    
    #predict Image
    if(st.button("predict")):
        st.balloons()
        st.write("Our Prediction")
        result_index=model_prediction(test_image)
        #Reading Lables
        with open("labels.txt") as f:
            content= f.readlines()
        label = []
      
        for i in content:
            label.append(i[:-1])
        st.success("Model is predicting It's a {}".format(label[result_index]))


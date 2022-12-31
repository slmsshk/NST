# Import for UI
import UI
import streamlit as st

# Import for Image manupulation
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
import time,os

# PLotting libraries
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
import functools




# ++++++++++++++++++++++++++++++++++++++
# Clean Ram
# ==========================================

reset=st.button('click to reset and start fresh')

if reset:
    # os.remove('style.png')
    try:
        os.remove("st.jpg")
        UI.write('done1 st',tag='p',padding=1,bg='green')
        os.remove("im.jpg")
        UI.write('done2 im',tag='p',padding=1,bg='green')
        os.remove("final.jpg")
        UI.write('done',tag='p',padding=1,bg='green')
    except:
        UI.write(':)',tag='p',padding=1,bg='yellow')
        
refresh=st.button('refresh')

if refresh:
    st.experimental_rerun()


# ++++++++++++++++++++++++++++++++++++++
# Page setup
# ==========================================


UI.add_bg_from_local('Style Background.jpg')

UI.write("NST-Neural Styles Transfer",tag='h1',bg='maroon',fontsize=30)


# ++++++++++++++++++++++++++++++++++++++
# Taking User Upload
# ======================================

col1,col2=st.columns(2)

with col1:
    
    UI.write('Upload your photo',bg='green',fontsize=20,tag='h2')
    image_file = st.file_uploader(" ",type=['jpg'])

    if image_file is not None:

        col1.write(image_file.name)
        # Open St format to Image format
        img = Image.open(image_file)
        col1.image(img) #Display the image
        cv2.imwrite(img=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR),filename='im.jpg') #Save the file
        # cv2.imwrite(img=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR),filename='i'+image_file.name)
        

with col2:
    
    UI.write('Upload Style photo',bg='green',fontsize=20,tag='h2')
    style_file = st.file_uploader("Close the file after Upload",type=['jpg'],key='style')

    if style_file is not None:

        col2.write(style_file.name)
        # Open St format to Image format
        sty = Image.open(style_file)
        col2.image(sty) #Display the image
        cv2.imwrite(img=cv2.cvtColor(np.array(sty),cv2.COLOR_RGB2BGR),filename='st.jpg') #Save the file


UI.write('Neural Style transfer image',tag='h1',fontsize=35,bg='orange',color='white')




but=st.button('press for Style transfer')


if but:

# Load compressed models from tensorflow_hub
    os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

    def tensor_to_image(tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)



    content_path ='im.jpg' #r"C:\Users\Slmss\OneDrive\Pictures\Camera Roll\IMG_20221001_144029.jpg"#tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    style_path ='st.jpg'# tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


    def load_img(path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img


    # In[6]:


    def imshow(image, title=None):
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)

        plt.imshow(image)
        if title:
            plt.title(title)


    # In[7]:


    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # plt.subplot(1, 2, 1)
    # imshow(content_image, 'Content Image')

    # plt.subplot(1, 2, 2)
    # imshow(style_image, 'Style Image')


    # In[8]:


    import tensorflow_hub as hub
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    i=tensor_to_image(stylized_image)

    cv2.imwrite(img=cv2.cvtColor(np.array(i),cv2.COLOR_RGB2BGR),filename='final.jpg')





try:
    with open("final.jpg", "rb") as file:
        down = st.download_button(
                label="Download image",
                data=file,
                file_name="style.png",
                mime="image/png"
            )
    st.image('final.jpg')

except:
    UI.write('Happy styling',bg='Orange',padding='45px')

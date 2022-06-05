import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import save_img
import tensorflow_text as text
import shutil
import numpy as np
import os
import json
import sys
import random
from bs4 import BeautifulSoup
import re
from functools import partial
data_dir = "D:\Arafath\Major\streamlit-hatefulmemedection"
SLANG_PATH = "static/slang.txt"
import webbrowser # inbuilt module


image_shape = (256,256,3)
trainable = False
max_seq_length = 128
units = 512
embedding_dim = 768
BATCH_SIZE = 50
BUFFER_SIZE = 200
attention_features_shape = 64


#==================================== Model ==================================
##================================Preloading function=======================================
@st.cache()
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase



# opening slang map
with open(SLANG_PATH) as file:
    slang_map = dict(map(str.strip, line.partition('\t')[::2])
    for line in file if line.strip())

slang_words = sorted(slang_map, key=len, reverse=True)
regex = re.compile(r"\b({})\b".format("|".join(map(re.escape, slang_words))))
replaceSlang = partial(regex.sub, lambda m: slang_map[m.group(1)])


def text_preprocessing(final):
  preprocessed_text = []
  for sentance in final:
      sentance = BeautifulSoup(sentance, 'lxml').get_text()
      sentance = replaceSlang(sentance)
      sentance = decontracted(sentance)
      sentance = re.sub("\S*\d\S*", "", sentance).strip()
      sentance = re.sub('[^A-Za-z]+', ' ', sentance)
      preprocessed_text.append(sentance.strip())
  return preprocessed_text



@st.cache(allow_output_mutation=True)
def load_model():
  image_model = tf.keras.applications.ResNet152V2(include_top=False,
                                                weights='imagenet')
  new_input = image_model.input
  hidden_layer = image_model.layers[-1].output
  image_features_extract_model = tf.keras.Model(new_input, hidden_layer, name = "image_feature_extractor")


  tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

  tfhub_handle_encoder = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1"

  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing_text_layer')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)

  text_features_extract_model= tf.keras.Model([text_input], [outputs['sequence_output'],outputs['pooled_output']], name = 'text_feature_extractor')

  class CNN_Encoder(tf.keras.Model):
      def __init__(self, embedding_dim):
          super(CNN_Encoder, self).__init__()
          self.fc = tf.keras.layers.Dense(units = embedding_dim,kernel_initializer='glorot_uniform',use_bias=False)

      def call(self, x):
          x = self.fc(x)
          x = tf.nn.relu(x)
          return x


  class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
      super(BahdanauAttention, self).__init__()
      self.W1 = tf.keras.layers.Dense(units)
      self.W2 = tf.keras.layers.Dense(units)
      self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
      hidden_with_time_axis = tf.expand_dims(hidden, 1)

      attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                          self.W2(hidden_with_time_axis)))

      score = self.V(attention_hidden_layer)
      attention_weights = tf.nn.softmax(score, axis=1)
      context_vector = attention_weights * features
      context_vector = tf.reduce_sum(context_vector, axis=1)

      return context_vector, attention_weights

  class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units,class_size = 2):
      super(RNN_Decoder, self).__init__()
      self.units = units
      self.gru = tf.keras.layers.GRU(self.units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
      
      self.fc1 = tf.keras.layers.Dense(self.units)

      self.fc2 = tf.keras.layers.Dense(64, activation='relu', name='Dense_3_layer')

      self.dropout  = tf.keras.layers.Dropout(0.3)

      self.fc3  = tf.keras.layers.Dense(class_size, activation='softmax', name='classifier')

      self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
      context_vector, attention_weights = self.attention(features, hidden)
      x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
      output, state = self.gru(x)
      x = self.fc1(output)
      x = tf.reshape(x, (-1, x.shape[2]))
      x = self.fc2(x)
      x = self.dropout(x)
      x = self.fc3(x)

      return x, state, attention_weights

    def reset_state(self, batch_size):
      return tf.zeros((batch_size, self.units))


  encoder = CNN_Encoder(embedding_dim)
  decoder = RNN_Decoder(embedding_dim, units,class_size = 2)
  optimizer = tf.keras.optimizers.Adam()

  checkpoint_path = "checkpoints/train"
  ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer = optimizer)

  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)

  start_epoch = 0
  if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

  return image_features_extract_model, text_features_extract_model, encoder, decoder


@st.cache(allow_output_mutation=True)
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    img = tf.keras.applications.resnet_v2.preprocess_input(img)
    return img


class_values = ["non-hateful","hateful"]
@st.cache(allow_output_mutation=True)
def evaluate(image,text):
    image_features_extract_model, text_features_extract_model, encoder, decoder = load_model()
    attention_plot = np.zeros((1, attention_features_shape))
    hidden = decoder.reset_state(batch_size=1)
    img_tensor = tf.expand_dims(load_image(image), 0)
    img_tensor = image_features_extract_model(img_tensor)
    img_tensor = tf.reshape(img_tensor,
             (img_tensor.shape[0], -1, img_tensor.shape[3]))
    text = tf.convert_to_tensor(text)
    text = tf.expand_dims(text, 0)
    text_features_seq,pooled_features = text_features_extract_model(tf.convert_to_tensor(text))
    dec_input = tf.expand_dims(pooled_features, axis = 1)
    features = encoder(img_tensor)
    predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
    
    predicted_id = np.argmax(predictions, axis = 1).tolist()
    return predictions,class_values[predicted_id[0]]


#=================================== Title ===============================
st.title("""
Offensive Meme Detectioin
	""")




#========================== File Selector ===================================
st.write("""
#### Select from the Drop Down test memes
""")

DEFAULT = '< PICK A VALUE >'
option = st.selectbox('Select a Meme',
                      [x for x in os.listdir(os.path.join("./images")) if '.png' in x ])


img_file_buffer_option = "./images/"+option
txt_file_buffer_option = "./images/"+option.split('.')[0]+'.txt'

st.write("""**Or Upload Image with Text**""")

img_file_buffer = st.file_uploader("Upload an image(.png preffered) here 👇🏻")
txt_file_buffer = st.file_uploader("Upload a text file(.txt preffered) here 👇🏻")



#========================== File Uploader ===================================



text = ""
if option and img_file_buffer is None:
  img_file_buffer = img_file_buffer_option
  txt_file_buffer = txt_file_buffer_option
if txt_file_buffer:
  for line in txt_file_buffer:
    text = line

if len(text) > 2 :

  st.write("""Selected Text inside the image""")
  st.subheader(text)


try:
	image = Image.open(img_file_buffer)

	st.write("""Preview Of Selected Image!""")

	if image is not None:st.image(image,use_column_width=False, width=400)

except:
	st.write("""
		### ❗ Any Picture hasn't selected yet!!!
		""")


#================================= Predict Button ============================
st.text("""""")
submit = st.button("Predict")


##=================prediction================================================#
def generate_result(prediction):
  st.write("""## 🎯 RESULT""")
  st.write(prediction)
  if prediction == "hateful":st.write("""## Model predicts it as a Hateful Meme!!!""")
  else:
    st.write("""## Model predicts it as a Non-Hateful Meme!!!""")

#=========================== Predict Button Clicked ==========================
if submit:

  save_img("test_image.png", np.array(image))

  image_path = "test_image.png"
  # Predicting
  st.write("👁️ Predicting...")
  pred, predicted_class = evaluate(image_path,text)
  st.write(pred)
  generate_result(predicted_class)



#================================= Title Image ===========================
# st.text("""""")
# img_path_list = ["static/image_1.jpg",
# 				"static/image_2.jpg"]
# index = random.choice([0,1])
# image = Image.open(img_path_list[index])
# st.image(
# 	        image,
# 	        use_column_width=True,
# 	    )



#============================ Behind The Scene ==========================
st.write("""
## Behind The Scene
	""")
st.write("""
To see how it works, please click the button below!
	""")
st.text("""""")
github = st.button("👉🏼 Click Here To See How It Works")
if github:
	github_link = "https://github.com/yasararafathali/hatefulmemesdetection"
	try:
		webbrowser.open(github_link)
	except:
		st.write("""
    		⭕ Something Went Wrong!!! Please Try Again Later!!!
    		""")

#======================== Time To See The Magic ===========================
  
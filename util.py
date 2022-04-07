import re
import json
import numpy as np
import nltk
from keras_preprocessing.text import tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import onnxruntime as rt


nltk.download('stopwords')
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


def tweet_preprocessor(text):
  # remove urls
  tweet1 = re.sub(r'http\S+', ' ', text)
  # remove html tags
  tweet2 = re.sub(r'<.*?>', ' ', tweet1)
  # remove digits
  tweet3 = re.sub(r'\d+', ' ', tweet2)
  # remove hashtags
  tweet4 = re.sub(r'#\w+', ' ', tweet3)
  review = re.sub('[^a-zA-Z]', ' ', tweet4)
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review if word not in all_stopwords]
  review = ' '.join(review)
  return [review]


def predict_sentiment_onnx(text):
  classes = ('Extremely Negative', 'Extremely Positive', 'Negative', 'Neutral', 'Positive')
  tokenizer = None
  with open('./tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
  processed_text = tweet_preprocessor(text)
  transformed_text = tokenizer.texts_to_sequences(processed_text)  # tokenizer our text
  transformed_text = pad_sequences(transformed_text, 28)
  ort_session = rt.InferenceSession("./corona_NLP_sentiment_model.onnx")
  ort_inputs = {ort_session.get_inputs()[0].name: transformed_text.astype(np.float32)}
  prediction = ort_session.run(None, ort_inputs)
  true_id = np.asarray(prediction[0]).argmax(axis=1)
  result = {
      "prediction":  {
          "class": classes[true_id[0]],
          "prob": str(max(prediction[0][0]))
      },
      "probabilities": {},
      "text": text
  }
  for i, pred in enumerate(list(prediction[0][0])):
    result["probabilities"][classes[i]] = str(pred)
  result["probabilities"] = dict(sorted(result["probabilities"].items(), key=lambda x: x[1]))
  return result

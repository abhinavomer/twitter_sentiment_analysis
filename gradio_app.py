import pickle
import gradio as gr

# Load the pre-trained model and vectorizer using context managers
with open('vectorizer_twitter.pickle', 'rb') as file:
    vectorizer = pickle.load(file)
with open('model_twitter.pickle', 'rb') as file:
    model = pickle.load(file)

def predict_sentiment(tweet):
    # Transform the tweet using the vectorizer
    transformed_tweet = vectorizer.transform([tweet])

    # Predict the sentiment using the model
    prediction = model.predict(transformed_tweet)[0]

    # Return the predicted sentiment
    return "Negative" if prediction == 0 else "Positive"


# Launch the Gradio interface
gr.Interface(fn=predict_sentiment, inputs='textbox', outputs='textbox', title="Twitter Sentiment Analysis").launch()

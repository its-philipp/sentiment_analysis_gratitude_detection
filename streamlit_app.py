import streamlit as st
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
import json
import requests
import time
import pandas as pd
import re
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from keras.models import load_model

st.set_page_config(layout="wide")

## Removing all special characters in comments
def process_content(content):
    return " ".join(re.findall("[A-Za-z]+",content))

# Converting links to html tags
def path_to_image_html(path):
    return '<img src="' + path + '" width="200" >'


data = pd.read_csv("training_dataset.csv")
X = data.drop(columns=["video_id"])
y = X.pop("label")
X_train_list = list(X["comment"])

# Load the model
model = load_model('cnn-lstm_model.h5')

# Call the YouTube API
api_key = 'AIzaSyDA0-qOAgaltH0DhTLa9Y1IdbIHfUlAPm0' # Enter your own API key – this one won't work

youtube_api = build('youtube', 'v3', developerKey = api_key)

# function for getting the comments
def get_comments(youtube, video_id, token):
    """
    Recursive function that retrieves the comments (top-level ones) a given video has.
    """

    global all_comments
    totalReplyCount = 0
    token_reply = None

    if (len(token.strip()) == 0):
        all_comments = []
    try:
        if (token == ''):
            video_response=youtube.commentThreads().list(part='snippet',maxResults=100,videoId=video_id,order='relevance').execute()
        else:
            video_response=youtube.commentThreads().list(part='snippet',maxResults=100,videoId=video_id,order='relevance',pageToken=token).execute()

        # Loop comments from the video:
        for indx, item in enumerate(video_response['items']):
            # Append coments:
            all_comments.append("COMMENT WITH " + str(item['snippet']['totalReplyCount']) + " replies: " + item['snippet']['topLevelComment']['snippet']['textDisplay'])

            # Get total reply count:
            totalReplyCount = item['snippet']['totalReplyCount']

            # If the comment has replies, get them:
            if (totalReplyCount > 0):
                # Get replies - first batch:
                replies_response=youtube.comments().list(part='snippet',maxResults=100,parentId=item['id']).execute()
                for indx, reply in enumerate(replies_response['items']):
                    # Append the replies to the main array:
                    all_comments.append((" "*2) + "=>FIRST CALLBACK REPLY: " + reply['snippet']['textDisplay'])

                # If the reply has a token for get more replies, loop those replies
                # and add those replies to the main array:
                while "nextPageToken" in replies_response:
                    token_reply = replies_response['nextPageToken']
                    replies_response=youtube.comments().list(part='snippet',maxResults=100,parentId=item['id'],pageToken=token_reply).execute()
                    for indx, reply in enumerate(replies_response['items']):
                        all_comments.append((" "*4) + "==>WHILE GETTING REPLIES: " + reply['snippet']['textDisplay'])

        # Check if the video_response has more comments:
        if "nextPageToken" in video_response:
            return get_comments(youtube, video_id, video_response['nextPageToken'])
        else:
            # Remove empty elements added to the list "due to the return in both functions":
            all_comments = [x for x in all_comments if len(x) > 0]
            print("Fin")
            return []
    except Exception as e:
        # Handle the exception when comments are turned off/deactivated for a video
        print(f"Exception: {e}")
        print("Comments are turned off for this video. Searching for comments in the next video.")
        return all_comments

st.header("Gratitude Maximzer")
st.markdown(""" # 🎬 Welcome to the Gratitude Maximizer app where you get video recommendations which will make you feel more grateful!""")

# Call the YouTube API
search_terms = st.text_input('Please enter a search word:')

results = youtube_api.search().list(q=search_terms, part='snippet', type='video',
                                    order='viewCount', maxResults=10).execute()

# creating dataframe out of search word
video_ids = []
thumbnails = []
date = []
titles = []
descriptions = []
channel_titles = []

if search_terms:
    for item in results['items']:
        video_ids.append(item['id']['videoId'])
        thumbnails.append(item['snippet']['thumbnails']['high']['url'])
        titles.append(item['snippet']['title'])
        descriptions.append(item['snippet']['description'])
        channel_titles.append(item['snippet']['channelTitle'])

    dataframe = pd.DataFrame({'thumbnail': thumbnails, 'video_id': video_ids, 'title': titles, 'description': descriptions, 'channel_title': channel_titles})
    dataframe['video_id'] = 'https://www.youtube.com/watch?v=' + dataframe['video_id'].astype(str)
    dataframe = dataframe.rename(columns={'video_id': 'url'})

    gratitude_score_df = []
    video_id_df = []

    for video_id in video_ids:
        all_comments=[]
        qtyReplies = 0
        qtyMainComments = 0

        # getting comments
        comments = get_comments(youtube_api,video_id,'')

        if not all_comments:
            continue

        # converting into dataframe
        df = pd.DataFrame(all_comments,columns=['comment'])

        # data cleaning
        ## Converting to str type
        df['comment'] = df['comment'].astype(str)

        ## Removing all the emoji's from the dataframe
        df = df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))

        ## Removing all the url's from the data frame‚
        df['comment'] = df['comment'].apply(lambda x: re.split('<a href="https:\/\/.*', str(x))[0])

        ## Removing all special characters
        df['comment'] = df['comment'].apply(process_content)

        ## Converting to lower case
        df['comment'] = df['comment'].str.lower()

        ## Removing empty rows
        df['comment'].replace('', np.nan, inplace=True)
        df.dropna()

        #delete:
        #first callback reply
        df['comment'] = df['comment'].replace(re.compile('first callback reply', re.IGNORECASE), '', regex=True)

        #it was suggested to use .loc instead
        df.loc[:, 'comment'] = df['comment'].replace(re.compile('comment with replies', re.IGNORECASE), '', regex=True)

        #delete:
        #comment with replies
        df['comment'] = df['comment'].replace(re.compile('while getting replies', re.IGNORECASE), '', regex=True)

        df['comment'] = df['comment'].str.strip()  # Remove leading and trailing whitespace
        df = df[df['comment'] != '']

        # predicting labels
        MAX_SEQ_LENGTH = 128
        # Tokenize input using Keras Tokenizer
        tokenizer = Tokenizer()
        # Tokenize input using the same Tokenizer instance used for training
        X_new_list = df['comment'].tolist()
        tokenizer.fit_on_texts(X_train_list)
        X_new_seq = tokenizer.texts_to_sequences(X_new_list)
        X_new_padded = pad_sequences(X_new_seq, maxlen=MAX_SEQ_LENGTH, padding='post')

        # Predict labels for the new dataset
        predictions = model.predict(X_new_padded)

        # Assuming the threshold for considering a comment as 'positive' is 0.5
        threshold = 0.5
        predicted_labels = (predictions > threshold).astype(int)

        # Add the predicted labels to the new DataFrame
        df['predicted_labels'] = predicted_labels

        # calculating gratitude score
        gratitude_score = (df["predicted_labels"].sum()) / len(df["predicted_labels"])
        gratitude_score_df.append(gratitude_score)
        video_id_df.append(video_id)

        if len(video_id_df)==5:
            break

    gs_vid_df = pd.DataFrame({'video_id': video_id_df, 'gratitude_score': gratitude_score_df})
    gs_vid_df['video_id'] = 'https://www.youtube.com/watch?v=' + gs_vid_df['video_id'].astype(str)
    gs_vid_df = gs_vid_df.rename(columns={'video_id': 'url'})
    dataframe = dataframe.merge(gs_vid_df, on='url')
    dataframe.index = dataframe.index + 1
    dataframe = dataframe.sort_values(by=['gratitude_score'], ascending=False)

    st.markdown(dataframe.to_html(render_links=True, escape=False, formatters=dict(thumbnail=path_to_image_html)),unsafe_allow_html=True)

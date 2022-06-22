from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')

app = Flask(__name__)

def read_pickle(filename):
    file_name = './Model/%s'%(filename)
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def retrieve_recommendation(user_input):
    cleaned_df = read_pickle('cleaned_df.pkl')
    product_id_map = read_pickle('product_id_map.pkl')
    user_final_rating = read_pickle('user_user_recommend.pkl')
    classifier_lr_tfidf = read_pickle('lr_tfidf_model.pkl')
    tfidf_vectorizer = read_pickle('tfidf_vectorizer.pkl')
    
    user_present = user_input in list(user_final_rating.index)
    if not user_present:
        return False
    
    d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
    top20 = pd.merge(d, product_id_map, left_on = 'id', right_on = 'id', how = 'left')

    merged_top_20 = pd.merge(top20, cleaned_df[['id', 'cleaned_reviews']], left_on='id', right_on='id', how='left')
    
    tfidf_top20 = tfidf_vectorizer.transform(merged_top_20['cleaned_reviews'])
    predicted_sentiments = classifier_lr_tfidf.predict(tfidf_top20)
    predicted_sentiments = pd.DataFrame(predicted_sentiments, columns = ['predicted_sentiment'])
    
    recommended = pd.concat([merged_top_20, predicted_sentiments], axis=1)
    
    popular_percentage = round((recommended.groupby('id')['predicted_sentiment'].sum() / recommended.groupby('id')['predicted_sentiment'].count()) * 100, 2)
    sorted_recommendations = recommended.groupby('id')['predicted_sentiment'].count().sort_values(ascending=False)

    review_count_percentage = pd.merge(sorted_recommendations, popular_percentage, left_on='id', right_on='id', how='left')
    review_count_percentage = review_count_percentage.rename(columns = {'predicted_sentiment_x': 'ReviewsCount', 'predicted_sentiment_y':'Positive_sentiment_rate'})
    
    final_recommendation = pd.merge(review_count_percentage, product_id_map, left_on='id', right_on='id', how='left')
    recommended_products = list(final_recommendation[final_recommendation['Positive_sentiment_rate'] > final_recommendation['Positive_sentiment_rate'].mean()]['name'])[:20]
    return recommended_products  

@app.route("/", methods=['POST', 'GET'])
def recommend():
    if request.method == 'POST':
        user_input = request.form['username']
        resp = retrieve_recommendation(user_input.strip())
        
        resp = [] if not resp else resp
        return render_template('index.html', cnt=len(resp), resp=resp, user=user_input)
    if request.method == 'GET':
        return render_template('index.html', user='')

if __name__ == '__main__':
    app.run()

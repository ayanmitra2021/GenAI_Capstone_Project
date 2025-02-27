Product Recommendation System Capstone Project

This folder contains the following files:
1. GenAI_Capstone_Project_Sentiment_Based_Recommendation v1.0.docx - A general documentation on how the overall project is implemented. The objectives, alternatives, expected outcome, future prospects etc. are documented in this file.
2. app.py - The flask application file containing the endpoint which is invoked when the frontend is loaded. It has python program that loads the model, vectorizer, and matrices that is used for item and user based recommendations.
3. GenAI_Capstone_Project.ipynb - The Jupyter notebook file where all the steps recommended for this capstone project are implemented. It has 7 sections, starting from data reading, cleansing, text processing, feature selection, training etc. This notebook is used to produce the model, vectorizer and the metrices that is further used in the flask application for recommendation.
4. item_user_matrix.joblib, sentiment_model.joblib, tfidf_vectorizer.joblib, user_item_matrix.joblib - The saved models, vectorizers, and metrices files that is used by the recommendation system.
5. /templates/index.html - The index.html that is used for the frontend.

How to run this application:
1. Download all the above files in a folder.
2. Make sure you have flask and all the other dependencies installed. 
3. Please make sure the index.html file is located inside the /templates folder otherwise the page will not render.
4. Make sure the sample30.csv is located in the same folder where the app.py file is available. 
5. Go to command prompt and execute the following command "python app.py". 
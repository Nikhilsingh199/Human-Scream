#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip install librosa')
#


# Name : Nikhil Singh
# Reg no : 12109514
# Roll No : 44
# Subject : CSM354

# ## HUMAN SCREAM DETECTION
# 
# IT will Take Audio file as an input and extact the mfcc features from it . And it will predict like whether it has human scream , it is having high risk or not . We can implement this model for building an application which can send message to concern authority. Once , medium or high risk detected.
# have used various libraries for different functionalities : 
# Librosa:: Librosa is a package for music and audio analysis. It provides functions to load audio files in various formats, compute various audio features (such as Mel-frequency cepstral coefficients (MFCCs).
# here , in this project we will extract the mfcc feature from the given audio and label the scream and not scream .
# 
# resampy:Resampy is a library for high-quality audio sample rate conversion. It provides efficient algorithms for resampling audio signals from one sample rate to another while preserving the signal quality as much as possible.
# 
# pandas:to read the mfcc.csv.
# 
# sklearn:to import various algorithm and model like linear logistic regression and support vector machine .
# 
# matplotlib:for making diagram
# 
# seaborn:for making diagrams
# 
# joblib:to save the svm and mlp model.
# 
# tensorflow:to load and use mlp model.
# 

# In[ ]:




pip install resampy
# # EXTRACTING MFCC FEATURES FROM AUDIO FOLDER.
# HACINF SUB FOLDERS SCREAMING(WAV FILE IN WHICH HUMAN SCREAM PRESENT) AND NOT SCREAMING(WAV FILE HUMAN SCREAM IS NOT PRESENT )

# In[34]:


import librosa
import pandas as pd
import os


# In[35]:


import os
import zipfile
import librosa

# Path to the ZIP archive
zip_file_path = r'C:\Users\Lenovo\Downloads\archive (9).zip'

# Temporary directory to extract files
extracted_folder = r'C:\Users\Lenovo\Downloads\extracted_files'

# Extract the ZIP archive
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

# Define the folders after extraction
not_screaming_folder = os.path.join(extracted_folder, 'NotScreaming')
screaming_folder = os.path.join(extracted_folder, 'Screaming')



# In[36]:






# Function to extract MFCCs from audio files
def extract_features(file_path):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        # Flatten the MFCCs matrix
        mfccs_flattened = mfccs.mean(axis=1)
        return mfccs_flattened
    except Exception as e:
        print(f"Error encountered while processing {file_path}: {e}")
        return None

# Create empty lists to store features and labels
features = []
labels = []

# Process "screaming" files
for filename in os.listdir(screaming_folder):
    if filename.endswith('.wav'):
        file_path = os.path.join(screaming_folder, filename)
        mfccs = extract_features(file_path)
        if mfccs is not None:
            features.append(mfccs)
            labels.append('Screaming')

# Process "not screaming" files
for filename in os.listdir(not_screaming_folder):
    if filename.endswith('.wav'):
        file_path = os.path.join(not_screaming_folder, filename)
        mfccs = extract_features(file_path)
        if mfccs is not None:
            features.append(mfccs)
            labels.append('NotScreaming')


# In[38]:


# Convert features and labels into a DataFrame
df = pd.DataFrame(features)
df['label'] = labels

# Save DataFrame to a CSV file
csv_filename = 'mfcc_features.csv'
df.to_csv(csv_filename, index=False)
print(f"MFCC features saved to {csv_filename}")


# In[39]:


df.head()


# In[40]:


df.tail()


# In[41]:


df.describe()


# In[42]:


# Assuming the last column contains the labels
screaming_count = df.iloc[:, -1].value_counts().get('Screaming', 0)

# Print the total number of data points labeled as "Screaming"
print("Total number of data points labeled as 'Screaming':", screaming_count)


# In[43]:


# Assuming the last column contains the labels
not_screaming_count = df.iloc[:, -1].value_counts().get('NotScreaming', 0)

print("Total number of data points labeled as 'NotScreaming':", not_screaming_count)


# In[44]:


# Select all columns except the last one (label column)
features_df = df.iloc[:, :-1]

# Print the features
print("Features of the dataset:")
print(features_df)


# # ADDING 264 SCREAMING 
#     TAKING 264 WAV FILES AND CONVERTING IT INTO MFCC FEATURES AND GIVING THEM LABEL SCREAMING AND MERGING THE DATAFRAME IN THE EXISTING CSV FILE MFCC_FEATURES NAMED IT combined_mfcc_features.csv.WE WILL DO THIS LATER

# In[45]:


import os
os.environ['LIBROSA_LOADERS_BACKEND'] = 'audioread'
import librosa


# In[46]:


import os
import zipfile
import librosa
import pandas as pd

# Function to extract MFCCs from audio files
def extract_features(file_path):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        # Flatten the MFCCs matrix
        mfccs_flattened = mfccs.mean(axis=1)
        return mfccs_flattened
    except Exception as e:
        print(f"Error encountered while processing {file_path}: {e}")
        return None

# Path to the directory containing the "Screaming" audio files
screaming_folder2 = r'C:\Users\Lenovo\Downloads\Screamaud'

# Create empty lists to store features and labels
features = []
labels = []

# Process "screaming" files
for filename in os.listdir(screaming_folder2):
    if filename.endswith('.wav'):
        file_path = os.path.join(screaming_folder2, filename)
        mfccs = extract_features(file_path)
        if mfccs is not None:
            features.append(mfccs)
            labels.append('Screaming')
           

# Convert features and labels into a DataFrame
df1 = pd.DataFrame(features)
df1['label'] = labels

# Save DataFrame to a CSV file
csv_filename = 'screaming_mfcc_features.csv'
df1.to_csv(csv_filename, index=False)
print(f"MFCC features of 'Screaming' audio files saved to {csv_filename}")


# In[47]:


df1.head()


# In[48]:


df1.tail()


# In[49]:


df = pd.read_csv("mfcc_features.csv")
df.head()


# # LINEAR REGRESSION

# We separate the features (X) and labels (y) from the DataFrame df.
# If the labels are not in numerical format, we transform them into numerical format. In this example, we assign the label "Screaming" as 1 and "NotScreaming" as 0.
# We split the dataset into training and testing sets using train_test_split() function.
# We initialize and fit a linear regression model using the training data.
# We predict on the testing set and calculate the mean squared error as a measure of the model's performance.

# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Separate features and labels
X = df.iloc[:, :-1]  # Features (MFCCs)
y = df['label']      # Labels (Screaming)

# Transform labels into numerical format if needed
y = y.replace({'Screaming': 1, 'NotScreaming': 0})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[19]:


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()


# In[ ]:





# ## Logistic Regression
# 
# 
# Logistic regression can be used with your MFCC (Mel-Frequency Cepstral Coefficients) features dataset for classification tasks. If your dataset includes MFCC features along with corresponding labels indicating different classes or categories, logistic regression can help you build a classifier to predict the class labels based on the MFCC features

# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('mfcc_features.csv')

# Separate features and target variable
X = data.drop(columns=['label'])  # Features
y = data['label']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the logistic regression model
model = LogisticRegression(multi_class='ovr')  # OvR strategy
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
#One-vs-Rest (OvR) or One-vs-All (OvA)


# In[23]:


from sklearn.metrics import f1_score

# Calculate F1-score
f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' averaging for multi-class classification
print("F1-score:", f1)


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# This code will create a heatmap of the confusion matrix, where each cell represents the number of samples that belong to a particular combination of true and predicted classes. The diagonal elements represent correct predictions, while off-diagonal elements represent misclassifications.
# 
# Additionally, you can also visualize the decision boundaries of the logistic regression model if your feature space is two-dimensional. However, since MFCC features are typically high-dimensional, it may not be feasible to visualize the decision boundaries directly.

# ## SVM MODEL

# In[24]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('mfcc_features.csv')

# Split features and labels
X = data.iloc[:, :-1]  # Features (MFCCs)
y = data['label']      # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model using joblib
joblib.dump(svm_model, 'svm_model.pkl')


# In[ ]:


get_ipython().system('pip install tensorflow')


# In[ ]:


pip install --upgrade numpy


# ## MPN 
# model Multilayer perceptrons mode

# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('mfcc_features.csv')

# Split features (MFCCs) and labels
X = data.iloc[:, :-1]  # Features
y = data['label']      # Labels

# Convert labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the MLP model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # Number of classes = 4 (scream, noise, speech, shout)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Save the model
model.save('mpn_model.h5')


# ## TESTING SVM AND MPN MODEL 
# 
# ON AN AUDIO FILE : 

# Installing compartible version of numpy with tensorflow

# In[ ]:


pip install --upgrade numba llvmlite numpy


# In[ ]:


pip install --upgrade librosa soundfile


# In[ ]:


pip install --upgrade numpy


# In[ ]:


import numpy as np

print(np.__version__)


# In[ ]:


pip install --upgrade numpy


# In[ ]:


import numpy as np

print(np.__version__)


# In[ ]:


pip install numpy==1.20.3


# In[ ]:


import numpy as np

print(np.__version__)


# In[ ]:


pip install tensorflow==2.11.0


# In[ ]:


import tensorflow as tf
print(tf.__version__)


# In[ ]:


import librosa
print(librosa.__version__)


# ## Testing svm and mpn

# In[3]:


import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa

# Load the trained models
svm_model = joblib.load('svm_model.pkl')
mlp_model = tf.keras.models.load_model('mpn_model.h5')

# Function to extract MFCC features from audio file
def extract_features(file_path):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        # Flatten the MFCCs matrix
        mfccs_flattened = mfccs.mean(axis=1)
        return mfccs_flattened
    except Exception as e:
        print(f"Error encountered while processing {file_path}: {e}")
        return None

# Path to the new audio file
audio_file_path = r'C:\Users\Lenovo\Downloads\extracted_files\Screaming\hU3MwNvgK68_out.wav'

# Extract features from the new audio file
new_audio_features = extract_features(audio_file_path)

if new_audio_features is not None:
    # Make predictions using the SVM model
    svm_prediction = svm_model.predict([new_audio_features])[0]
    
    # Make predictions using the MLP model
    mlp_prediction = np.argmax(mlp_model.predict(np.expand_dims(new_audio_features, axis=0)), axis=-1)[0]
    
    # Determine risk level based on predictions
    if svm_prediction == 1 and mlp_prediction == 1:
        risk_level = 'high'
    elif svm_prediction == 1 or mlp_prediction == 1:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    # Generate alert message based on risk level
    if risk_level == 'high':
        print("High-risk alert message: Human scream detected by both models!")
    elif risk_level == 'medium':
        print("Medium-risk alert message: Human scream detected by one of the models!")
    else:
        print("No risk alert message: No human scream detected.")
else:
    print("Error extracting features from the audio file.")


# In[5]:


import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa

# Load the trained models
svm_model = joblib.load('svm_model.pkl')
mlp_model = tf.keras.models.load_model('mpn_model.h5')

# Function to extract MFCC features from audio file
def extract_features(file_path):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        # Flatten the MFCCs matrix
        mfccs_flattened = mfccs.mean(axis=1)
        return mfccs_flattened
    except Exception as e:
        print(f"Error encountered while processing {file_path}: {e}")
        return None

# Path to the new audio file
audio_file_path = r'C:\Users\Lenovo\Downloads\extracted_files\Screaming\Hu18136at0Y_out.wav'

# Extract features from the new audio file
new_audio_features = extract_features(audio_file_path)

if new_audio_features is not None:
    # Make predictions using the SVM model
    svm_prediction = svm_model.predict([new_audio_features])[0]
    
    # Make predictions using the MLP model
    mlp_prediction = np.argmax(mlp_model.predict(np.expand_dims(new_audio_features, axis=0)), axis=-1)[0]
    
    # Determine risk level based on predictions
    if svm_prediction == 1 and mlp_prediction == 1:
        risk_level = 'high'
    elif svm_prediction == 1 or mlp_prediction == 1:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    # Generate alert message based on risk level
    if risk_level == 'high':
        print("High-risk alert message: Human scream detected by both models!")
    elif risk_level == 'medium':
        print("Medium-risk alert message: Human scream detected by one of the models!")
    else:
        print("No risk alert message: No human scream detected.")
else:
    print("Error extracting features from the audio file.")


# In[4]:


import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
from IPython.display import Audio

# Load the trained models
svm_model = joblib.load('svm_model.pkl')
mlp_model = tf.keras.models.load_model('mpn_model.h5')

# Function to extract MFCC features from audio file
def extract_features(file_path):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        # Flatten the MFCCs matrix
        mfccs_flattened = mfccs.mean(axis=1)
        return mfccs_flattened, sr, audio
    except Exception as e:
        print(f"Error encountered while processing {file_path}: {e}")
        return None, None, None

# Path to the new audio file
audio_file_path = r'C:\Users\Lenovo\Downloads\WhatsApp Audio 2024-04-02 at 9.25.14 AM.wav'

# Extract features from the new audio file
new_audio_features, sr, audio = extract_features(audio_file_path)

if new_audio_features is not None:
    # Make predictions using the SVM model
    svm_prediction = svm_model.predict([new_audio_features])[0]
    
    # Make predictions using the MLP model
    mlp_prediction = np.argmax(mlp_model.predict(np.expand_dims(new_audio_features, axis=0)), axis=-1)[0]
    
    # Determine risk level based on predictions
    if svm_prediction == 1 and mlp_prediction == 1:
        risk_level = 'high'
    elif svm_prediction == 1 or mlp_prediction == 1:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    # Generate alert message based on risk level
    if risk_level == 'high':
        print("High-risk alert message: Human scream detected by both models!")
    elif risk_level == 'medium':
        print("Medium-risk alert message: Human scream detected by one of the models!")
    else:
        print("No risk alert message: No human scream detected.")
    
    # Listen to the audio
    Audio(data=audio, rate=sr)
else:
    print("Error extracting features from the audio file.")


# In[3]:


import librosa
import numpy as np
import joblib

# Load the SVM model
svm_model = joblib.load('svm_model.pkl')

# Define a function to extract MFCC features from a WAV file
def extract_features(file_path):
    try:
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=None)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfccs_processed = np.mean(mfccs.T, axis=0)  # Take the mean along the time axis
        
        return mfccs_processed
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# File path of the WAV file to test
wav_file_path = r'C:\Users\Lenovo\Downloads\WhatsApp Audio 2024-04-02 at 9.25.14 AM.wav'

# Extract features from the WAV file
features = extract_features(wav_file_path)

# Make prediction using the SVM model
if features is not None:
    prediction = svm_model.predict([features])[0]
    
    # Interpret the prediction
    if prediction == 'Screaming':
        print("The SVM model detected a scream in the audio.")
    else:
        print("The SVM model did not detect a scream in the audio.")


# ## Testing the model on an mfcc feature containing human scream
# HUMAN SCREAM :861
# 'NotScreaming': 2631

# In[2]:


import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained models
svm_model = joblib.load('svm_model.pkl')
mlp_model = tf.keras.models.load_model('mpn_model.h5')

# Function to determine risk level based on predictions
def determine_risk_level(svm_prediction, mlp_prediction):
    if svm_prediction == 1 and mlp_prediction == 1:
        return 'high'
    elif svm_prediction == 1 or mlp_prediction == 1:
        return 'medium'
    else:
        return 'low'

# Function to generate alert message based on risk level
def generate_alert_message(risk_level):
    if risk_level == 'high':
        return "High-risk alert message: Human scream detected by both models!"
    elif risk_level == 'medium':
        return "Medium-risk alert message: Human scream detected by one of the models!"
    else:
        return "No risk alert message: No human scream detected."

# Load the dataset
dataset = np.genfromtxt('mfcc_features.csv', delimiter=',', skip_header=1)

# Extract features and labels
mfcc_features = dataset[:, :-1]
labels = dataset[:, -1]

# Make predictions using the SVM model
svm_predictions = svm_model.predict(mfcc_features)

# Make predictions using the MLP model
mlp_predictions = np.argmax(mlp_model.predict(mfcc_features), axis=-1)

# Determine risk level for each prediction
risk_levels = [determine_risk_level(svm_pred, mlp_pred) for svm_pred, mlp_pred in zip(svm_predictions, mlp_predictions)]

# Generate alert messages based on risk levels
alert_messages = [generate_alert_message(risk_level) for risk_level in risk_levels]

# Print alert messages for each prediction
for idx, alert_message in enumerate(alert_messages):
    print(f"Prediction {idx+1}: {alert_message} (True label: {labels[idx]})")


# ## NEW DATA WITH LABLE SCREAMING ADDED TO EXISTING CSV MFCC_FEATURE

# In[56]:


import pandas as pd

# Read the first dataset
df1 = pd.read_csv("mfcc_features.csv")

# Read the second dataset
df2 = pd.read_csv("screaming_mfcc_features.csv")

# Concatenate both datasets
combined_df = pd.concat([df1, df2], ignore_index=True)

# Optionally, shuffle the rows of the combined DataFrame
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Display the combined DataFrame
print(combined_df.head())
# Save the combined DataFrame to a new CSV file
combined_csv_filename = 'combined_mfcc_features.csv'
combined_df.to_csv(combined_csv_filename, index=False)
print(f"Combined dataset saved to {combined_csv_filename}")


# In[53]:


combined_df.head()


# In[88]:


combined_df.tail()


# # SVM MODEL

# In[57]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('combined_mfcc_features.csv')

# Split features and labels
X = data.iloc[:, :-1]  # Features (MFCCs)
y = data['label']      # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model using joblib
joblib.dump(svm_model, 'svm_model.pkl')


# # MPN MODEL

# In[58]:


import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('combined_mfcc_features.csv')

# Split features (MFCCs) and labels
X = data.iloc[:, :-1]  # Features
y = data['label']      # Labels

# Convert labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the MLP model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # Number of classes = 4 (scream, noise, speech, shout)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Save the model
model.save('mpn_model.h5')


# In[59]:


import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained models
svm_model = joblib.load('svm_model.pkl')
mlp_model = tf.keras.models.load_model('mpn_model.h5')

# Function to determine risk level based on predictions
def determine_risk_level(svm_prediction, mlp_prediction):
    if svm_prediction == 1 and mlp_prediction == 1:
        return 'high'
    elif svm_prediction == 1 or mlp_prediction == 1:
        return 'medium'
    else:
        return 'low'

# Function to generate alert message based on risk level
def generate_alert_message(risk_level):
    if risk_level == 'high':
        return "High-risk alert message: Human scream detected by both models!"
    elif risk_level == 'medium':
        return "Medium-risk alert message: Human scream detected by one of the models!"
    else:
        return "No risk alert message: No human scream detected."

# Load the dataset
dataset = np.genfromtxt('combined_mfcc_features.csv', delimiter=',', skip_header=1)

# Extract features and labels
mfcc_features = dataset[:, :-1]
labels = dataset[:, -1]

# Make predictions using the SVM model
svm_predictions = svm_model.predict(mfcc_features)

# Make predictions using the MLP model
mlp_predictions = np.argmax(mlp_model.predict(mfcc_features), axis=-1)

# Determine risk level for each prediction
risk_levels = [determine_risk_level(svm_pred, mlp_pred) for svm_pred, mlp_pred in zip(svm_predictions, mlp_predictions)]

# Generate alert messages based on risk levels
alert_messages = [generate_alert_message(risk_level) for risk_level in risk_levels]

# Print alert messages for each prediction
for idx, alert_message in enumerate(alert_messages):
    print(f"Prediction {idx+1}: {alert_message} (True label: {labels[idx]})")


# # LOGISTIC REGRESSION MODEL

# In[69]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('combined_mfcc_features.csv')

# Separate features and target variable
X = data.drop(columns=['label'])  # Features
y = data['label']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the logistic regression model
model = LogisticRegression(multi_class='ovr')  # OvR strategy
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
#One-vs-Rest (OvR) or One-vs-All (OvA)
import joblib

# Define the file path to save the model
model_file_path = "logistic_regression_model.pkl"

# Save the trained model to the file path
joblib.dump(model, model_file_path)

print("Model saved successfully.")


# In[62]:


from sklearn.metrics import f1_score

# Calculate F1-score
f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' averaging for multi-class classification
print("F1-score:", f1)


# In[63]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# # KNN MODEL

# In[68]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv("combined_mfcc_features.csv")

# Split features and labels
X = df.drop(columns=['label'])
y = df['label']

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate classification report
print(classification_report(y_test, y_pred))
import joblib

# Define the file path to save the model
model_file_path = "knn_model.pkl"

# Save the trained model to the file path
joblib.dump(knn_classifier, model_file_path)

print("Model saved successfully.")


# In[65]:


# Assuming the last column contains the labels
screaming_count = df.iloc[:, -1].value_counts().get('Screaming', 0)

# Print the total number of data points labeled as "Screaming"
print("Total number of data points labeled as 'Screaming':", screaming_count)
# Assuming the last column contains the labels
not_screaming_count = df.iloc[:, -1].value_counts().get('NotScreaming', 0)

print("Total number of data points labeled as 'NotScreaming':", not_screaming_count)


# # CNN MODEL

# In[66]:


import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Preprocessing
X = df.drop(columns=['label']).values  # Features
y = df['label'].map({'Screaming': 1, 'NotScreaming': 0}).values  # Labels

# Reshape features for CNN input (assuming 20 MFCC coefficients)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definition
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Model evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)


# In[67]:


# Define the file path to save the model
model_path = 'scream_detection_cnn_model.h5'

# Save the model
model.save(model_path)

print(f"Model saved at: {model_path}")


# In[74]:


import librosa
import joblib
import tensorflow as tf
import numpy as np

class ScreamDetectionModel:
    def __init__(self, svm_model_path, mlp_model_path, cnn_model_path, knn_model_path, logistic_regression_model_path):
        self.svm_model = joblib.load(svm_model_path)
        self.mlp_model = tf.keras.models.load_model(mlp_model_path)
        self.cnn_model = tf.keras.models.load_model(cnn_model_path)
        self.knn_model = joblib.load(knn_model_path)
        self.logistic_regression_model = joblib.load(logistic_regression_model_path)

    def extract_mfcc(self, audio_file):
        # Load audio file
        y, sr = librosa.load(audio_file, sr=None)
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        # Flatten the MFCCs matrix
        mfcc_flattened = mfcc.mean(axis=1)
        return mfcc_flattened

    def predict(self, audio_file):
        # Extract MFCC features from the audio file
        mfcc_features = self.extract_mfcc(audio_file)
        
        # Predict using different models
        svm_prediction = self.svm_model.predict([mfcc_features])[0]
        mlp_prediction = "Screaming" if self.mlp_model.predict(np.array([mfcc_features]))[0][0] > 0.5 else "Not Screaming"
        cnn_prediction = "Screaming" if self.cnn_model.predict(np.expand_dims(np.expand_dims(mfcc_features, axis=0), axis=-1))[0][0] > 0.5 else "Not Screaming"
        knn_prediction = self.knn_model.predict([mfcc_features])[0]
        logistic_regression_prediction = self.logistic_regression_model.predict([mfcc_features])[0]

        return {
            "SVM": "Screaming" if svm_prediction == 1 else "Not Screaming",
            "MLP": mlp_prediction,
            "CNN": cnn_prediction,
            "KNN": "Screaming" if knn_prediction == 1 else "Not Screaming",
            "LogisticRegression": "Screaming" if logistic_regression_prediction == 1 else "Not Screaming"
        }

# Example usage:
model = ScreamDetectionModel(
    svm_model_path='svm_model.pkl',
    mlp_model_path='mpn_model.h5',
    cnn_model_path='scream_detection_cnn_model.h5',
    knn_model_path='knn_model.pkl',
    logistic_regression_model_path='logistic_regression_model.pkl'
)

audio_file_path = r'C:\Users\Lenovo\Downloads\Screamaud\Scream (82).wav'
predictions = model.predict(audio_file_path)

for model_name, prediction in predictions.items():
    print(f"{model_name} model detected {prediction}!")


# In[75]:


# Instantiate the ScreamDetectionModel class with the paths to your trained models
model = ScreamDetectionModel(
    svm_model_path='svm_model.pkl',
    mlp_model_path='mpn_model.h5',
    cnn_model_path='scream_detection_cnn_model.h5',
    knn_model_path='knn_model.pkl',
    logistic_regression_model_path='logistic_regression_model.pkl'
)

# Define the audio file path
audio_file_path = r'C:\Users\Lenovo\Downloads\Screamaud\Scream (83).wav'

# Predict using the model
predictions = model.predict(audio_file_path)

# Display the predictions
for model_name, prediction in predictions.items():
    print(f"{model_name} model detected {prediction}!")


# In[79]:



model = ScreamDetectionModel(
    svm_model_path='svm_model.pkl',
    mlp_model_path='mpn_model.h5',
    cnn_model_path='scream_detection_cnn_model.h5',
    knn_model_path='knn_model.pkl',
    logistic_regression_model_path='logistic_regression_model.pkl'
)

# Define the audio file path
audio_file_path = r'C:\Users\Lenovo\Downloads\Screamaud\Scream (86).wav'

# Predict using the model
predictions = model.predict(audio_file_path)

# Display the predictions
for model_name, prediction in predictions.items():
    print(f"{model_name} model detected {prediction}!")


# In[80]:


model = ScreamDetectionModel(
    svm_model_path='svm_model.pkl',
    mlp_model_path='mpn_model.h5',
    cnn_model_path='scream_detection_cnn_model.h5',
    knn_model_path='knn_model.pkl',
    logistic_regression_model_path='logistic_regression_model.pkl'
)
#giving error 
# Define the audio file path
audio_file_path = r'C:\Users\Lenovo\Downloads\Screamaud\test_EN2002b_1210_1220_dknox_3_4.wav'

# Predict using the model
predictions = model.predict(audio_file_path)

# Display the predictions
for model_name, prediction in predictions.items():
    print(f"{model_name} model detected {prediction}!")


# In[81]:


def risk_assessment(audio_file_path):
    # Instantiate the ScreamDetectionModel class with the paths to your trained models
    model = ScreamDetectionModel(
        svm_model_path='svm_model.pkl',
        mlp_model_path='mpn_model.h5',
        cnn_model_path='scream_detection_cnn_model.h5',
        knn_model_path='knn_model.pkl',
        logistic_regression_model_path='logistic_regression_model.pkl'
    )

    # Predict using the model
    predictions = model.predict(audio_file_path)

    # Count the number of models predicting "Screaming"
    screaming_count = sum(1 for prediction in predictions.values() if prediction == 'Screaming')

    # Generate risk assessment message based on the count
    if screaming_count >= 3:
        return "High risk"
    elif screaming_count == 2:
        return "Medium risk"
    elif screaming_count == 1:
        return "Low risk"
    else:
        return "No risk"

# Define the audio file path
audio_file_path = r'C:\Users\Lenovo\Downloads\Screamaud\Scream (83).wav'

# Get the risk assessment
risk_message = risk_assessment(audio_file_path)
print("Risk assessment:", risk_message)


# In[83]:


import librosa
import joblib
import tensorflow as tf
import numpy as np

class ScreamDetectionModel:
    def __init__(self, svm_model_path, mlp_model_path, cnn_model_path, knn_model_path, logistic_regression_model_path):
        self.svm_model = joblib.load(svm_model_path)
        self.mlp_model = tf.keras.models.load_model(mlp_model_path)
        self.cnn_model = tf.keras.models.load_model(cnn_model_path)
        self.knn_model = joblib.load(knn_model_path)
        self.logistic_regression_model = joblib.load(logistic_regression_model_path)

    def extract_mfcc(self, audio_file):
        # Load audio file
        y, sr = librosa.load(audio_file, sr=None)
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        # Flatten the MFCCs matrix
        mfcc_flattened = mfcc.mean(axis=1)
        return mfcc_flattened

    def predict(self, audio_file):
        # Extract MFCC features from the audio file
        mfcc_features = self.extract_mfcc(audio_file)
        
        # Predict using different models
        svm_prediction = self.svm_model.predict([mfcc_features])[0]
        mlp_prediction = "Screaming" if self.mlp_model.predict(np.array([mfcc_features]))[0][0] > 0.5 else "Not Screaming"
        cnn_prediction = "Screaming" if self.cnn_model.predict(np.expand_dims(np.expand_dims(mfcc_features, axis=0), axis=-1))[0][0] > 0.5 else "Not Screaming"
        knn_prediction = self.knn_model.predict([mfcc_features])[0]
        logistic_regression_prediction = self.logistic_regression_model.predict([mfcc_features])[0]

        return {
            "SVM": "Screaming" if svm_prediction == 1 else "Not Screaming",
            "MLP": mlp_prediction,
            "CNN": cnn_prediction,
            "KNN": "Screaming" if knn_prediction == 1 else "Not Screaming",
            "LogisticRegression": "Screaming" if logistic_regression_prediction == 1 else "Not Screaming"
        }
    
    def risk_assessment(self, audio_file_path):
        # Predict using the model
        predictions = self.predict(audio_file_path)

        # Count the number of models predicting "Screaming"
        screaming_count = sum(1 for prediction in predictions.values() if prediction == 'Screaming')

        # Generate risk assessment message based on the count
        if screaming_count >= 3:
            return "High risk"
        elif screaming_count == 2:
            return "Medium risk"
        elif screaming_count == 1:
            return "Low risk"
        else:
            return "No risk"

# Example usage:
audio_file_path = r'C:\Users\Lenovo\Downloads\Screamaud\Scream (83).wav'
model = ScreamDetectionModel(
    svm_model_path='svm_model.pkl',
    mlp_model_path='mpn_model.h5',
    cnn_model_path='scream_detection_cnn_model.h5',
    knn_model_path='knn_model.pkl',
    logistic_regression_model_path='logistic_regression_model.pkl'
)

risk_message = model.risk_assessment(audio_file_path)
print("Risk assessment:", risk_message)
for model_name, prediction in predictions.items():
    print(f"{model_name} model detected {prediction}!")


# In[ ]:





import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# LOADING DATASET FROM MNIST
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# NORMALIZING DATA BETWEEN 0 AND 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# RESHAPING DIMENSIONS AND RESIZING TO 28X28
IMG_SIZE = 28
x_trainr = x_train.reshape(60000, 784)
x_testr = x_test.reshape(10000, 784)

print("Training Samples: ", x_trainr.shape)
print("Testing Samples: ", x_testr.shape)

# FEATURE SCALING
scaler = StandardScaler()
scaler.fit(x_trainr)

xTraining = scaler.transform(x_trainr)
xTesting = scaler.transform(x_testr)

# TRAINING AND FITTING THE MODEL
classifier = svm.SVC()
classifier.fit(x_trainr, y_train)
#
yP = classifier.predict(x_testr)

# EVALUATION METRICS
print("\nCONFUSION MATRIX: ")
print(confusion_matrix(y_test, yP))

print("\nCLASSIFICATION REPORT: ")
print(classification_report(y_test, yP, zero_division=1))

print("\nACCURACY: ")
print(accuracy_score(y_test, yP)*100)

# SAVING THE MODEL
filename = 'SVM_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# EXAMPLE OF PREDICTION
print("PREDICTED BY MODEL:", yP[0])
print("ORIGINAL", y_test[0])

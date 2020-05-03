
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model


model = load_model('model.h5')

x_test = np.load('xtest.npy')
y_test = np.load('ytest.npy')


print('\n# Evaluate on test data')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)

print('\n# Generate predictions for test samples')
predictions = model.predict(x_test)

y_pred = np.array([])
for x in predictions:
    for a in x:
        if(a>0.5):
            y_pred = np.append(y_pred,1)
        else:
            y_pred = np.append(y_pred,0)


print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
target_names = ['Defect', 'NoDefect']
print(classification_report(y_test, y_pred, target_names=target_names))


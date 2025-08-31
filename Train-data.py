import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

data_directory = pickle.load(open("./data.pickle", 'rb'))

data = np.asarray(data_directory['data'])
labels = np.asarray(data_directory['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print(f"{score*100}% of samples were classified correctly !")

pickle_path = os.path.join(os.path.dirname(__file__), "model.p")
with open(pickle_path, "wb") as f:
    pickle.dump({"model": model}, f)
print(f"Wrote pickle to: {pickle_path}")
import pickle
with open('textClass_model.pkl', 'rb') as file:
    model = pickle.load(file)

text = "switch off the radio"

pred = model.predict([text])[0]

print(pred)
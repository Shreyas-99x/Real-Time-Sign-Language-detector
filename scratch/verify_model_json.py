import json

with open('web/model/model.json') as f:
    m = json.load(f)
print('Format:', m.get('format'))
print('Classes:', len(m.get('classes', [])), m.get('classes'))
print(f"Accuracy: {m.get('accuracy') * 100:.2f}%")
print('Weights layers:', len(m.get('weights', [])))
print('Biases layers:', [len(b) for b in m.get('biases', [])])

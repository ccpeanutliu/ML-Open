import csv
import numpy as np
import sys
from keras.models import load_model

test_path = sys.argv[1]
ans_path = sys.argv[2]
model = load_model('strong.h5')


test_x = []
with open(file = test_path, mode = 'r', encoding = "GB18030") as csvfile:
    rows = csv.reader(csvfile)
    first = True
    for row in rows:
        temp = []
        if(first == True):
            first = False
            continue
        row = row[1].split(' ')
        for i in range(len(row)):
            row[i] = float(row[i])
        test_x.append(row)  
    csvfile.close()

test_x = np.array(test_x, dtype = float)
test_x = np.reshape(test_x, (len(test_x), 48, 48, 1))

result = model.predict(test_x)
ans = []
for i in range(result.shape[0]):
    max = 0
    index = 0
    for j in range(result.shape[1]):
        if(result[i][j]>max):
            max = result[i][j]
            index = j
    ans.append(index)

with open(ans_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "label"])
    for i in range(len(ans)):
        writer.writerow([str(i), ans[i]])
import connect, predict

train, test = connect.create_dataset()

predict.classify(train, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 1, test)
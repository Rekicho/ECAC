import connect, predict

train, test = connect.create_dataset()

predict.classify(train, [1, 2, 17], 3, test)
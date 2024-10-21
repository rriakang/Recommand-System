#dataset name
dataset = 'ml-1m'

#model name
model = 'NeuMF-end'

#paths
main_path = './ml-1m/'

train_rating = main_path + '{}.train.rating.txt'.format(dataset)
test_rating = main_path + '{}.test.rating.txt'.format(dataset)
test_negative = main_path + '{}.test.negative.txt'.format(dataset)
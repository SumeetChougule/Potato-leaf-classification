import tensorflow as tf

daily_sales = [21, 22, -108, 31, -1, 32, 34, 31]

tf_dataset = tf.data.Dataset.from_tensor_slices(daily_sales)

for sales in tf_dataset.as_numpy_iterator():
    print(sales)

for sales in tf_dataset.take(3):
    print(sales)

tf_dataset = tf_dataset.filter(lambda x: x > 0)

tf_dataset = tf_dataset.map(lambda x: x * 72)

tf_dataset = tf_dataset.shuffle(2)

for sales in tf_dataset.as_numpy_iterator():
    print(sales)


for sales_batch in tf_dataset.batch(2):
    print(sales_batch.numpy())


tf_dataset = tf.data.Dataset.from_tensor_slices(daily_sales)

tf_processed = (
    tf_dataset.filter(lambda x: x > 0).map(lambda y: y * 72).shuffle(2).batch(2)
)

for sales in tf_processed.as_numpy_iterator():
    print(sales)

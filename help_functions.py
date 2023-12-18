def batch_data(k, x_train, y_train, number_of_batches, sizes_batches):
    if k == number_of_batches - 1:
        x = x_train[
            sizes_batches[k - 1] * k : sizes_batches[k - 1] * k + sizes_batches[k]
        ]
        y = y_train[
            sizes_batches[k - 1] * k : sizes_batches[k - 1] * k + sizes_batches[k]
        ]
    else:
        x = x_train[sizes_batches[k] * k : sizes_batches[k] * (k + 1)]
        y = y_train[sizes_batches[k] * k : sizes_batches[k] * (k + 1)]

    return x, y


def calculate_batches(x_train, batch_size):
    if len(x_train) % batch_size == 0:
        number_of_batches = int(len(x_train) / batch_size)
        sizes_batches = [batch_size for i in range(number_of_batches)]
    else:
        number_of_batches = int(len(x_train) / batch_size) + 1
        size_last_batch = len(x_train) - batch_size * int(len(x_train) / batch_size)
        sizes_batches = [batch_size for i in range(number_of_batches - 1)]
        sizes_batches.append(size_last_batch)

    return number_of_batches, sizes_batches

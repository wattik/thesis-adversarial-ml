def accuracy(model, query_profiles, labels):
    predictions = model.predict(query_profiles)

    misclassified = list(filter((lambda x: x[0] != x[1]), zip(predictions, labels)))

    return 1 - (len(misclassified) / len(labels))

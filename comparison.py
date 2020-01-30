import glob
import os
import numpy as np

from keras.models import load_model

import utils

def compare_models(model1, model2):
    """Compares the layers and weights of the two models and returns a tuple. The
    first element is the result of the comparison. The other two are the
    offending indices of the layer and weight, if any.
    """

    if len(model1.layers) != len(model2.layers):
        return False
    
    for i_layer, layer in enumerate(model1.layers):
        weights = layer.get_weights()
        
        for i_array, array in enumerate(weights):
            if not ((array == model2.layers[i_layer].get_weights()[i_array]).all()):
                return (False, i_layer, i_array)
    return (True, -1, -1)

def compare_xy(xy_train1, xy_train2):
    """Compares the XY training data and returns True if they are the same, False
    otherwise.
    """
    
    if not np.array_equal(xy_train1['x_train'], xy_train2['x_train']):
        return (False, "X")
    if not np.array_equal(xy_train1['y_train'], xy_train2['y_train']):
        return (False, "Y")
    return (True, "")

def compare_arrays(array1, array2):
    """Compares the arrays saved by the neural network and returns True if they are
    the same, False otherwise.
    """
    
    return np.array_equal(np.array(array1), np.array(array2))

def check_same_models(d, stub_fmt, num_runs):
    """Checks if all models in this directory with the provided stub are the same.
    """
    all_same = True
    num_checked = 0
    messages = []
    
    for i_run in range(num_runs):

        f = os.path.join(d, stub_fmt % i_run, "cifar10_ResNet29v2_model.h5")
        model = load_model(f)
        num_checked += 1
        
        # Save and skip if it's the first model
        if 0 == i_run:
            first_model = model
            continue

        # Check all the layers and weights in the model
        same_model, i_layer, i_array = compare_models(model, first_model)
        if not same_model:
            all_same = False
            messages.append("- Different model (layers and weights compared with run #0) in this run: #%d (file = %s, layer = %d, array = %d)" % (i_run, f, i_layer, i_array))

    return ["Models all the same: " + str(all_same) +
            " (number of models checked: %d/%d)" % (num_checked, utils.get_num_runs())] + messages
        
def check_same_xy_train(d, stub_fmt, num_runs):
    """Checks if all the XY training data are the same.
    """

    all_same = True
    num_checked = 0
    messages = []

    for i_run in range(num_runs):
        f = os.path.join(d, stub_fmt % i_run, "xy_train.npz")
        xy_train = np.load(f)
        num_checked += 1

        if 0 == i_run:
            first_xy_train = xy_train
            continue

        same, offending = compare_xy(xy_train, first_xy_train)
        if not same:
            all_same = False
            messages.append("- Different XY training data (compared with run #0) in this run: #%d (file = %s, offending = %s)" % (i_run, batches_file, offending))

    return ["XY training data all the same: " + str(all_same) +
            "(number of models checked: %d/%d)" % (num_checked, utils.get_num_runs())] + messages

def check_same_batch_arrays(d, stub_fmt, num_runs, num_epochs):
    """Checks if all the batch shuffles are the same.
    """

    all_same = True
    num_checked = 0
    messages = []

    for i_run in range(num_runs):
        arrays = []
        for i_epoch in range(num_epochs):
            f = os.path.join(d, stub_fmt % i_run, "array %d.txt" % i_epoch)
            arrays.append(np.loadtxt(f, delimiter = ","))
        num_checked += 1

        # Save and skip if it's the first model
        if 0 == i_run:
            first_arrays = arrays
            continue

        same = compare_arrays(arrays, first_arrays)
        if not same:
            all_same = False
            messages.append("- Different batches array (compared with run #0) in this run: #%d" % i_run)

    return ["Batches all the same: " + str(all_same) +
            "(number of models checked: %d/%d)" % (num_checked, utils.get_num_runs())] + messages

def compare_all(d = utils.SAVED_MODELS_DIR):
    """Compare models and batches when run with the same or different seeds.
    """

    num_runs = utils.get_num_runs()
    num_epochs = utils.get_num_epochs()
    
    # Iterate on same or different seeds
    full_message = ""
    for same_seed in range(2):
        messages = []
        stub_fmt = utils.seed_to_str_fmt(same_seed)

        messages.extend(check_same_models(d = d, stub_fmt = stub_fmt, num_runs = num_runs))
        messages.extend(check_same_xy_train(d = d, stub_fmt = stub_fmt, num_runs = num_runs))
        messages.extend(check_same_batch_arrays(d = d, stub_fmt = stub_fmt, num_runs = num_runs, num_epochs = num_epochs))

        seed_str = utils.seed_to_str(same_seed)
        full_message += "Comparison results for %s:\n\n" % seed_str + "\n".join(messages) + "\n\n"

    return full_message

def main():
    message = compare_all(d = utils.SAVED_MODELS)
    print(utils.PRINT_START + message + utils.PRINT_END)
        
if "__main__" == __name__:
    print("Current directory: " + os.getcwd())
    main()        


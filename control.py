import json
import os

import cifar10_resnet
import numpy as np

# Our libraries
import comparison
import utils

def main():

    num_runs = utils.get_num_runs()
    num_epochs = utils.get_num_epochs()
    results = dict()

    # Iterate on different or same seed (0 or 1)
    for same_seed in range(2):

        seed_string = utils.seed_to_str(same_seed)
        scores_filepath = utils.SCORES_FILEPATH_FMT % seed_string
        all_scores = []
        
        for i in range(num_runs):
            print(utils.PRINT_START + "Same seed = %d, experiment %d" % (same_seed, i) + utils.PRINT_END)

            if same_seed:
                seed = 0
            else:
                seed = i

            model_str = utils.seed_to_str_fmt(same_seed) % i
            history, scores = cifar10_resnet.script(seed=seed, model_index=model_str, epochs=num_epochs)
            print("Scores: ")
            print(scores)
            all_scores.append(scores)
            output_history = {'acc':history.history['acc'],
                                  'val_acc':history.history['val_acc'],
                                  'loss':history.history['loss'],
                                  'val_loss':history.history['val_loss']}
            # Save scores
            np.savetxt(scores_filepath, np.array(all_scores), delimiter=',')

            # Save loss and accuracy history
            with open(utils.HISTORY_FILEPATH_FMT % (seed_string, i), "w+") as f:
                f.write(json.dumps(output_history))

        # Calculate standard deviations
        results[seed_string] = np.std(np.array(all_scores), axis = 0)

    # Check which models are the same
    comparison_message = comparison.compare_all()
    with open(utils.COMPARISON_FILEPATH, "w+") as f:
        f.write(comparison_message)

    # Print standard deviation results to the user
    print(utils.PRINT_START)
    for key in results:
        print("Standard deviations for: " + key + "\n\n" + str(results[key]) + "\n\n")
    print(utils.PRINT_END)
    
    # Print comparison results to the user
    print("\n\n" + comparison_message + utils.PRINT_END)

if "__main__" == __name__:
    main()

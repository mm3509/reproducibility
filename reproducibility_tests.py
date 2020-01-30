import os

import tensorflow as tf
import numpy as np


def current_system():
    """Returns one of the three tested systems: "colab", "azure", "macOS"."""
    try:
        f = __file__
    except NameError:
        return "colab"
    
    filepath = os.path.abspath(os.path.dirname(f))

    if filepath.startswith("/notebooks"):
        return "azure"
    return "macOS"


def tf_major_version():
    """Returns major version of TensorFlow"""
    return int(tf.__version__.split(".")[0])

def tensorflow2_noneager(eager):
    """Returns True if the current version of Tensorflow is 2 and we are in non-eager mode"""
    return 2 == tf_major_version() and not eager

def format_number(n):
    """Returns the number string-formatted with 12 number after comma."""
    return "%1.12f" % n


def set_top_level_seeds():
    """Sets TensorFlow graph-level seed and Numpy seed."""
    if 1 == tf_major_version():
        tf.set_random_seed(0)
    else:
        tf.random.set_seed(0)
    np.random.seed(0)

def generate_random_numbers_non_eager(op_seed=None):
    """Returns random normal draws, non-eager mode"""

    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                          intra_op_parallelism_threads=1)) as sess:

        set_top_level_seeds()

        if op_seed:
            t = tf.random.normal([100, 100], seed=op_seed)
        else:
            t = tf.random.normal([100, 100])

        return sess.run(t)

    
def generate_random_numbers_eager(op_seed=None):
    """Returns random normal draws, eager mode"""

    set_top_level_seeds()

    if op_seed:
        t = tf.random.normal([100, 100], seed=op_seed)
    else:
        t = tf.random.normal([100, 100])

    return t


def generate_random_numbers_helper(eager, op_seed=None):
    """Wrapper for eager and non-eager functions"""

    if eager:
        return generate_random_numbers_eager(op_seed=op_seed)
    return generate_random_numbers_non_eager(op_seed=op_seed)


def generate_random_number_stats_str_eager(op_seed=None):
    """Returns mean and standard deviation from random normal draws"""

    t = generate_random_numbers_helper(eager=True, op_seed=op_seed)
    
    mean = tf.reduce_mean(t)
    sdev = tf.sqrt(tf.reduce_mean(tf.square(t - mean)))

    return [format_number(n) for n in (mean, sdev)]


def generate_random_number_stats_str_non_eager(op_seed=None):
    
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                          intra_op_parallelism_threads=1)) as sess:
                           
        t = generate_random_numbers_helper(eager=False, op_seed=op_seed)
    
        mean = tf.reduce_mean(t)
        sdev = tf.sqrt(tf.reduce_mean(tf.square(t - mean)))

        return [format_number(sess.run(n)) for n in (mean, sdev)]


def generate_random_number_stats_str_helper(eager, op_seed=None):
    """Wrapper for eager and non-eager functions"""

    if eager:
        return generate_random_number_stats_str_eager(op_seed=op_seed)
    return generate_random_number_stats_str_non_eager(op_seed=op_seed)




def generate_random_number_1_seed(eager):
    """Returns a single random number with graph-level seed only."""
    num = generate_random_numbers_helper(eager)[0, 0]
    return num


def generate_random_number_2_seeds(eager):
    """Returns a single random number with graph-level seed only."""
    num = generate_random_numbers_helper(eager, op_seed=1)[0, 0]
    return num


def generate_stats_1_seed(eager):
    """Returns mean and standard deviation wtih graph-level seed only."""
    return generate_random_number_stats_str_helper(eager)


def generate_stats_2_seeds(eager):
    """Returns mean and standard deviation with graph and operation seeds."""
    return generate_random_number_stats_str_helper(eager, op_seed=1)


class Tests(tf.test.TestCase):
    """Run tests for reproducibility of TensorFlow."""

    def test_version(self):
        self.assertTrue(tf.__version__ == "1.12.0" or
                        tf.__version__.startswith("2.0.0-dev2019"))

    def type_helper(self, eager):

        # Skip non-eager mode in TensorFlow 2,
        # which is not yet documented
        if tensorflow2_noneager(eager):
            return
        
        num = generate_random_number_1_seed(eager)
        num_type = num.dtype

        # conditional with `eager`, not with `tf.executing_eagerly()`
        # because the latter is always True and set in the bottom of the script
        # and non-eager mode is defined inside the call `generate_random_number_1_seed()`
        # with a local session
        if eager:
            self.assertEqual(num_type, tf.float32)
        else:
            self.assertEqual(num_type, np.float32)

    def test_type_eager(self):
        self.type_helper(eager = True)

    def test_type_non_eager(self):
        self.type_helper(eager = False)
    
    def random_number_1_seed_helper(self, eager):

        # Skip non-eager mode in TensorFlow 2,
        # which is not yet documented
        if tensorflow2_noneager(eager):
            return

        num = generate_random_number_1_seed(eager)
        num_str = format_number(num)

        if eager:
            expected_number = "1.511062622070"
        else:
            expected_number = "-1.409554481506"
            
        self.assertEqual(num_str, expected_number)

    def test_random_number_1_seed_eager(self):
        self.random_number_1_seed_helper(eager = True)
        
    def test_random_number_1_seed_non_eager(self):
        self.random_number_1_seed_helper(eager = False)
        
    def random_number_2_seeds_helper(self, eager):

        # Skip non-eager mode in TensorFlow 2,
        # which is not yet documented
        if tensorflow2_noneager(eager):
            return

        num = generate_random_number_2_seeds(eager)
        num_str = format_number(num)
        self.assertEqual(num_str, "0.680345416069")
    
    def test_random_number_2_seeds_eager(self):
        self.random_number_2_seeds_helper(eager = True)
        
    def test_random_number_2_seeds_non_eager(self):
        self.random_number_2_seeds_helper(eager = False)

    def arithmetic_1_seed_helper(self, eager):

        # Skip non-eager mode in TensorFlow 2,
        # which is not yet documented
        if tensorflow2_noneager(eager):
            return
        
        mean, sd = generate_stats_1_seed(eager)

        # Expected means
        if 2 == tf_major_version():
            if "azure" == current_system():
                expected_mean = "0.000620655250"
            elif current_system() in ["colab", "macOS"]:
                expected_mean = "-0.008264398202"
            else:
                assert False, "Not tested: " + current_system()
        else:
            if not azure():
                if eager:
                    expected_mean = "-0.008264393546"
                else:
                    expected_mean = "0.001438469742"
            else:
                if eager:
                    expected_mean = "-0.008264395408"
                else:
                    if tf.test.is_gpu_available():
                        expected_mean = "0.001438470092"
                    else:
                        expected_mean = "0.001438470441"


        # Expected standard deviations
        if 2 == tf_major_version():
            expected_sd = "0.995371103287"
        else:
            if eager:
                expected_sd = "0.995371103287"
            if not eager:
                if not azure():
                    expected_sd = "0.996351540089"
                else:
                    if tf.test.is_gpu_available():
                        expected_sd = "0.996351540089"
                    else:
                        expected_sd = "0.996351480484"
                    

        self.assertEqual(mean, expected_mean)
        self.assertEqual(sd, expected_sd)

    def test_arithmetic_1_seed_eager(self):
        self.arithmetic_1_seed_helper(eager = True)
        
    def test_arithmetic_1_seed_non_eager(self):
        self.arithmetic_1_seed_helper(eager = False)

    def arithmetic_2_seeds_helper(self, eager):

        # Skip non-eager mode in TensorFlow 2,
        # which is not yet documented
        if tensorflow2_noneager(eager):
            return

        mean, sd = generate_stats_2_seeds(eager)

        if 2 == tf_major_version():
            expected_mean = "0.000620646286"
        else:
            if not azure():
                expected_mean = "0.000620653736"
            else:
                if not tf.test.is_gpu_available():
                    expected_mean = "0.000620655250"
                else:
                    if eager:
                        expected_mean = "0.000620648789"
                    else:
                        expected_mean = "0.000620654318"
            
        if 2 == tf_major_version():
            expected_sd = "0.997191071510"
        else:
            if tf.test.is_gpu_available():
                expected_sd = ["0.997191190720", "0.997191071510"]
            else:
                expected_sd = "0.997191190720"

        self.assertEqual(mean, expected_mean)

        if str == type(expected_sd):
            self.assertEqual(sd, expected_sd)
        else:
            self.assertTrue(sd in expected_sd)

    def test_arithmetic_2_seeds_eager(self):
        self.arithmetic_2_seeds_helper(eager = True)

    def test_arithmetic_2_seeds_non_eager(self):
        self.arithmetic_2_seeds_helper(eager = False)

if __name__ == '__main__':

    # Syntax specific to TensorFlow 1
    if 1 == tf_major_version():
        tf.reset_default_graph()
        tf.enable_eager_execution() # this will not be valid when starting a new session
        tf.logging.set_verbosity(tf.logging.ERROR)

    # For compatibility with Google Colab, we do not run `tf.test.main()`, but
    # these lines instead:
    import unittest
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=2)


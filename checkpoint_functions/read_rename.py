import tensorflow as tf
import numpy as np
import math

def read_rename(checkpoint_dir = None, rename_dictionary = {}, mode = 'rename', dry_run= False):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):

            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            if mode == 'read':
                print('Variable name is {} with size {}'.format(var_name,np.shape(var)))

            elif mode == 'rename':
                if dry_run:
                    print('This is just a dry run')
                    for k,v in rename_dictionary.items():
                        print('Variable {} will be replaced by {}'.format(k,v))
                else:
                    if var_name in rename_dictionary:
                        # Make changes to the var value if necessary 
                        # Example 
                        # var = 63*[math.sqrt(2. / (49))]
                        # var = np.asarray(var,dtype=np.float32)
                        var = tf.Variable(var, name=rename_dictionary[var_name])

            else:
                print('ERROR : Wrong mode entered. It should either be read or rename')
              
 
if __name__ == "__main__":
    """
    To read/rename variables in a checkpoint file
    parameter checkpoint_dir : path to the checkpoint file
    parameter rename_dictionary : A dictionary file with keys as original name and value as the new name
    paramter mode : Either read existing variables or rename variables
    paramter dry_run : True if you want to perform a trial of renaming variables
    """
    read_rename(checkpoint_dir = './model.ckpt-6000', rename_dictionary = {}, mode = 'read', dry_run= True)

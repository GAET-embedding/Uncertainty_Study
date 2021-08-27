from __future__ import print_function
import os
import time
import shutil
import torch
import dill

class Checkpoint(object):
    CHECKPOINT_DIR_NAME = 'checkpoints'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'
    PARAMETERS = 'parameters.txt'

    def __init__(self, model, optimizer, epoch, eval_results=None, path=None):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self._path = path
        self.eval_results = eval_results

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    def save(self, experiment_dir):
        """
        Saves the current model and related training parameters into a subdirectory of the checkpoint directory.
        The name of the subdirectory is the current local time in Y_M_D_H_M_S format.
        Args:
            experiment_dir (str): path to the experiment root directory
        Returns:
             str: path to the saved checkpoint subdirectory
        """
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

        if self.eval_results is not None:
            # print(self.eval_results)
            assert isinstance(self.eval_results, dict)
            # present the dict in str form
            # res_str = ''.join(''.join(str(x) for x in tup) for tup in self.eval_results.items())

        self._path = os.path.join(
            experiment_dir, self.CHECKPOINT_DIR_NAME, date_time,
        )
        path = self._path

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        torch.save(
            {'epoch': self.epoch, 'optimizer': self.optimizer},
            os.path.join(path, self.TRAINER_STATE_NAME)
        )
        torch.save(self.model, os.path.join(path, self.MODEL_NAME))

        # save parameters to txt
        txt_file = open(os.path.join(path, self.PARAMETERS), "w")

        txt_file.write(f"ckpt name: '{date_time}'\n")
        txt_file.write(f"epoch: {self.epoch}\n")

        if self.eval_results is not None: 
            for key, value in self.eval_results.items():
                txt_file.write(str(key)+': '+str(value)+'\n')
            # if 'acc' in self.eval_results:
            #     txt_file.write(f"acc: {self.eval_results['acc']}\n")
            # if 'p' in self.eval_results:
            #     txt_file.write(f"p: {self.eval_results['p']}\n")
            # if 'r' in self.eval_results:
            #     txt_file.write(f"r: {self.eval_results['r']}\n")
            # if 'f1' in self.eval_results:
            #     txt_file.write(f"f1: {self.eval_results['f1']}\n")
        
        txt_file.close()

        return path

    @classmethod
    def load(cls, path):
        """
        Loads a Checkpoint object that was previously saved to disk.
        Args:
            path (str): path to the checkpoint subdirectory
        Returns:
            checkpoint (Checkpoint): checkpoint object with fields copied from those stored on disk
        """
        print(f'load checkpoint from {path}')
        if torch.cuda.is_available():
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME))
            model = torch.load(os.path.join(path, cls.MODEL_NAME))
        else:
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME), map_location=lambda storage, loc: storage)
            model = torch.load(os.path.join(path, cls.MODEL_NAME), map_location=lambda storage, loc: storage)
        
        # model.flatten_parameters() # make RNN parameters contiguous
        optimizer = resume_checkpoint['optimizer']
        return Checkpoint(
            model=model, 
            optimizer=optimizer,
            epoch=resume_checkpoint['epoch'],
            path=path
        )

    @classmethod
    def get_latest_checkpoint(cls, experiment_path):
        """
        Given the path to an experiment directory, returns the path to the last saved checkpoint's subdirectory.

        Precondition: at least one checkpoint has been made (i.e., latest checkpoint subdirectory exists).
        Args:
            experiment_path (str): path to the experiment directory
        Returns:
             str: path to the last saved checkpoint's subdirectory
        """
        checkpoints_path = os.path.join(experiment_path, cls.CHECKPOINT_DIR_NAME)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])

    @classmethod
    def get_oldest_checkpoint(cls, experiment_path):
        """
        Given the path to an experiment directory, returns the path to the last saved checkpoint's subdirectory.

        Precondition: at least one checkpoint has been made (i.e., latest checkpoint subdirectory exists).
        Args:
            experiment_path (str): path to the experiment directory
        Returns:
             str: path to the last saved checkpoint's subdirectory
        """
        checkpoints_path = os.path.join(experiment_path, cls.CHECKPOINT_DIR_NAME)
        all_times = sorted(os.listdir(checkpoints_path))
        return os.path.join(checkpoints_path, all_times[0])
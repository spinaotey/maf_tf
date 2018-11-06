import tensorflow as tf
import numpy as np
import numpy.random as rng
import os

class idx_streamer:
    """
    Index streaming class to get randomized batch size indices samples in a uniform way (using epochs).
    """
    def __init__(self,N,batch_size):
        """
        Constructor defining streamer parameters.
        :param N: total number of samples.
        :param batch_size: batch size that the streamer has to generate.
        """
        self.N = N
        self.sequence = np.arange(N)
        self.batch_size = batch_size
        self.stream = []
        self.epoch = -1
    
    def gen(self):
        """
        Index stream generation function. Outputs next batch indices.
        :return: List of batch indices.
        """
        while len(self.stream) < self.batch_size:
            rng.shuffle(self.sequence)
            self.stream += list(self.sequence)
            self.epoch +=1
        stream = self.stream[:self.batch_size]
        self.stream = self.stream[self.batch_size:]
        return stream

class Trainer:
    """
    Training class for the standard MADEs/MAFs classes using a tensorflow optimizer.
    """
    def __init__(self, model, optimizer=tf.train.AdamOptimizer, optimizer_arguments={}):
        """
        Constructor that defines the training operation.
        :param model: made/maf instance to be trained.
        :param optimizer: tensorflow optimizer class to be used during training.
        :param optimizer_arguments: dictionary of arguments for optimizer intialization.
        """
        
        self.model = model
        
        if hasattr(self.model,'batch_norm') and self.model.batch_norm is True:
            self.has_batch_norm = True
        else:
            self.has_batch_norm = False
        self.train_op = optimizer(**optimizer_arguments).minimize(self.model.trn_loss)
            
            
    def train(self, sess, train_data, val_data=None, p_val = 0.05, max_iterations=1000, batch_size=100,
              early_stopping=20, check_every_N=5, saver_name='tmp_model', show_log=False):
        """
        Training function to be called with desired parameters within a tensorflow session.
        :param sess: tensorflow session where the graph is run.
        :param train_data: train data to be used.
        :param val_data: validation data to be used for early stopping. If None, train_data is splitted 
             into p_val percent for validation randomly.  
        :param p_val: percentage of training data randomly selected to be used for validation if
             val_data is None.
        :param max_iterations: maximum number of iterations for training.
        :param batch_size: batch size in each training iteration.
        :param early_stopping: number of iterations for early stopping criteria.
        :param check_every_N: check every N iterations if model has improved and saves if so.
        :param saver_name: string of name (with or without folder) where model is saved. If none is given,
            a temporal model is used to save and restore best model, and removed afterwards.
        """
        
        train_idx = np.arange(train_data.shape[0])
        
        # If no validation data was found, split training into training and 
        # validation data using p_val percent of the data
        if val_data == None:
            rng.shuffle(train_idx)
            val_data = train_data[train_idx[-int(p_val*train_data.shape[0]):]]
            train_data = train_data[train_idx[:-int(p_val*train_data.shape[0])]]
            train_idx = np.arange(train_data.shape[0])
        
        # Early stopping variables
        bst_loss = np.infty
        early_stopping_count = 0
        saver = tf.train.Saver()
        
        # Batch index streamer
        streamer = idx_streamer(train_data.shape[0],batch_size)
        
        # Main training loop
        for iteration in range(max_iterations):
            batch_idx = streamer.gen()
            if self.has_batch_norm:
                sess.run(self.train_op,feed_dict={self.model.input:train_data[batch_idx],self.model.training:True})
            else:
                sess.run(self.train_op,feed_dict={self.model.input:train_data[batch_idx]})
            # Early stopping check
            if iteration%check_every_N == 0:
                if self.has_batch_norm:
                    self.model.update_batch_norm(train_data,sess)
                this_loss = sess.run(self.model.trn_loss,feed_dict={self.model.input:val_data})
                if show_log:
                    train_loss = sess.run(self.model.trn_loss,feed_dict={self.model.input:train_data})
                    print("Iteration {:05d}, Train_loss: {:05.4f}, Val_loss: {:05.4f}".format(iteration,train_loss,this_loss))
                if this_loss < bst_loss:
                    bst_loss = this_loss
                    saver.save(sess,"./"+saver_name)
                    early_stopping_count = 0
                else:
                    early_stopping_count += check_every_N
            if early_stopping_count >= early_stopping:
                break
                
        if show_log:
            print("Training finished")
            print("Best Iteration {:05d}, Val_loss: {:05.4f}".format(iteration-early_stopping,bst_loss))
        # Restore best model and save batch norm mean and variance if necessary
        saver.restore(sess,"./"+saver_name)
        if self.has_batch_norm:
            self.model.update_batch_norm(train_data,sess)
        
        # Remove model data if temporal model data was used
        if saver_name == 'tmp_model':
            for file in os.listdir("./"):
                if file[:len(saver_name)] == saver_name:
                    os.remove(file)
                    
                    
class ConditionalTrainer(Trainer):
    """
    Training class for the conditional MADEs/MAFs classes using a tensorflow optimizer.
    """           
    def train(self, sess, train_data, val_data=None, p_val = 0.05, max_iterations=1000, batch_size=100,
              early_stopping=20, check_every_N=5, saver_name='tmp_model', show_log=False):
        """
        Training function to be called with desired parameters within a tensorflow session.
        :param sess: tensorflow session where the graph is run.
        :param train_data: a tuple/list of (X,Y) with training data where Y is conditioned on X.
        :param val_data: a tuple/list of (X,Y) with validation data where Y is conditioned on X to be 
            used for early stopping. If None, train_data is splitted into p_val percent for validation
            randomly.  
        :param p_val: percentage of training data randomly selected to be used for validation if
             val_data is None.
        :param max_iterations: maximum number of iterations for training.
        :param batch_size: batch size in each training iteration.
        :param early_stopping: number of iterations for early stopping criteria.
        :param check_every_N: check every N iterations if model has improved and saves if so.
        :param saver_name: string of name (with or without folder) where model is saved. If none is given,
            a temporal model is used to save and restore best model, and removed afterwards.
        :param show_log: boolean if showing training evolution or not.
        """
        
        train_data_X, train_data_Y  = train_data
        train_idx = np.arange(train_data_X.shape[0])
        
        # If no validation data was found, split training into training and 
        # validation data using p_val percent of the data
        if val_data == None:
            rng.shuffle(train_idx)
            N = train_data_X.shape[0]
            val_data_X = train_data_X[train_idx[-int(p_val*N):]]
            train_data_X = train_data_X[train_idx[:-int(p_val*N)]]
            val_data_Y = train_data_Y[train_idx[-int(p_val*N):]]
            train_data_Y = train_data_Y[train_idx[:-int(p_val*N)]]
            train_idx = np.arange(train_data_X.shape[0])
        else:
            val_data_X, val_data_Y = val_data
            
        # Early stopping variables
        bst_loss = np.infty
        early_stopping_count = 0
        saver = tf.train.Saver()
        
        # Batch index streamer
        streamer = idx_streamer(train_data_X.shape[0],batch_size)
        
        # Main training loop
        for iteration in range(max_iterations):
            batch_idx = streamer.gen()
            if self.has_batch_norm:
                sess.run(self.train_op,feed_dict={self.model.input:train_data_X[batch_idx],
                                                  self.model.y:train_data_Y[batch_idx],
                                                  self.model.training:True})
            else:
                sess.run(self.train_op,feed_dict={self.model.input:train_data_X[batch_idx],
                                                  self.model.y:train_data_Y[batch_idx]})
            # Early stopping check
            if iteration%check_every_N == 0:
                if self.has_batch_norm:
                    self.model.update_batch_norm([train_data_X,train_data_Y],sess)
                this_loss = sess.run(self.model.trn_loss,feed_dict={self.model.input:val_data_X,
                                                                    self.model.y:val_data_Y})
                if show_log:
                    train_loss = sess.run(self.model.trn_loss,feed_dict={self.model.input:train_data_X,
                                                                         self.model.y:train_data_Y})
                    print("Iteration {:05d}, Train_loss: {:05.4f}, Val_loss: {:05.4f}".format(iteration,train_loss,this_loss))
                if this_loss < bst_loss:
                    bst_loss = this_loss
                    saver.save(sess,"./"+saver_name)
                    early_stopping_count = 0
                else:
                    early_stopping_count += check_every_N
            if early_stopping_count >= early_stopping:
                break
        if show_log:
            print("Training finished")
            print("Best iteration {:05d}, Val_loss: {:05.4f}".format(iteration-early_stopping,bst_loss))
        # Restore best model  and save batch norm mean and variance if necessary
        saver.restore(sess,"./"+saver_name)
        if self.has_batch_norm:
                    self.model.update_batch_norm([train_data_X,train_data_Y],sess)
        # Remove model data if temporal model data was used
        if saver_name == 'tmp_model':
            for file in os.listdir("./"):
                if file[:len(saver_name)] == saver_name:
                    os.remove(file)

class WeightedTrainer(Trainer):
    """
    Training class for the conditional MADEs/MAFs classes using a tensorflow optimizer.
    """           
    def train(self, sess, train_data, weights, val_data=None, p_val = 0.05, max_iterations=1000, batch_size=100,
              early_stopping=20, check_every_N=5, saver_name='tmp_model', show_log=False):
        """
        Training function to be called with desired parameters within a tensorflow session.
        :param sess: tensorflow session where the graph is run.
        :param train_data: a tuple/list of (X,Y) with training data where Y is conditioned on X.
        :param val_data: a tuple/list of (X,Y) with validation data where Y is conditioned on X to be 
            used for early stopping. If None, train_data is splitted into p_val percent for validation
            randomly.  
        :param p_val: percentage of training data randomly selected to be used for validation if
             val_data is None.
        :param max_iterations: maximum number of iterations for training.
        :param batch_size: batch size in each training iteration.
        :param early_stopping: number of iterations for early stopping criteria.
        :param check_every_N: check every N iterations if model has improved and saves if so.
        :param saver_name: string of name (with or without folder) where model is saved. If none is given,
            a temporal model is used to save and restore best model, and removed afterwards.
        :param show_log: boolean if showing training evolution or not.
        """
        train_idx = np.arange(train_data.shape[0])
        
        # If no validation data was found, split training into training and 
        # validation data using p_val percent of the data
        if val_data == None:
            rng.shuffle(train_idx)
            N = train_data.shape[0]
            val_data = train_data[train_idx[-int(p_val*N):]]
            train_data = train_data[train_idx[:-int(p_val*N)]]
            train_weights = weights[train_idx[:-int(p_val*N)]]
            val_weights = weights[train_idx[-int(p_val*N):]]
            train_idx = np.arange(train_data.shape[0])
        
        # Early stopping variables
        bst_loss = np.infty
        early_stopping_count = 0
        saver = tf.train.Saver()
        
        # Batch index streamer
        streamer = idx_streamer(train_data.shape[0],batch_size)
        
        # Main training loop
        for iteration in range(max_iterations):
            batch_idx = streamer.gen()
            if self.has_batch_norm:
                sess.run(self.train_op,feed_dict={self.model.input:train_data[batch_idx],self.model.training:True,
                                                  self.model.weights:train_weights[batch_idx]})
            else:
                sess.run(self.train_op,feed_dict={self.model.input:train_data[batch_idx],
                                                  self.model.weights:train_weights[batch_idx]})
            # Early stopping check
            if iteration%check_every_N == 0:
                if self.has_batch_norm:
                    self.model.update_batch_norm(train_data,sess)
                this_loss = sess.run(self.model.trn_loss,feed_dict={self.model.input:val_data,
                                                                    self.model.weights:val_weights})
                if show_log:
                    train_loss = sess.run(self.model.trn_loss,feed_dict={self.model.input:train_data,
                                                                         self.model.weights:train_weights})
                    print("Iteration {:05d}, Train_loss: {:05.4f}, Val_loss: {:05.4f}".format(iteration,train_loss,this_loss))
                if this_loss < bst_loss:
                    bst_loss = this_loss
                    saver.save(sess,"./"+saver_name)
                    early_stopping_count = 0
                else:
                    early_stopping_count += check_every_N
            if early_stopping_count >= early_stopping:
                break
                
        if show_log:
            print("Training finished")
            print("Best Iteration {:05d}, Val_loss: {:05.4f}".format(iteration-early_stopping,bst_loss))
        # Restore best model and save batch norm mean and variance if necessary
        saver.restore(sess,"./"+saver_name)
        if self.has_batch_norm:
            self.model.update_batch_norm(train_data,sess)
        
        # Remove model data if temporal model data was used
        if saver_name == 'tmp_model':
            for file in os.listdir("./"):
                if file[:len(saver_name)] == saver_name:
                    os.remove(file)

                    

class WeightedConditionalTrainer(Trainer):
    """
    Training class for the conditional MADEs/MAFs classes using a tensorflow optimizer.
    """           
    def train(self, sess, train_data, weights, val_data=None, p_val = 0.05, max_iterations=1000, batch_size=100,
              early_stopping=20, check_every_N=5, saver_name='tmp_model', show_log=False):
        """
        Training function to be called with desired parameters within a tensorflow session.
        :param sess: tensorflow session where the graph is run.
        :param train_data: a tuple/list of (X,Y) with training data where Y is conditioned on X.
        :param val_data: a tuple/list of (X,Y) with validation data where Y is conditioned on X to be 
            used for early stopping. If None, train_data is splitted into p_val percent for validation
            randomly.  
        :param p_val: percentage of training data randomly selected to be used for validation if
             val_data is None.
        :param max_iterations: maximum number of iterations for training.
        :param batch_size: batch size in each training iteration.
        :param early_stopping: number of iterations for early stopping criteria.
        :param check_every_N: check every N iterations if model has improved and saves if so.
        :param saver_name: string of name (with or without folder) where model is saved. If none is given,
            a temporal model is used to save and restore best model, and removed afterwards.
        :param show_log: boolean if showing training evolution or not.
        """
        
        train_data_X, train_data_Y  = train_data
        train_idx = np.arange(train_data_X.shape[0])
        
        # If no validation data was found, split training into training and 
        # validation data using p_val percent of the data
        if val_data == None:
            rng.shuffle(train_idx)
            N = train_data_X.shape[0]
            val_data_X = train_data_X[train_idx[-int(p_val*N):]]
            train_data_X = train_data_X[train_idx[:-int(p_val*N)]]
            val_data_Y = train_data_Y[train_idx[-int(p_val*N):]]
            train_data_Y = train_data_Y[train_idx[:-int(p_val*N)]]
            train_weights = weights[train_idx[:-int(p_val*N)]]
            val_weights = weights[train_idx[-int(p_val*N):]]
            train_idx = np.arange(train_data_X.shape[0])
        else:
            val_data_X, val_data_Y = val_data
            
        # Early stopping variables
        bst_loss = np.infty
        early_stopping_count = 0
        saver = tf.train.Saver()
        
        # Batch index streamer
        streamer = idx_streamer(train_data_X.shape[0],batch_size)
        
        # Main training loop
        for iteration in range(max_iterations):
            batch_idx = streamer.gen()
            if self.has_batch_norm:
                sess.run(self.train_op,feed_dict={self.model.input:train_data_X[batch_idx],
                                                  self.model.y:train_data_Y[batch_idx],
                                                  self.model.weights:train_weights[batch_idx],
                                                  self.model.training:True})
            else:
                sess.run(self.train_op,feed_dict={self.model.input:train_data_X[batch_idx],
                                                  self.model.y:train_data_Y[batch_idx],
                                                  self.model.weights:train_weights[batch_idx]})
            # Early stopping check
            if iteration%check_every_N == 0:
                if self.has_batch_norm:
                    self.model.update_batch_norm([train_data_X,train_data_Y],sess)
                this_loss = sess.run(self.model.trn_loss,feed_dict={self.model.input:val_data_X,
                                                                    self.model.y:val_data_Y,
                                                                    self.model.weights:val_weights})
                if show_log:
                    train_loss = sess.run(self.model.trn_loss,feed_dict={self.model.input:train_data_X,
                                                                         self.model.y:train_data_Y,
                                                                         self.model.weights:train_weights})
                    print("Iteration {:05d}, Train_loss: {:05.4f}, Val_loss: {:05.4f}".format(iteration,train_loss,this_loss))
                if this_loss < bst_loss:
                    bst_loss = this_loss
                    saver.save(sess,"./"+saver_name)
                    early_stopping_count = 0
                else:
                    early_stopping_count += check_every_N
            if early_stopping_count >= early_stopping:
                break
        if show_log:
            print("Training finished")
            print("Best iteration {:05d}, Val_loss: {:05.4f}".format(iteration-early_stopping,bst_loss))
        # Restore best model  and save batch norm mean and variance if necessary
        saver.restore(sess,"./"+saver_name)
        if self.has_batch_norm:
                    self.model.update_batch_norm([train_data_X,train_data_Y],sess)
        # Remove model data if temporal model data was used
        if saver_name == 'tmp_model':
            for file in os.listdir("./"):
                if file[:len(saver_name)] == saver_name:
                    os.remove(file)
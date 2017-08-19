
import numpy as np

class Dataset:

    def __init__(self,data, label):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._label = label
        self._num_examples = data.shape[0]

        pass

    @property
    def data(self):
        
        return self._data

    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self._data[idx]  # get list of `num` random samples
            self._label = self._label[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            label_rest_part = self._label[start: self._num_examples]

            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self._data[idx0]  # get list of `num` random samples
            self._label = self._label[idx0]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]  
            label_new_part = self._label[start:end]
            data = np.concatenate((data_rest_part, data_new_part), axis=0)
            label = np.concatenate((label_rest_part, label_new_part), axis = 0)
            neg_data, neg_label = self.get_negative(batch_size, label )

            return data, label, neg_data, neg_label
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            data, label =  self._data[start:end], self._label[start:end]
            neg_data, neg_label = self.get_negative(batch_size, label )
            return data, label, neg_data, neg_label


    # random generate negative samples 
    # may be take a long time 
    def get_negative(self, batch_size, labels):

        fake_ids = np.random.randint(self._num_examples, size=batch_size)
        self._label[fake_ids]
        collision_flag = range(batch_size)
        while True:
            collision_flag =\
                np.array( np.where( np.sum( abs( self._label[fake_ids] - labels), axis = 1) == 0)[0])
            if len( collision_flag) == 0:
                break
            fake_ids[collision_flag] = np.random.randint(self._num_examples, size = len(collision_flag))
        return self._data[fake_ids], self._label[fake_ids]
    def get_image_by_index(self, label, b_size = 1):
        new_label = np.argmax(self._label, axis = 1)
        idxs = np.where(new_label == label)[0]
        idxs = np.random.shuffle(idxs)[: b_size]
        return self._data[idxs]
    def get_all_images(self, batch_size):
        label_dim = self._label.shape[1]
        new_label = np.argmax(self._label, axis = 1)
        imgs = []
        labels = []
        for i in range(batch_size):
            i = i % label_dim
            idxs = np.where(new_label == i)[0]
            assert len(idxs) > 0
            imgs.append(self._data[idxs[0], :])
            labels.append(self._label[idxs[0]])
        imgs = np.array(imgs)
        labels = np.array(labels)
        return imgs, labels

        
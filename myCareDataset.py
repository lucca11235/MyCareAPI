from dataclasses import dataclass
import numpy as np

@dataclass
class window:
  data: np.array
  origin_file: str
  packet: int
  label: int


class myCareDataset:
    def __init__(self, directories):
        self.directories = directories
        self.data = None
        self.fs = 100

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

    def _load(self, f):
        try:
            d = np.load(f)['arr_0']
        except:
            d = np.load(f)
        return d.T


    def window_data(self, window_size: int, jump: int):
        window_size = window_size*self.fs
        jump = jump*self.fs

        for f,d in self.data.items():
          subcarriers, packets = d.shape
          n_windows = 1 + (packets - window_size) // jump

          # Calculate indices for the window starts and extract the windows
          window_indices = np.arange(window_size//2, n_windows * jump - window_size//2, jump)
          windows = np.array([window(d[:,i:i + window_size],f,i,-1) for i in window_indices])
          self.data[f] = windows

    def apply_pipeline(self,pipeline):
      f = self.directories[0]

      if isinstance(self.data[f][0], np.ndarray):
        for f in tqdm(self.directories):
          self.data[f] = pipeline(self.data[f])

      if isinstance(self.data[f][0], window):
        for f in tqdm(self.directories):
          for w in self.data[f]:
            w.data = pipeline(w.data)



    def load_data(self):
        data = {}
        for f in tqdm(self.directories):
            try:
                data[f] = self._load(f)
            except Exception as E:
                continue

        print(f"\nNumber of files loaded: {len(data)}")
        self.data = data


    def create_dataset(self,
                       p = 0.75,
                       mode = 0):

      if not 0 <= p <= 100:
            raise ValueError("'p' must be between 0 and 100.")


      if mode == 0:

        train_a_directories, test_b_directories = self._train_a_test_b_split_directories(p)

        X_train = self._create_X_array(train_a_directories)
        X_test = self._create_X_array(test_b_directories)

        y_train = self._create_y_array(train_a_directories)
        y_test = self._create_y_array(test_b_directories)

        return X_train ,X_test ,y_train ,y_test

      if isinstance(mode,list):
        train_a_directories, test_b_directories = self._train_a_test_b_split_directories(p)

        X_train = self._create_X_array(train_a_directories)
        X_test = self._create_X_array(test_b_directories)

        y_train = self._create_y_array(train_a_directories, mode = mode)
        y_test = self._create_y_array(test_b_directories,mode = mode)

        return X_train ,X_test ,y_train ,y_test


    def _train_a_test_b_split_directories(self, p):
        files = list(self.data.keys())
        n_elements_to_select = int(len(files) * p)
        train_a_directories = random.sample(files, n_elements_to_select)
        test_b_directories = [f for f in files if f not in train_a_directories]
        return train_a_directories, test_b_directories

    def _create_X_array(self, directories):
        return np.array([self.data[f][i].data for f in directories for i in range(len(self.data[f]))])

    def _create_y_array(self, directories,mode = 0):
        numWindows = len(self.data[self.directories[0]])
        if mode == 0:
          return np.array([0 if self.data[f][i].packet < 9200 else 1 for f in directories for i in range(numWindows)])

        if isinstance(mode,list):
          return np.array([1 if self.data[f][i].origin_file in mode else 0 for f in directories for i in range(numWindows)])


    def decimate(self):
      f = self.directories[0]

      if isinstance(self.data[f][0], np.ndarray):
        for f in self.directories:
          self.data[f] = self.data[f][:,::10]

      if isinstance(self.data[f][0], window):
        for f in self.directories:
          for w in self.data[f]:
            w.data = w.data[:,::10]

      self.fs = 10

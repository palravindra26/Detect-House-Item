# Import required modules
import os
import shutil

from openimages.download import download_images

from PIL import Image

import pandas as pd

class DownloadOI:
  def __init__(self, classes, data_path, csv_path, limit=None):
    '''
    params :
      classes : list of all the required labels need to download
      data_path : path where the data files need to be stored
      csv_path : path where the csv files are stored
    '''

    self.classes = classes
    self.data_path = self.get_path(data_path)
    self.csv_path = self.get_path(csv_path)
    self.limit = limit

  def create_classes_csv(self):
    '''
    This function will create a csv file containing
    class name along with their labels
    '''
    os.mkdir(f'/{self.data_path}/classes')

    classes_dict = {'class_name':[], 'id':[]}

    for id, name in enumerate(self.classes):
      classes_dict['class_name'].append(name)
      classes_dict['id'].append(id)

    classes_df = pd.DataFrame(classes_dict)
    classes_df.to_csv(f'/{self.data_path}/classes/classes.csv',header=False , index=False)

  def download_OI_images(self):
    '''
    This function will download the images from open images since the
    openimages modules download images in 'data_path/images/class_name/images'
    this function will copy all the images from class_name/images to
    data_path/images and delete class_name folder
    '''

    #Download Data
    download_images(f"/{self.data_path}/images", self.classes, None, csv_dir=self.csv_path, limit=self.limit)

    dest_path = f'/{self.data_path}/images/'

    #Copy files from data_path/images/class_name/images to data_path/images/
    for class_name in self.classes:
      for file in os.listdir(f"/{self.data_path}/images/{class_name.lower()}/images"):
        if os.path.isfile(f'/{self.data_path}/images/{file}'):
          continue
        src_path = f"/{self.data_path}/images/{class_name.lower()}/images/{file}"
        shutil.move(src_path, dest_path)

    #Delete data_path/images/class_name/ folders
    for class_name in self.classes:
      shutil.rmtree(f'/{self.data_path}/images/{class_name.lower()}/')

  def form_annotation_file(self, df, save_dest=None, name='file_name'):
    '''
    This function form annotation csv file as required by keras-retina.
    Params:
      df: pandas Dataframe
      save_dest: path where to store the csv file
      name: name of the file

    Note:
      if save_dest is None this function will return pandas DataFrame
    '''

    annotation = {
      'location': [],
      'x1': [],
      'y1': [],
      'x2': [],
      'y2': [],
      'class_name': []
    }

    id_to_class, id_list = self.create_dict_id_class()
    name_of_images = self.images_in_dir()

    new_df = df[df['LabelName'].isin(id_list)].reset_index()
    new_df = new_df[new_df['ImageID'].isin(name_of_images)].reset_index()

    for i in range(new_df.shape[0]):
      image_id = new_df.loc[i]['ImageID']
      id_name = new_df.loc[i]['LabelName']

      class_name = id_to_class[id_name]
      path = f'/{self.data_path}/images/{image_id}.jpg'

      x1 = new_df.loc[i]['XMin']
      x2 = new_df.loc[i]['XMax']
      y1 = new_df.loc[i]['YMin']
      y2 = new_df.loc[i]['YMax']

      with Image.open(path) as img:
        width, height = img.width, img.height

      x1 = int(round(x1 * width))
      x2 = int(round(x2 * width))
      y1 = int(round(y1 * height))
      y2 = int(round(y2 * height))

      annotation['location'].append(path)
      annotation['x1'].append(x1)
      annotation['x2'].append(x2)
      annotation['y1'].append(y1)
      annotation['y2'].append(y2)
      annotation['class_name'].append(class_name)

    if save_dest==None:
      return pd.DataFrame(annotation)
    else:
      df = pd.DataFrame(annotation)
      df.to_csv(save_dest+'/'+name+'.csv', header=False, index=False)

  def create_annotations_csv(self):
    '''
    This function will create different annotation file for
    train, test and validation
    '''

    os.mkdir(f'/{self.data_path}/annotation')

    train_file = pd.read_csv(f'/{self.csv_path}/train-annotations-bbox.csv')
    test_file = pd.read_csv(f'/{self.csv_path}/test-annotations-bbox.csv')
    validation_file = pd.read_csv(f'/{self.csv_path}/validation-annotations-bbox.csv')

    save_path = f'/{self.data_path}/annotation'

    self.form_annotation_file(train_file, save_dest=save_path, name='train')
    self.form_annotation_file(test_file, save_dest=save_path, name='test')
    self.form_annotation_file(validation_file, save_dest=save_path, name='validation')

  def create_dict_id_class(self):
    '''
    This function will create and return a dictionary which contains
    id as key and class as values and list of ids
    '''

    class_desc = pd.read_csv(f'/{self.csv_path}/class-descriptions-boxable.csv', names=['id', 'name'])

    id_list = []
    for class_name in self.classes:
      id_list.append(class_desc[class_desc['name']==class_name]['id'].values[0])

    id_to_class = dict()
    for class_name, id_name  in zip(self.classes, id_list):
      id_to_class[id_name] = class_name

    return id_to_class, id_list

  def images_in_dir(self):
    name_of_images = []
    for class_name in self.classes:
      for file in os.listdir(f"/{self.data_path}/images/"):
        name_of_images.append(file[:-4])
    return name_of_images

  def get_path(self, path):
    if (path[0] == '/' and path[1] == '/') or (path[0] == '\\' and path[1] == '\\'):
      return path[1:-1]
    elif path[0] == '/' or path[0] == '\\':
        return path[1:]
    elif path[-1] == '/' or path[-1] == '\\':
        return path[:-1]
    else:
        return path

  def form_dataset(self):
    '''
    This function will create all the required files for the training
    '''

    self.download_OI_images()
    print('Images Downloaded')

    self.create_classes_csv()
    print('classes.csv file created!')

    self.create_annotations_csv()
    print('Annotations files created')

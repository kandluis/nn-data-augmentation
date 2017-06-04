import errno
import os
import shutil
import sys

import numpy as np

np.random.seed(231)

def makedir(path):
  try:
    os.makedirs(path)
  except OSError as exc: 
    if exc.errno == errno.EEXIST and os.path.isdir(path): pass
    else: raise

def subset_tiny_imagenet(input_path, output_path, nclasses=20, percent=0.5, flatten=True):
  """
  Copies a subset of the TinyImageNet datasets.
  Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
  TinyImageNet-200 have the same directory structure, so this can be used
  to load any of them. The function can then be used to copy a random sample
  of this subset. A random set of nclasses are copied and for each class
  percent data is kept.

  If flatten, it also outputs a top level images directory with all of
  our data in flattened format. This is to faciliate style transfers.

  Inputs:
  - intput_path: String giving path to the directory where the orignal data is located.
  - output_path: String giving path to the directory where we output the data.
  - nclasses: Integer specifiying the number of classes to keep in training data.
  - percent: Double specifiing the percentage of the data for each class to keep.
  Returns:
    Nothing. The output is written. Note that the script deletes everything in the
    given output directory if it exists.
  """
  # Delete output directory.
  if os.path.isdir(output_path):
    shutil.rmtree(output_path)

  # First load wnids
  with open(os.path.join(input_path, 'wnids.txt'), 'r') as f:
    wnids = [x.strip() for x in f]

  # select a random subset of the classes
  assert(nclasses <= len(wnids))
  wnids = np.random.choice(wnids, size=nclasses, replace=False)

  # Write the selected classes.
  makedir(output_path)
  with open(os.path.join(output_path, 'wnids.txt'), 'w') as f:
    for wnid in wnids:
      f.write("%s\n" % wnid)

  # Map wnids to integer labels
  wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

  # We can just copy 'words.txt' since it will be a superset.
  shutil.copyfile(os.path.join(input_path, 'words.txt'),
                  os.path.join(output_path, 'words.txt'))

  # Copy the train data over.
  for wnid in wnids:
    # To figure out the filenames we need to open the boxes file
    boxes_file = os.path.join(input_path, 'train', wnid, '%s_boxes.txt' % wnid)
    with open(boxes_file, 'r') as f:
      lines = [line.strip('\n') for line in f]
    num_images = len(lines)
    num_to_keep = 0 if percent == 0 else int(percent * num_images + 1)

    lines = np.random.choice(lines, size=num_to_keep, replace=False)

    # Write out the selected images to their own boxes files.
    output_class_path = os.path.join(output_path, 'train', wnid)
    makedir(output_class_path)
    boxes_file = os.path.join(output_class_path, '%s_boxes.txt' % wnid)
    with open(boxes_file, 'w') as f:
      for line in lines:
        f.write("%s\n" % line)

    # copy the selected images
    filenames = [line.split('\t')[0] for line in lines]
    for img_file in filenames:
      img_file_path = os.path.join(input_path, 'train', wnid, 'images', img_file)
      out_img_file_path = os.path.join(output_path, 'train', wnid, 'images')
      makedir(out_img_file_path)
      shutil.copyfile(img_file_path, os.path.join(out_img_file_path, img_file))
      if flatten:
        makedir(os.path.join(output_path, 'testA'))
        shutil.copyfile(img_file_path,
                        os.path.join(output_path,
                                     'testA',
                                     "+".join(['train', 'wnid', 'images', img_file])))

  # Next copy validation data but only for the classes we've selected.
  with open(os.path.join(input_path, 'val', 'val_annotations.txt'), 'r') as f:
    lines = [line.strip('\n') for line in f]

  # Only take the classes we've selected in training.
  lines = [line for line in lines if line.split('\t')[1] in wnid_to_label]
    
  num_images = len(lines)
  num_to_keep = 0 if percent == 0 else int(percent * num_images + 1)

  lines = np.random.choice(lines, size=num_to_keep, replace=False)

  # Write out the selected validation images.
  makedir(os.path.join(output_path, 'val'))
  with open(os.path.join(output_path, 'val', 'val_annotations.txt'), 'w') as f:
    for line in lines:
      f.write("%s\n" % line)

  # copy the selected images
  filenames = [line.split('\t')[0] for line in lines]
  for img_file in filenames:
    img_file_path = os.path.join(input_path, 'val', 'images', img_file)
    out_img_file_path = os.path.join(output_path, 'val', 'images')
    makedir(out_img_file_path)
    shutil.copyfile(img_file_path, os.path.join(out_img_file_path, img_file))
    if flatten:
        makedir(os.path.join(output_path, 'flat'))
        shutil.copyfile(img_file_path,
                        os.path.join(output_path,
                                     'flat',
                                     "+".join(['val', 'images', img_file])))

  # We skip the test since we don't have access to the labels.
  
if __name__ == '__main__':
  assert(len(sys.argv) == 3)
  input_path = sys.argv[1]
  output_path = sys.argv[2]
  print("Copying from %s to %s 100 classes and 0.2 of the data for each." % (input_path, output_path))
  subset_tiny_imagenet(input_path, output_path)
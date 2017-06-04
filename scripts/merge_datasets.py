import errno
import os
import shutil
import sys

def makedir(path):
  try:
    os.makedirs(path)
  except OSError as exc: 
    if exc.errno == errno.EEXIST and os.path.isdir(path): pass
    else: raise

def write_tiny_imagenet_set(inpath, outpath):
	"""
	Given a path containing the results of our data augmentation, recreates a training
	and validation set similar in format to the ImageNet.

	inpath: Directory of the lowest level 'results' folder for our augmented data.
	outpath: Directory where sets of data should be placed.
	"""
	paths = [os.path.join(inpath, d) for d in os.listdir(inpath)]
	dirnames = ["_".join(os.path.basename(path).split('_')[:-1]) for path in paths]
	for dirname, path in zip(dirnames, paths):
		if os.path.isdir(os.path.join(outpath, dirname)): 
			shutil.rmtree(os.path.join(outpath, dirname))

		imgs_path = os.path.join(path, 'latest_test', 'images', 'fake_B')
		images = os.listdir(imgs_path)
		for image in images:
			output_path = os.path.join(outpath, dirname, *image.split("+")[:-1])
			makedir(output_path)
			shutil.copyfile(os.path.join(imgs_path, image), os.path.join(output_path, image.split("+")[-1]))
		# Copy the metadata from the true train/val folders.
		img_metadata = [['train', nid, '%s_boxes.txt' % nid] for nid in os.listdir(os.path.join(outpath, 'our-imagenet-100', 'train'))] 
		for const in [['wnids.txt'], ['words.txt'], ['val', 'val_annotations.txt']] + img_metadata:
			shutil.copyfile(os.path.join(outpath, 'our-imagenet-100', *const),
											os.path.join(outpath, dirname, *const))

		print("Set %s is now done." % dirname)

if __name__ == '__main__':
	assert(len(sys.argv) == 3)
	inpath = sys.argv[1]
	outpath = sys.argv[2]

	write_tiny_imagenet_set(inpath, outpath)


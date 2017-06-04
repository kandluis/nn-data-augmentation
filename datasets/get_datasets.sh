# Get TinyImageNet data set
curl -LOk http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm tiny-imagenet-200.zip

# Copy our custom set of data to train on and run trials.
python extract_subset_data.py tiny-imagenet-200 our-imagenet-100

# Remove the original training data.
rm -rf tiny-imagenet-200/
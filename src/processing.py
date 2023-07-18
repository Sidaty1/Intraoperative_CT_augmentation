from nibabel.processing import resample_to_output
from nibabel import Nifti1Image, save, load
from numpy import array
from os.path import isfile, exists
import sys


# sanity check
assert len(sys.argv) == 7, 'Additionnal information is required'
for i in range(1, 4):
    assert isfile(isfile(sys.argv[i])), f"No such a file: {sys.argv[i]}"
assert isinstance(sys.argv[4], list), "The voxel size should be a list"
assert isinstance(sys.argv[5], float), "Down sampling scale should be a float"
assert exists(sys.argv[6]), f"No such a directory: {sys.argv[6]} to store downsampled data"
output_directory = sys.argv[6]

# load the volumes
without_contrast_volume = load(sys.argv[1])
with_contrast_volume = load(sys.argv[2])
segmentation = load(sys.argv[3])

# compute new voxel size
voxel_size = sys.argv[4]*sys.argv[5]

# down sample the data
without_contrast_downsampled = resample_to_output(without_contrast_volume, voxel_size)
with_contrast_downsampled = resample_to_output(with_contrast_volume, voxel_size)
segmentation_downsampled = resample_to_output(segmentation, voxel_size)

# get the rigid transformations
without_contrast_affine = without_contrast_downsampled.affine
without_contrast_data = without_contrast_downsampled.get_fdata()

with_contrast_affine = with_contrast_downsampled.affine
with_contrast_data = with_contrast_downsampled.get_fdata()

segmentation_affine = segmentation_downsampled.affine
segmentation_data = segmentation_downsampled.get_fdata()

# scale voxels between 0 and 1
without_contrast_data = (without_contrast_data - without_contrast_data.min())/(without_contrast_data.max() - without_contrast_data.min())*255
with_contrast_data = (with_contrast_data - with_contrast_data.min())/(with_contrast_data.max() - with_contrast_data.min())*255
segmentation_data = (segmentation_data - segmentation_data.min())/(segmentation_data.max() - segmentation_data.min())*255

# create and store the volumes
without_contrast_downsampled = Nifti1Image(without_contrast_data, without_contrast_affine)
with_contrast_downsampled = Nifti1Image(with_contrast_data, with_contrast_affine)
segmentation_downsampled = Nifti1Image(segmentation_data, segmentation_affine)

save(without_contrast_downsampled, output_directory + '/without_contrast_downsampled.nii')
save(with_contrast_downsampled, output_directory + '/with_contrast_downsampled.nii')
save(segmentation_downsampled, output_directory + '/segmentation_downsampled.nii')
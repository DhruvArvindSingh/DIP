import SimpleITK as sitk
import matplotlib.pyplot as plt

# Load images
fixed = sitk.ReadImage('fixedImage.png', sitk.sitkFloat32)
moving = sitk.ReadImage('movingImage.png', sitk.sitkFloat32)

# Initialize registration framework
registration = sitk.ImageRegistrationMethod()

# Similarity metric (equivalent to MATLAB mutual information)
registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

# Interpolator
registration.SetInterpolator(sitk.sitkLinear)

# Optimizer (same as MATLAB gradient descent)
registration.SetOptimizerAsRegularStepGradientDescent(
    learningRate=1.0,
    minStep=1e-4,
    numberOfIterations=300,
    gradientMagnitudeTolerance=1e-8
)

# Affine transform (like MATLAB 'affine')
transform = sitk.CenteredTransformInitializer(
    fixed,
    moving,
    sitk.AffineTransform(2),    # 2D affine
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)

registration.SetInitialTransform(transform, inPlace=False)

# Run registration
final_transform = registration.Execute(fixed, moving)

# Apply the transformation to the moving image
registered = sitk.Resample(
    moving,
    fixed,
    final_transform,
    sitk.sitkLinear,
    0.0,
    moving.GetPixelID()
)

# Show result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Fixed Image")
plt.imshow(sitk.GetArrayViewFromImage(fixed), cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Registered Image")
plt.imshow(sitk.GetArrayViewFromImage(registered), cmap='gray')

plt.show()
